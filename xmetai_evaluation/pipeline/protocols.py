# -*- coding: utf-8 -*-
"""验证协议：决定样本空间怎么遍历、配对怎么做。

一个协议只回答三个问题：

    prepare()      这次评测要先把哪些数据准备好（读观测、建索引）；
    samples()      要遍历哪些样本，每个样本对应长表里的哪个结果键；
    build_batch()  这个样本怎么变成可计算的配对数据。

样本循环在 ``execution/executor.py``、状态合并与落盘在 ``pipeline/runner.py``，
协议里不允许出现它们。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import DataIndex, DataRequest, EvaluationBatch
from xmetai_evaluation.core.errors import ConfigError
from xmetai_evaluation.pipeline.matcher import Matcher, add_derived_variables
from xmetai_evaluation.pipeline.spec import PipelineSpec, daily_times, hourly_times
from xmetai_evaluation.pipeline.typhoon import diagnose_track, match_errors
from xmetai_evaluation.transforms.regrid import align_lon_to, align_to_grid
from xmetai_evaluation.transforms.temporal import window_sum_at

log = logging.getLogger(__name__)

#: 作用于 DataBundle 的变换（在协议里按声明顺序执行）
BUNDLE_TRANSFORMS = ("ensemble_mean",)


def station_observation_span(
    init_times: List[datetime],
    offset_hours: float,
    window_hours: int,
    lead_max_hours: int,
) -> Tuple[datetime, datetime]:
    """站点观测需要覆盖的时间窗（协议与执行计划层共用的唯一口径）。

    最早起报的窗口起点在 valid - window + 1h，最晚起报的最长时效是收尾。
    """
    offset = timedelta(hours=offset_hours)
    return (
        min(init_times) + offset + timedelta(hours=1 - window_hours),
        max(init_times) + offset + timedelta(hours=lead_max_hours),
    )


def typhoon_storm_filter(options: Dict[str, Any]) -> Optional[List[str]]:
    """要评哪几号台风：``storm_ids``（列表）或 ``tcid``（单号），都不写 = 全部。

    协议和执行计划层共用这一份解析——两边各写一遍，迟早出现"计划层按全部
    台风切块、协议只跑其中一号"这种不报错的白跑。
    """
    wanted = options.get("storm_ids")
    if wanted is None and options.get("tcid") is not None:
        wanted = [options.get("tcid")]
    if not wanted:
        return None
    return [str(value) for value in wanted]


def babj_seed_moment(
    moments: Iterable[datetime],
    init_bjt: datetime,
    min_offset_hours: float = 6.0,
    step_hours: float = 6.0,
) -> Optional[datetime]:
    """链式诊断的起始中心该取哪一条实况；这个起报不是场次时返回 ``None``。

    取**起报后、时效网格上的第一条实况**（不早于「起报 + min_offset_hours」，
    且时刻正好落在 ``init_bjt + 时效`` 的网格上）；一条都没有就返回 ``None``，
    调用方跳过这个场次。

    为什么起点不是起报时刻本身，而是往后一个时效：链的种子要保证**第一条
    时效能配上实况**。报文里的分析场是 6 小时一个（北京时 02/08/14/20），
    台风当天的第一条定位常常落在 20:00（12 UTC）——拿 08:00 当种子的话，
    那条实况永远不是任何时效的验证时刻（第一个时效 valid = 14:00），整条
    曲线会一个实况都配不上，白出一个全空场次。从第一个时效的 valid 时刻起
    找，就不会有这个问题。

    **种子必须正好落在某个时效的验证时刻上**：验证时刻是 ``init_bjt + 时效``
    （``step_hours`` 网格），而报文里存在 17:00/05:00 这类**不在网格上**的
    记录（观测段有 3 小时一条的时候）。归档（``tc/fuxi``）的口径是"第一条
    **能配上时效**的实况"——非网格记录跳过、取网格上的第一条。不跳的后果是
    init_pos 与归档差一条记录：2025 年 2507@0722 的 17:00 非网格、20:00 才是
    第一条能配的（两者相距 ~1.3°），逐位对拍就差这一场。

    默认 6.0 就是第一个时效的步长，与标准归档（``tc/fuxi``）同口径：那边
    的 ``init_pos`` 实测恒等于"起报后第一条能配上时效的实况"。改用起报时刻
    本身当种子会让绝大多数场次的种子前移约 6 小时（中位 ~100km），虽然被
    ±4° 的早期搜索框吸收、曲线多半仍然一致，但边界场次（台风刚生成/快消散、
    msl 场平缓）会静默对不上。

    **起报日必须落在该台风的生命史日期窗内**（首条实况的日期 ≤ 起报北京时
    日期 ≤ 末条实况的日期），否则返回 ``None``：这个起报不是这号台风的场次。
    没有这道窗，"只往后找"会把生成前的起报日全部放行——起报之后总能找到
    第一条实况（哪怕在几天后的生成时刻），一号台风于是凭空多出从年初到它
    消散的每一天场次，预报侧还有一条全无实况可配的"曲线"。标准归档
    （``tc/fuxi``）的口径就是"起报日落在生命史日期窗内"，这里按它复现
    （2025 年 157 个场次实测逐一相同）。

    只往后找，不往前找：往前找等于用"预报还没起步"的位置当起点，时效与
    实况对不上。

    ``moments`` 可以是某号台风的 ``{北京时: 记录}`` 字典（协议侧），也可以
    是某号台风实况时刻的集合（计划层——**必须逐号传，不能传并集**：并集的
    日期窗会把各台风之间的空档日也放行）。
    """
    moments = set(moments)
    if not moments:
        return None
    first, last = min(moments), max(moments)
    if not (first.date() <= init_bjt.date() <= last.date()):
        return None
    if min_offset_hours <= 0:
        # 0 = 种子就取起报时刻那条；等价于"起点不往后挪"
        return init_bjt if init_bjt in moments else None
    step = timedelta(hours=float(step_hours))
    limit = init_bjt + timedelta(hours=min_offset_hours)
    later = [
        moment
        for moment in moments
        if moment >= limit and (moment - init_bjt) % step == timedelta(0)
    ]
    return min(later) if later else None


#: 台风链路的协议名（确定性 + 集合）。执行计划层有两处按协议形态分支——起报
#: 时刻由报文反推（``typhoon_track`` 那一支）、观测角色必须是 slice——判的是
#: 名字不是类，所以加新协议时这份清单要跟着一起加，漏了会静默走成站点协议。
TYPHOON_PROTOCOLS = ("typhoon_track", "typhoon_track_ens")


def babj_init_times(
    observation: Any,
    init_times: List[datetime],
    offset_hours: float,
    storm_ids: Optional[List[str]] = None,
    min_offset_hours: float = 6.0,
    step_hours: float = 6.0,
) -> List[datetime]:
    """从 BABJ 报文反推哪些起报时刻对得上实况（协议与执行计划层共用的唯一口径）。

    报文只在北京时的整点上有分析场，而预报的起报时刻是 UTC：一个起报能用，
    当且仅当 ``babj_seed_moment`` 给它找得到起始中心（不是"起报时刻正好有
    实况"，见那个函数）。旧链路就是这么**反推**起报时刻的，而不是先跑一遍
    再靠报错筛掉。

    计划层用它在切块前把对不上的起报日剔出去。不剔的后果不是"少跑几场"：
    台风空档期那些起报日仍会各切出一个块，每块读到 0 个样本、被判失败块，
    manifest 里堆一片「工作块失败」——看着像跑挂了，实际只是那天没有台风。
    同时也白读一遍全部报文。

    ``storm_ids`` 必须参与筛选，否则会漏掉一层：只跑 2501 时，7 月那些"有
    别的台风、没有 2501"的日子照样会切出空块。所以这里**逐号台风**判
    ``babj_seed_moment``（生成前的起报日由那道日期窗剔掉）——判定必须逐号，
    不能把几号台风的实况时刻并起来判：并集的日期窗会把台风之间的空档日
    也放行，切出来的块在协议侧一个场次都建不出，又回到"空块堆进 manifest"。

    ``min_offset_hours`` 必须与协议侧同源（``plan.py`` 两处传的是同一个配置
    项 ``seed_min_offset_hours``）。这里收窄了、协议没收窄，就会静默少场次
    ——正是这个函数跟 ``babj_seed_moment`` 共用一份规则要避免的事。

    ``step_hours`` 同理：时效步长（种子须落在 ``init_bjt + 时效`` 网格上），
    plan 侧从预报布局取（``_layout_step_hours``）。

    报文是 KB 级、数量以十计，这里整目录读一次换准确的起报集，划算。
    """
    request = DataRequest(source_id=observation.source_id, variables=["storm"])
    bundle = observation.reader.read(request, observation.catalog.discover(request))
    payload = bundle.payload
    names = [str(value) for value in payload["storm"].values]
    scope = [
        index
        for index, name in enumerate(names)
        if storm_ids is None or name in storm_ids
    ]
    per_storm: List[set] = []
    for index in scope:
        column = payload["lat"].values[:, index]
        moments = {
            payload["time"].values[k].astype("datetime64[us]").item()
            for k in range(column.size)
            if np.isfinite(column[k])
        }
        if moments:
            per_storm.append(moments)
    offset = timedelta(hours=offset_hours)
    return [
        value
        for value in init_times
        if any(
            babj_seed_moment(
                moments, value + offset, min_offset_hours, step_hours
            )
            is not None
            for moments in per_storm
        )
    ]


def sample_leads(available_leads: Sequence[float], window_hours: float) -> List[float]:
    """可评的采样时效集：只取**完整累积窗**的末端时效。

    规则只此一份——计划层（切时效窗）与协议（遍历样本）都调它，避免两处各写
    一遍导致口径漂移。约束对没有累积变换的流程同样成立：``window_hours`` 以内
    的时效不是可评的窗口末端（预报已是 6h 累积量时 ``window_hours=6`` 只保留
    6 的倍数，天然排掉 0 时效的瞬时场，否则会与 6h 累积实况配出垃圾分）。
    """
    span = float(window_hours)
    return [
        float(lead)
        for lead in available_leads
        if float(lead) >= span and abs(float(lead) % span) < 1e-6
    ]


def select_observation_files(
    observation: Any, observation_var: str, start: datetime, end: datetime
) -> Tuple[DataRequest, List[Any]]:
    """按时间窗从 catalog 挑观测文件（协议直读与加载策略层共用）。"""
    request = DataRequest(
        source_id=observation.source_id, variables=[observation_var]
    )
    catalog_files = observation.catalog.discover(request).available
    selected = [
        path
        for path in catalog_files
        if start <= observation.catalog.file_time(path) <= end
    ]
    selected.sort(key=observation.catalog.file_time)
    return request, selected


def _never_requires_members() -> bool:
    """没有 needs_members() 的指标（如自定义实现）默认不需要成员。"""
    return False


def parse_regions(raw: Any) -> List[Tuple[str, float, float]]:
    """解析 ``options["regions"] = {带名: {"lat_min": 南界, "lat_max": 北界}}``。

    带名进结果坐标（长表里现成的 ``region`` 列）、边界闭区间。只有显式
    声明才分区——不提供任何"预设带"，要哪几条带写在配置里一目了然。

    格点与站点两条协议共用：解析和校验只该有一份，复制出去的副本迟早
    只在一边上修。**闭区间是刻意的**，闭区间下相邻带在公共边界上重叠
    （如 ``±20`` 与 ``20-90`` 都含 20.0），改成一端开会让两类产物的分带
    口径不一致。带怎么挡由各自的 ``restrict_to_latitude_band`` 决定。
    """
    if not raw:
        return []
    if not isinstance(raw, dict):
        raise ConfigError(
            "regions 必须是 {带名: {'lat_min': 南界, 'lat_max': 北界}} 字典，"
            f"收到 {type(raw).__name__}"
        )
    parsed: List[Tuple[str, float, float]] = []
    for name, bounds in raw.items():
        try:
            lat_min = float(bounds["lat_min"])  # type: ignore[index]
            lat_max = float(bounds["lat_max"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"纬度带 {name!r} 需要数值 lat_min / lat_max，收到 {bounds!r}"
            ) from exc
        if lat_min >= lat_max:
            raise ConfigError(
                f"纬度带 {name!r} 的 lat_min({lat_min}) 必须小于 lat_max({lat_max})"
            )
        parsed.append((str(name), lat_min, lat_max))
    return parsed


def regions_summary(regions: Sequence[Tuple[str, float, float]]) -> List[str]:
    """带列表 → manifest 里的 ``"tropics[-20, 20]"`` 形式（空就不写这个键）。"""
    return [f"{name}[{lat_min:g}, {lat_max:g}]" for name, lat_min, lat_max in regions]


@dataclass
class Sample:
    """一个待评测样本：结果键 + 长表坐标 + 协议内部负载。"""

    key: Tuple
    coordinates: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineContext:
    """执行上下文：数据源、变换链、指标由执行器组装后交给协议。"""

    spec: PipelineSpec
    forecast: Any
    observation: Any
    reference: Any = None
    transforms: Dict[str, Any] = field(default_factory=dict)
    metrics: List[Any] = field(default_factory=list)
    #: 数据加载策略层（分块执行时由执行器注入；None 表示协议自己直读）
    loader: Any = None

    def transform(self, name: str) -> Any:
        return self.transforms.get(name)


class Protocol(ABC):
    """验证协议基类。"""

    name = "protocol"

    #: 跨工作块出现重复结果键时的合并语义（执行层读，协议自己不用）：
    #:   append  同键的状态追加（如 station 协议：同 lead 下多个起报累积）；
    #:   replace 后块覆盖前块（如 grid 协议：同一 valid_time 只认最新起报）。
    duplicate_key_policy = "append"

    def __init__(self, spec: PipelineSpec):
        self.spec = spec

    def prepare(self, context: PipelineContext) -> None:
        """准备一次性的数据（观测、索引等）。"""

    @abstractmethod
    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        """产出样本；生成器内部可以流式读取预报，控制内存峰值。"""

    @abstractmethod
    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        """把样本配对成 EvaluationBatch；返回 None 表示这个样本跳过。"""

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        """长表的兜底坐标。"""
        return {}

    def summary(self) -> Dict[str, Any]:
        """追加进 manifest 的协议统计。"""
        return {}


class StationValidTimeProtocol(Protocol):
    """站点观测与格点预报按有效时刻配对。

    - 样本空间：每个起报的每个完整窗口时效（window, 2*window, ...）；
    - 结果键：``("lead_h", lead)``；
    - 观测：一次读入后按 [valid - window + 1h, valid] 闭区间累积；
    - 空间：预报双线性插值到站点。
    """

    name = "station_valid_time"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        self.forecast_var = spec.variables.get("forecast") or spec.forecast.params.get("variable")
        self.observation_var = spec.variables.get("observation") or spec.observation.params.get(
            "variable"
        )
        if not self.forecast_var or not self.observation_var:
            raise ConfigError(
                "station_valid_time 协议需要在配置里声明 forecast/observation 的 variable"
            )
        self.window_hours = spec.window_hours
        self.sample_unit = str(spec.options.get("sample_unit", "station"))
        self.init_times: List[datetime] = []
        self.forecast_reader = None
        self.forecast_request: Optional[DataRequest] = None
        self.forecast_index: Optional[DataIndex] = None
        self.observation_bundle = None
        self.observation_ds: Optional[xr.Dataset] = None
        self.station_lats: Optional[np.ndarray] = None
        self.station_lons: Optional[np.ndarray] = None
        self.window_leads: Optional[List[float]] = None
        #: 预报读取失败的起报（ISO 串）。报列表不报计数：切时效后同一个起报
        #: 会出现在多个块里，跨块相加会把一个坏起报数成好几个。
        self.skipped_init_times: List[str] = []
        self.ref_reader = None
        #: 分纬度带评估：[(带名, 南界, 北界)]，来自 options["regions"]；
        #: 空 = 不分区，输出与没有这个能力之前逐行一致。与上面 ``prepare()``
        #: 里那个单区域筛选 ``options["region"]`` 是**两回事**：那个把站点整体
        #: 裁到 lat/lon 盒子里（永久改 ``observation_ds``），这个是在裁完的站点
        #: 上再逐带出样本。同时配两种时注意：全球行的 ``region`` 会兜底成单区域
        #: 的名字（见 ``defaults()``），于是按「region 为空 = 全球」取数的下游
        #: 会取不到那行——要么只配 ``regions``，要么在配置里给单区域一个名字前
        #: 想清楚这一点。
        self.regions = parse_regions(spec.options.get("regions"))
        #: ``build_batch`` 的单条缓存，键 ``(起报, 时效)``；见那里的注释。
        #: 值可能是 ``None``（该 (起报, 时效) 一个有效站都没有），所以配一个
        #: 独立的哨兵键而不是判空。
        self._batch_cache_key: Any = None
        self._batch_cache: Optional[EvaluationBatch] = None
        #: 每块前 N 个样本打一条"样本诊断"（形状 + 有效站数），用来定位"样本
        #: 出不来"。**它是块口径、不是 run 口径**：这个协议实例每个工作块重建
        #: 一次（``executor.collect_chunk``），所以实际条数是 N × 块数——全年那趟
        #: 五千多块就是一万六千多行。默认关掉，排查样本层问题时再改回 3。
        self._diagnostics_left = 0

    def prepare(self, context: PipelineContext) -> None:
        forecast = context.forecast
        observation = context.observation
        self.forecast_reader = forecast.reader

        # 起报：显式声明优先（分块执行时计划层按块注入），否则按评测时段探测
        init_times = [
            datetime.fromisoformat(str(value))
            for value in self.spec.forecast.params.get("init_times", [])
        ]
        if not init_times:
            start, end = self.spec.period()
            discovered = forecast.catalog.discover(
                DataRequest(
                    source_id=forecast.source_id,
                    variables=[self.forecast_var],
                    init_times=daily_times(start, end),
                )
            )
            init_times = forecast.reader.available_init_times(discovered)
            if not init_times:
                raise ValueError(f"时间范围 {start} 到 {end} 内没有可用预报起报时间")
        if self.spec.limit:
            init_times = init_times[: self.spec.limit]
        self.init_times = init_times
        # 逐块流水（每块一次）：INFO 留给块级进度，这些细节归 DEBUG，
        # 需要时用 --log-file（文件 handler 是 DEBUG）全量取回。
        log.debug(
            "起报时间: %s 到 %s，共 %d 个",
            init_times[0],
            init_times[-1],
            len(init_times),
        )

        self.forecast_request = DataRequest(
            source_id=forecast.source_id,
            variables=[self.forecast_var],
            init_times=init_times,
            lead_times=self.spec.forecast.params.get("lead_times"),
        )
        self.forecast_index = forecast.catalog.discover(self.forecast_request)
        lead_max = int(forecast.reader.max_lead_hours(self.forecast_index))
        log.debug("预报文件发现完成：%d 个起报，最大时效 %sh", len(init_times), lead_max)

        # 观测：一次读取覆盖所有起报和时效，后续窗口只做内存索引。
        # 走加载策略层时（resident）整个 run 只读一次，块间共享同一份缓存。
        obs_start, obs_end = station_observation_span(
            init_times,
            self.spec.local_utc_offset_hours,
            self.window_hours,
            lead_max,
        )
        if context.loader is not None:
            self.observation_bundle = context.loader.materialize(
                "observation",
                DataRequest(
                    source_id=observation.source_id,
                    variables=[self.observation_var],
                    init_times=hourly_times(obs_start, obs_end),
                ),
            )
        else:
            obs_request, selected = select_observation_files(
                observation, self.observation_var, obs_start, obs_end
            )
            log.info(
                "观测窗口覆盖 %s 到 %s，共 %d 个文件",
                obs_start,
                obs_end,
                len(selected),
            )
            if not selected:
                raise ValueError("没有找到评估所需的观测文件")
            self.observation_bundle = observation.reader.read(
                obs_request,
                DataIndex(source_id=observation.source_id, available=selected),
            )
        self.observation_ds = self.observation_bundle.payload
        self.region = self.spec.options.get("region")
        if self.region:
            lat_range = self.region.get("lat", (-90.0, 90.0))
            lon_range = self.region.get("lon", (0.0, 360.0))
            lats = self.observation_ds["lat"].values
            lons = self.observation_ds["lon"].values
            keep = (
                (lats >= lat_range[0])
                & (lats <= lat_range[1])
                & (lons >= lon_range[0])
                & (lons <= lon_range[1])
            )
            self.observation_ds = self.observation_ds.isel(station=keep)
            log.info("区域筛选 %s 后站点数: %d", self.region, int(keep.sum()))
        self.station_lats = self.observation_ds["lat"].values
        self.station_lons = self.observation_ds["lon"].values
        self.station_weights = None
        if str(self.spec.options.get("weights", "none")) == "cos_lat":
            self.station_weights = np.cos(np.deg2rad(np.abs(self.station_lats)))
        log.debug("观测读取完成: %s", self.observation_ds.sizes)

        # 外部 BSS 气候概率参考（可选）：站点协议只认 ref_probability 这类逐站参考
        if context.reference is not None:
            self.ref_reader = getattr(context.reference, "reader", None)
            if self.ref_reader is None or not hasattr(self.ref_reader, "probabilities"):
                log.warning(
                    "站点协议忽略参考源 %s：不是逐站气候概率参考（需 ref_probability）",
                    context.reference.source_id,
                )
                self.ref_reader = None

    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        accumulator = context.transform("time_window_accumulator")
        reader = context.forecast.reader
        # 只有声明了需要成员的概率指标时，才额外保留成员级窗口场（多约 1/5 内存）
        keep_members = any(
            getattr(metric, "needs_members", _never_requires_members)()
            for metric in (context.metrics or [])
        )
        # 这一块到底产出了什么：一个样本都没产出时，块层面只会记成「没有成功
        # 处理任何评测批次」，原因得由这里说清楚。
        emitted = 0
        seen_leads: set = set()
        read_failures = 0

        for init_time in self.init_times:
            try:
                bundle = reader.read_one(
                    self.forecast_request, self.forecast_index, init_time
                )
                raw_field = bundle.payload[self.forecast_var].isel(init_time=0)
                bundle = self._apply_bundle_transforms(context, bundle)
                field = bundle.payload[self.forecast_var].isel(init_time=0)
                # 预报已是窗口累积量（如 6h 降水文件）时无需再累积
                windows = accumulator.transform(field) if accumulator is not None else field
                if "lead_time" in windows.coords:
                    seen_leads.update(
                        float(value) for value in windows.lead_time.values
                    )
                member_windows = None
                if keep_members:
                    if "member" in windows.dims:
                        member_windows = windows
                    elif "member" in raw_field.dims:
                        member_windows = (
                            accumulator.transform(raw_field) if accumulator is not None else raw_field
                        )
            except Exception as exc:
                self.skipped_init_times.append(init_time.isoformat())
                read_failures += 1
                log.exception("起报 %s 预报读取失败: %s", init_time, exc)
                continue

            if self.window_leads is None:
                # 分块执行时计划层按块声明采样时效（它已按采样格点切窗，是
                # (起报, 时效) 的纯划分）；没声明就按完整窗口末端自己推。
                # 这里刻意**不**与本次 init 的可用轴求交：这个列表只算一次，
                # 拿首个 init 的轴去交会把后面 init 本可评的时效静默截短。
                declared = self.spec.forecast.params.get("sample_leads")
                self.window_leads = (
                    [float(lead) for lead in declared]
                    if declared
                    else sample_leads(windows.lead_time.values, self.window_hours)
                )
                log.debug("评估时效: %s", self.window_leads)
                log.debug("站点数量: %d", len(self.station_lats))

            for lead in self.window_leads:
                if lead not in windows.lead_time.values:
                    continue
                emitted += 1
                lead_key = int(lead) if float(lead).is_integer() else float(lead)
                base_coordinates = {
                    "lead_h": lead_key,
                    "window_h": self.window_hours,
                    "variable": self.observation_var,
                    "sample_unit": self.sample_unit,
                }
                base_payload = {
                    "init_time": init_time,
                    "lead": lead,
                    "windows": windows,
                    "member_windows": member_windows,
                }
                # 全球行不带 region 坐标——不分区时的输出行和这个能力出现之前
                # 逐行一致（同 GridValidTimeProtocol.samples）。
                yield Sample(
                    key=("lead_h", lead_key),
                    coordinates=base_coordinates,
                    payload=base_payload,
                )
                # 每个纬度带再各出一个样本：键尾追加带名，region 进结果坐标
                # （长表现成的 region 列）。带掩码不在 build_batch 里叠——与
                # 格点同口径，执行层统一按 payload 里的 region_bounds 挡：
                # 站点批次的掩码维度是 station，由执行层按站点纬度切。
                for name, lat_min, lat_max in self.regions:
                    yield Sample(
                        key=("lead_h", lead_key, name),
                        coordinates={**base_coordinates, "region": name},
                        payload={
                            **base_payload,
                            "region_bounds": (lat_min, lat_max),
                        },
                    )

        if emitted == 0:
            # 一个样本都没产出：块会记成「没有成功处理任何评测批次」。原因无非
            # 两种——声明要评的时效根本没被累积器产出（多因该起报的预报文件不
            # 够窗口所需步数，且预报侧静默丢了帧），或者起报在读取阶段就全失败
            # 了（那种情况上面已有 ERROR）。这里把两者一起打出来，省得再猜。
            log.warning(
                "起报窗口无样本可评：声明采样时效 %s，累积器只产出时效 %s，"
                "预报读取失败 %d 个起报",
                self.window_leads,
                sorted(seen_leads)[:20],
                read_failures,
            )

    def _apply_bundle_transforms(self, context: PipelineContext, bundle: Any) -> Any:
        """按声明顺序应用作用于 DataBundle 的变换（如集合降维）。"""
        for item in self.spec.transforms:
            if item.name not in BUNDLE_TRANSFORMS:
                continue
            transform = context.transform(item.name)
            if transform is not None and hasattr(transform, "transform"):
                bundle = transform.transform(bundle)
        return bundle

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        interpolator = context.transform("grid_to_station")
        if interpolator is None:
            raise ValueError("station_valid_time 协议需要 grid_to_station 变换")

        init_time = sample.payload["init_time"]
        lead = sample.payload["lead"]
        windows = sample.payload["windows"]
        # 分纬度带时同一个 (起报, 时效) 连着发 1 + N 个样本（全球 + 每带一个），
        # 而这里每次都要重做「观测窗口求和 + 全场双线性插值到站」——全场插值
        # 是这条流程最贵的一步，不缓存就是白算 N 倍。批次只装站点尺寸的数组
        # （网格场留在 sample.payload 里、不进批次），所以只记上一个就够：
        # ``samples()`` 按 (起报, 时效) 分组顺序发，同组的样本必然相邻，
        # 内存 O(1)。协议实例是**每块重建**的（见 executor.collect_chunk），
        # 这个缓存不跨块。
        cache_key = (init_time, lead)
        if self._batch_cache_key == cache_key:
            return self._batch_cache
        valid_time = init_time + timedelta(
            hours=float(lead) + self.spec.local_utc_offset_hours
        )

        observation = window_sum_at(
            self.observation_ds,
            self.observation_var,
            valid_time,
            self.window_hours,
            time_dim="time",
            require_complete=str(
                self.spec.options.get("observation_window", "complete")
            ) != "reference",
        )
        forecast_at_lead = windows.sel(lead_time=lead)
        forecast_at_station = interpolator.transform(
            forecast_at_lead, self.station_lats, self.station_lons
        )
        members = None
        member_windows = sample.payload.get("member_windows")
        if member_windows is not None:
            members = interpolator.transform(
                member_windows.sel(lead_time=lead), self.station_lats, self.station_lons
            )
        valid_mask = xr.DataArray(
            np.isfinite(forecast_at_station.values) & np.isfinite(observation.values),
            dims=forecast_at_station.dims,
            coords=forecast_at_station.coords,
        )
        weights = None
        if self.station_weights is not None:
            weights = xr.DataArray(
                self.station_weights,
                dims=["station"],
                coords={"station": forecast_at_station["station"].values},
            )
        reference = None
        if self.ref_reader is not None:
            station_ids = self.observation_ds["station"].values
            probs = self.ref_reader.probabilities(valid_time, station_ids)  # (n, 4)
            reference = xr.DataArray(
                probs,
                dims=("station", "threshold"),
                coords={
                    "station": station_ids,
                    "threshold": list(self.ref_reader.thresholds),
                },
            )
        if int(valid_mask.sum()) == 0:
            # 「这个 (起报, 时效) 没样本」同样要记进缓存，否则带样本会把这套
            # 计算再白做 N 遍
            self._batch_cache_key, self._batch_cache = cache_key, None
            return None
        if self._diagnostics_left > 0:
            self._diagnostics_left -= 1
            log.info(
                "样本诊断 init=%s lead=%s forecast_shape=%s obs_shape=%s n_valid=%d",
                init_time,
                lead,
                forecast_at_station.shape,
                observation.shape,
                int(valid_mask.sum()),
            )

        batch = EvaluationBatch(
            forecast=forecast_at_station,
            observation=observation,
            sample_keys=[
                {"init": init_time.isoformat(), "lead": float(lead)}
            ]
            * len(self.station_lats),
            valid_mask=valid_mask,
            members=members,
            reference=reference,
            weights=weights,
            sample_dim=self.sample_unit,
            alignment={
                "method": "bilinear",
                "window_hours": self.window_hours,
                "valid_time": valid_time.isoformat(),
                "timezone_offset_hours": self.spec.local_utc_offset_hours,
            },
            protocol_id=self.name,
        )
        self._batch_cache_key, self._batch_cache = cache_key, batch
        return batch

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "variable": self.observation_var,
            "sample_unit": self.sample_unit,
            "window_h": self.window_hours,
            "unit": self._unit(),
            "region": self.region.get("name", "") if self.region else "",
        }

    def _unit(self) -> str:
        """评分单位取自观测语义（预报与观测在配对时已要求同单位）。"""
        bundle = self.observation_bundle
        if bundle is None:
            return ""
        return str((bundle.semantic.units or {}).get(self.observation_var, ""))

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            # 报时刻列表而不是计数：切时效后同一个起报会出现在多个块里，
            # 跨块相加会放大（归并时列表取并集，计数由列表长度得出）
            "init_times": [value.isoformat() for value in self.init_times],
            "skipped_init_times": list(self.skipped_init_times),
            "window_hours": self.window_hours,
        }
        if self.regions:
            summary["regions"] = regions_summary(self.regions)
        if self.forecast_index is not None:
            summary["forecast_input_files"] = self._forecast_input_files()
        if self.observation_bundle is not None:
            summary["observation_input_files"] = list(
                self.observation_bundle.provenance.input_files
            )
        return summary

    def _forecast_input_files(self) -> List[str]:
        """预报输入文件清单（由 Reader 从索引还原，不依赖索引内部形状）。"""
        reader = getattr(self.forecast_reader, "input_files", None)
        if reader is None or self.forecast_index is None:
            return []
        return reader(self.forecast_index)


class GridValidTimeProtocol(Protocol):
    """格点预报与格点实况按时间配对（连续场评测）。

    采样键口径由流程参数 ``sample_by`` 决定：

    - ``valid_time``（缺省）：样本空间是预报 (init_time, lead_time) 展平出的
      有效时刻，**一个有效时刻一个样本**——多个起报够到同一时刻时只留最新
      起报；结果键 ``("valid_time", 有效时刻)``。
    - ``init_lead``：**每个 (起报, 时效) 各出一个样本**，同一有效时刻有几个
      起报就有几个样本（逐起报报满整段时效用它）；结果键
      ``("init_lead", 起报, 时效)``。

    其余相同：起报显式声明 ``init_times`` 时用它，否则按 ``start_date``/
    ``end_date`` 探测预报源里实际可用的起报（与 ``station_valid_time`` 同一
    口径）；集合降维由声明的 ``ensemble_mean`` 变换完成；空间上由 Matcher
    把预报插值到实况网格。
    """

    name = "grid_valid_time"

    #: 同一 valid_time 可能被多个起报够到（如逐日起报 + 长时效重叠）。
    #: 单段执行时 Matcher 内部"最新起报获胜"；分块执行下由执行层按块序
    #: 覆盖重现同一语义：同键只保留时间上最后一块（= 最新起报）的状态。
    #: ``init_lead`` 口径下键是 (起报, 时效)、不再撞车，``__init__`` 会把它
    #: 改成 append——那时再按同键覆盖就是把别的起报的样本丢掉。
    duplicate_key_policy = "replace"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        #: 采样键口径，见类 docstring；缺省 valid_time，行为与加这个开关之前一致。
        self.sample_by = str(spec.options.get("sample_by", "valid_time") or "valid_time")
        if self.sample_by not in ("valid_time", "init_lead"):
            raise ConfigError(
                f"sample_by 只能是 valid_time 或 init_lead，收到 {self.sample_by!r}"
            )
        self._init_lead = self.sample_by == "init_lead"
        if self._init_lead:
            self.duplicate_key_policy = "append"
        self.forecast_vars = _variable_list(spec.forecast.params, "forecast")
        self.observation_vars = _variable_list(spec.observation.params, "observation")
        self.forecast_bundle = None
        self.member_bundle = None
        self.observation_bundle = None
        self.reference_bundle = None
        #: 需要成员场的指标所路由到的变量；空表示这次评测不取成员
        self.member_vars: List[str] = []
        self.batches: Dict[Any, EvaluationBatch] = {}
        #: 分纬度带评估：[(带名, 南界, 北界)]，来自 options["regions"]；
        #: 空 = 不分区，输出与没有这个能力之前逐行一致
        self.regions = parse_regions(spec.options.get("regions"))

    def prepare(self, context: PipelineContext) -> None:
        forecast = context.forecast
        observation = context.observation
        # 起报：配置显式声明优先；没声明就按评测时段探测可用起报
        # （与 station_valid_time 同一口径，配置只需给 start_date/end_date）
        init_times = [
            datetime.fromisoformat(str(value))
            for value in self.spec.forecast.params.get("init_times", [])
        ]
        if not init_times:
            start, end = self.spec.period()
            discovered = forecast.catalog.discover(
                DataRequest(
                    source_id=forecast.source_id,
                    variables=self.forecast_vars,
                    init_times=daily_times(start, end),
                )
            )
            init_times = forecast.reader.available_init_times(discovered)
            if not init_times:
                raise ValueError(f"时间范围 {start} 到 {end} 内没有可用预报起报时间")
        if self.spec.limit:
            init_times = init_times[: self.spec.limit]
        # 逐块流水（每块一次）：INFO 留给块级进度，这些细节归 DEBUG，
        # 需要时用 --log-file（文件 handler 是 DEBUG）全量取回。
        log.debug(
            "起报时间: %s 到 %s，共 %d 个", init_times[0], init_times[-1], len(init_times)
        )
        # ``or []``：键写了但值是 None（"不限制时效"）时也要当"没声明"。
        lead_times = [
            float(value) for value in self.spec.forecast.params.get("lead_times") or []
        ]
        # 采样时效（窗口内要出分的样本）与读取时效（可能带累积预热）是两回事：
        # 分块执行时计划层只声明前者，后者是 lead_times。
        sample_lead_times = [
            float(value) for value in self.spec.forecast.params.get("sample_leads", [])
        ]

        forecast_request = DataRequest(
            source_id=forecast.source_id,
            variables=self.forecast_vars,
            init_times=init_times,
            lead_times=lead_times or None,
        )
        bundle = forecast.reader.read(
            forecast_request, forecast.catalog.discover(forecast_request)
        )
        raw_bundle = bundle
        # 派生风速（ws* = sqrt(u²+v²)）必须在 ensemble_mean **之前**、member 维还在时
        # 合成：逐成员开方再平均才是集合平均风速 E[ws]（参考实现 regr_ens.py:698 的
        # 口径）；平均之后再开方会把集合离散度折进风速（Jensen 间隙，长时效虚高
        # 8%~13%）。确定性流程没有 member 维，这里不合成、留给 Matcher 兜底
        # （matcher.py 的 add_derived_variables 对已合成的名字是空操作）。
        if (
            isinstance(bundle.payload, xr.Dataset)
            and "member" in bundle.payload.dims
        ):
            bundle.payload = add_derived_variables(
                bundle.payload, self.forecast_vars
            )
        for item in self.spec.transforms:
            transform = context.transform(item.name)
            if transform is not None and hasattr(transform, "transform"):
                bundle = transform.transform(bundle)
        self.forecast_bundle = bundle
        log.debug("预报读取完成: %s", bundle.payload.dims)

        # 概率/集合类指标需要原始成员：集合平均是把成员降到均值，不能替代成员
        keep_members = any(
            getattr(metric, "needs_members", _never_requires_members)()
            for metric in (context.metrics or [])
        )
        self.member_vars = self._member_variables(context) if keep_members else []
        if self.member_vars and "member" in raw_bundle.payload.dims:
            self.member_bundle = raw_bundle

        # 用**采样**时效而不是读取时效：读取集可能带累积预热时效，把预热时效也
        # 算进 valid_time 会让 matcher 多配出批次，那些批次既不是本窗的样本、
        # 又会跨窗重复，破坏采样集的纯划分。
        valid_times = sorted(
            {
                init_time + timedelta(hours=float(lead))
                for init_time in init_times
                for lead in (
                    sample_lead_times
                    or lead_times
                    or self._available_leads(bundle.payload)
                )
            }
        )
        observation_request = DataRequest(
            source_id=observation.source_id,
            variables=self.observation_vars,
            init_times=valid_times,
        )
        if context.loader is not None:
            # resident/window 时整段共享缓存，逐块只取自己跨度内的切片
            self.observation_bundle = context.loader.materialize(
                "observation", observation_request
            )
        else:
            self.observation_bundle = observation.reader.read(
                observation_request, observation.catalog.discover(observation_request)
            )
        log.debug("实况读取完成: %s", self.observation_bundle.payload.dims)

        if context.reference is not None:
            reference_request = DataRequest(
                source_id=context.reference.source_id,
                variables=self.observation_vars,
                init_times=valid_times,
            )
            if context.loader is not None:
                self.reference_bundle = context.loader.materialize(
                    "reference", reference_request
                )
            else:
                self.reference_bundle = context.reference.reader.read(
                    reference_request,
                    context.reference.catalog.discover(reference_request),
                )
            log.debug("气候态参考读取完成: %s", self.reference_bundle.payload.dims)

        matcher = Matcher(
            ensemble_reduction=str(self.spec.options.get("ensemble_reduction", "mean"))
        )
        for batch in matcher.match(
            forecast=self.forecast_bundle,
            observation=self.observation_bundle,
            variables=self.forecast_vars,
            sample_by=self.sample_by,
        ):
            batch.members = self._members_for(batch)
            batch.reference = self._reference_for(batch)
            self.batches[self._batch_key(batch)] = batch
            if len(self.batches) % 20 == 0:
                log.info("参考场装配进度 %d", len(self.batches))
        log.debug("配对完成：%d 个批次", len(self.batches))

    def _batch_key(self, batch: EvaluationBatch) -> Any:
        """批次在 ``self.batches`` 里的键，口径由 ``sample_by`` 决定。"""
        record = batch.sample_keys[0] if batch.sample_keys else {}
        if self._init_lead:
            return (str(record.get("init_time", "")), float(record.get("lead_h") or 0.0))
        return record.get("valid_time")

    @staticmethod
    def _available_leads(payload: xr.Dataset) -> List[float]:
        """没声明 ``lead_times`` 时，按预报文件自带的时效展开（单位：小时）。

        与 ``station_valid_time`` 同一口径（那边也是读 ``windows.lead_time.values``）。
        这里不能退化成 ``[0.0]``：那样实况和气候态只会被请求一个有效时刻，
        Matcher 随之只配出一个批次，评测静默缩水成 1/N 且不报错。
        """
        if "lead_time" not in payload.coords:
            return [0.0]
        return [float(value) for value in payload["lead_time"].values]

    def _member_variables(self, context: PipelineContext) -> List[str]:
        """需要成员场的指标路由到了哪些变量。

        只给这些变量取成员，而不是全部预报变量：成员场按 (init, lead) 展开后
        很大，且会一直留在 ``self.batches`` 里。
        """
        routed: List[str] = []
        for item, metric in zip(self.spec.metrics, context.metrics or []):
            if not getattr(metric, "needs_members", _never_requires_members)():
                continue
            for name in item.params.get("variables") or self.forecast_vars:
                if name not in routed:
                    routed.append(name)
        return routed

    def _members_for(self, batch: EvaluationBatch) -> Optional[xr.Dataset]:
        """把成员场按样本的 init/lead 取出来，插值到实况网格。

        始终返回 Dataset（哪怕只有一个变量），好让 ``narrow_batch`` 能按变量取用——
        返回裸 DataArray 会在路由到别的变量时被误用。
        """
        if self.member_bundle is None or not batch.sample_keys:
            return None
        available = [
            name for name in self.member_vars if name in self.member_bundle.payload.data_vars
        ]
        if not available:
            return None
        record = batch.sample_keys[0]
        try:
            field = self.member_bundle.payload[available].sel(
                init_time=np.datetime64(record["init_time"]),
                lead_time=float(record["lead_h"]),
            )
        except Exception as exc:
            log.debug("成员场取用失败（%s）：%s", record, exc)
            return None
        if "member" not in field.dims:
            return None
        # 对齐目标取**批次里的实况**，不是原始 bundle（理由同 ``_reference_for``）
        target = batch.observation
        aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
        if aligned is not None:
            return aligned
        return field.interp(lat=target["lat"], lon=target["lon"])

    def _reference_for(self, batch: EvaluationBatch) -> Optional[xr.Dataset]:
        """取该有效时刻的气候态参考场，插值到实况网格。

        同样返回 Dataset（全部观测变量），由 ``narrow_batch`` 按变量取用。
        """
        if self.reference_bundle is None or not batch.sample_keys:
            return None
        available = [
            name
            for name in self.observation_vars
            if name in self.reference_bundle.payload.data_vars
        ]
        if not available:
            return None
        record = batch.sample_keys[0]
        try:
            field = self.reference_bundle.payload[available].sel(
                valid_time=np.datetime64(record["valid_time"])
            )
        except Exception as exc:
            log.debug("气候态取用失败（%s）：%s", record, exc)
            return None
        # 对齐目标取**批次里的实况**，不是原始 bundle：Matcher 会把实况经度折算到
        # 预报那一圈（ERA5 的 -180..180 -> 0..360），原始 bundle 仍是折算前的。
        # 拿原始 bundle 当目标，参考场和实况就会落在两套经度上，ACC/活跃度
        # 按标签对齐后全是 NaN。
        target = batch.observation
        # 气候态可能和预报不在同一圈经度上；缺 wsX 时用分量现合成（参考实现如此兜底）
        field = align_lon_to(field, target["lon"].values)
        field = add_derived_variables(field, self.observation_vars)
        aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
        if aligned is not None:
            return aligned
        return field.interp(lat=target["lat"], lon=target["lon"])

    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        variable = self.forecast_vars[0] if len(self.forecast_vars) == 1 else ""
        for key in sorted(self.batches):
            batch = self.batches[key]
            record = batch.sample_keys[0] if batch.sample_keys else {}
            base_key = (
                ("init_lead", key[0], key[1])
                if self._init_lead
                else ("valid_time", key)
            )
            base_coordinates = {
                "variable": variable,
                # 键是 (起报, 时效) 时有效时刻只能从记录里取
                "valid_time": (
                    record.get("valid_time", "") if self._init_lead else key
                ),
                "init_time": record.get("init_time", ""),
                "lead_h": record.get("lead_h", ""),
                "sample_unit": batch.sample_dim,
            }
            # 全球行不带 region 坐标——不分区时的输出行和这个能力出现之前逐行一致
            yield Sample(
                key=base_key,
                coordinates=base_coordinates,
                payload={"batch_key": key},
            )
            # 每个纬度带再各出一个样本：键尾追加带名，region 进结果坐标
            # （长表现成的 region 列）。带掩码不在 build_batch 里叠——
            # 变量路由的指标会先过 narrow_batch、valid_mask 会被重算——
            # 执行层在收窄**之后**按 payload 里的 region_bounds 叠掩码。
            for name, lat_min, lat_max in self.regions:
                yield Sample(
                    key=base_key + (name,),
                    coordinates={**base_coordinates, "region": name},
                    payload={
                        "batch_key": key,
                        "region_bounds": (lat_min, lat_max),
                    },
                )

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        return self.batches.get(sample.payload["batch_key"])

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {
            "variable": self.forecast_vars[0] if len(self.forecast_vars) == 1 else "",
            "sample_unit": "grid",
            "unit": self._unit(),
        }

    def _unit(self) -> str:
        """评分单位取自实况语义。"""
        bundle = self.observation_bundle
        if bundle is None or len(self.forecast_vars) != 1:
            return ""
        return str((bundle.semantic.units or {}).get(self.forecast_vars[0], ""))

    def summary(self) -> Dict[str, Any]:
        # 同 station 协议：报有效时刻列表而非批次数，同一个有效时刻可以由
        # (早起报, 长时效) 与 (晚起报, 短时效) 两条路径够到。
        #
        # init_lead 口径下 batches 的键是 (起报, 时效) 对，一个 run 两万个，
        # 拿去跨块做列表并集是平方级的；这里报去重后的有效时刻，语义不变。
        if self._init_lead:
            valid_times = sorted(
                {
                    str(record.get("valid_time", ""))
                    for batch in self.batches.values()
                    for record in batch.sample_keys
                }
                - {""}
            )
        else:
            valid_times = sorted(self.batches)
        summary: Dict[str, Any] = {"valid_times": valid_times}
        if self.regions:
            summary["regions"] = regions_summary(self.regions)
        for name, bundle in (
            ("forecast", self.forecast_bundle),
            ("observation", self.observation_bundle),
        ):
            if bundle is None:
                continue
            provenance = bundle.provenance
            summary[f"{name}_input_files"] = list(provenance.input_files)
            summary[f"{name}_reader"] = f"{provenance.reader_id}@{provenance.reader_version}"
        return summary


class TyphoonTrackProtocol(Protocol):
    """台风路径与强度检验：从全球场链式诊断中心，对 BABJ 报文配对。

    - **样本空间**：每个起报 × 每号台风一个样本；
    - **结果键**：``("init", ISO时刻, "storm", 编号)``；
    - **预报侧**：整段时效的 MSL 场（+ u10/v10 派生的风速），从**起报后第一条
      不早于「起报 + ``seed_min_offset_hours``」的实况位置**起步（默认 6h，见
      ``babj_seed_moment``），逐时效在方框里搜最低气压中心，下一时效的搜索框
      中心是这一时效的诊断结果（链式，见 ``pipeline/typhoon.py``）；
    - **观测侧**：BABJ 报文里 ``起报时刻 + 时效 + tz_shift``（北京时）的实况；
    - **样本单位**：``storm``。

    配对的口径按 ``valid_time`` 走：一对不上实况的时效，观测侧留空、
    ``valid_mask`` 为假，指标据此外推——**不外推、不插值**。

    **链式要求整段时效落在同一个工作块里**：时效被切开时，后一块的搜索起点
    接不上前一块诊断出的中心（只能用实况位置重起），结果会静默偏掉。所以这里
    显式校验 ``lead_chunk_days=0``，切了就报错，不出假数。
    """

    name = "typhoon_track"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        self.forecast_var = str(spec.variables.get("forecast") or "msl")
        self.wind_vars = [
            str(name)
            for name in (spec.options.get("wind_variables") or ("u10m", "v10m"))
        ]
        self.tz_shift = float(spec.local_utc_offset_hours)
        self.center_half = float(spec.options.get("center_half_deg", 3.0))
        self.early_half = float(spec.options.get("early_half_deg", 4.0))
        self.early_hours = float(spec.options.get("early_hours", 12.0))
        self.intensity_half = float(spec.options.get("intensity_half_deg", 5.0))
        # 链的种子最早取到"起报后多久"的实况，见 babj_seed_moment。6.0 = 第一个
        # 时效的步长，与标准归档同口径；0 = 就取起报时刻那条（找不到就跳过）。
        self.seed_min_offset = float(spec.options.get("seed_min_offset_hours", 6.0))
        # 时效步长（种子须落在时效网格上，见 babj_seed_moment）；_prepare_common
        # 里按预报布局的实际步长覆盖。
        self._step_hours = 6.0
        # 想只跑某几号台风就填 storm_ids（单号可写 tcid），留空跑报文目录下全部
        self.storm_ids = typhoon_storm_filter(spec.options)
        self.batches: Dict[Any, EvaluationBatch] = {}
        self.sample_list: List[Sample] = []
        self.skipped_no_analysis: List[str] = []
        self.seed_gaps: List[str] = []
        self.forecast_bundle = None
        self.observation_bundle = None

    # ------------------------------------------------------------ 准备

    def prepare(self, context: PipelineContext) -> None:
        self._prepare_common(context)
        forecast = self.forecast
        forecast_request = self._forecast_request(self.init_times, self.lead_times)
        self.forecast_bundle = forecast.reader.read(
            forecast_request, forecast.catalog.discover(forecast_request)
        )
        log.debug("台风预报场读取完成: %s", self.forecast_bundle.payload.dims)
        self._diagnose_all(self.init_times, self.lead_times)

    def _forecast_request(
        self, init_times: List[datetime], lead_times: List[float]
    ) -> DataRequest:
        """预报场的请求：定中心要 msl，定强度要两个风分量。"""
        return DataRequest(
            source_id=self.forecast.source_id,
            variables=[self.forecast_var, *self.wind_vars],
            init_times=init_times,
            lead_times=lead_times or None,
        )

    def _prepare_common(self, context: PipelineContext) -> None:
        """起报时刻、时效切分校验、BABJ 报文——确定性链与集合链共用的前段。

        预报场怎么读留给各自的 ``prepare``：确定性一次读完；集合按成员逐个读
        （见 ``TyphoonTrackEnsProtocol``），一次只驻留一个成员的场。
        """
        forecast = context.forecast
        observation = context.observation
        self.forecast = forecast
        # 时效步长：种子必须落在时效网格上（见 babj_seed_moment），布局声明的
        # step_hours 就是网格步长（本仓台风两条链都是 6h）。
        layout = getattr(forecast.reader, "layout", None)
        self._step_hours = float(getattr(layout, "step_hours", None) or 6.0)
        init_times = [
            datetime.fromisoformat(str(value))
            for value in self.spec.forecast.params.get("init_times", [])
        ]
        if not init_times:
            start, end = self.spec.period()
            discovered = forecast.catalog.discover(
                DataRequest(
                    source_id=forecast.source_id,
                    variables=[self.forecast_var],
                    init_times=daily_times(start, end),
                )
            )
            init_times = forecast.reader.available_init_times(discovered)
            if not init_times:
                raise ValueError(f"时间范围 {start} 到 {end} 内没有可用预报起报时间")
        if self.spec.limit:
            init_times = init_times[: self.spec.limit]
        self.init_times = init_times

        lead_times = [
            float(value) for value in self.spec.forecast.params.get("lead_times") or []
        ]
        self.lead_times = lead_times
        probe = forecast.catalog.discover(
            DataRequest(
                source_id=forecast.source_id,
                variables=[self.forecast_var],
                init_times=init_times,
            )
        )
        lead_max = float(forecast.reader.max_lead_hours(probe))
        if lead_times and max(lead_times) < lead_max:
            raise ConfigError(
                f"台风链路要求整段时效落在同一个工作块里，但这一块只拿到 "
                f"0–{max(lead_times):g}h（数据里有 {lead_max:g}h）。链式诊断的搜索"
                f"起点是上一时效的诊断中心，时效被切开后后一块只能退回实况位置重起，"
                f"结果会偏而看不出来。请在配置的 execution 里写 lead_chunk_days: 0。"
            )

        # BABJ：一条报文一条路径，整个目录一次读完（理由见 io/babj_reader.py）
        if context.loader is not None:
            self.observation_bundle = context.loader.materialize(
                "observation",
                DataRequest(
                    source_id=observation.source_id,
                    variables=["storm"],
                    init_times=init_times,
                ),
            )
        else:
            obs_request = DataRequest(source_id=observation.source_id, variables=["storm"])
            self.observation_bundle = observation.reader.read(
                obs_request, observation.catalog.discover(obs_request)
            )
        self.babj = self.observation_bundle.payload
        log.debug("BABJ 报文读取完成: %s", self.babj.sizes)

    @staticmethod
    def _lead_axes(ds):
        """预报场的 (lead 排序索引, 排序后的时效, 纬, 经)。

        lead 轴按数值升序对齐：文件名序号推出来的时效天然有序，但显式排序之后，
        链式诊断的推进顺序与 ``leads_hours`` 才一定一致。
        """
        order = np.argsort(np.asarray(ds["lead_time"].values, dtype="f8"))
        leads = np.asarray(ds["lead_time"].values, dtype="f8")[order]
        glat = np.asarray(ds["lat"].values, dtype="f8")
        glon = np.asarray(ds["lon"].values, dtype="f8")
        return order, leads, glat, glon

    def _chain_rows(
        self,
        ds,
        glat: np.ndarray,
        glon: np.ndarray,
        order: np.ndarray,
        leads: np.ndarray,
        analyses: Dict[datetime, Dict[str, float]],
        init_time: datetime,
        origin: Dict[str, float],
        have_wind: bool,
    ) -> List[Dict[str, Any]]:
        """一个 (成员, 起报, 台风) 跑一次链式诊断并对 BABJ 实况配对。

        确定性链走一遍，集合链的**每个成员各走一遍**，走的是同一段代码——
        两条链的逐位同源靠这个保证，别把某一侧单独改掉。
        """
        field = np.asarray(self._series(ds, self.forecast_var, init_time, order))
        wind = None
        if have_wind:
            u = np.asarray(
                self._series(ds, self.wind_vars[0], init_time, order), dtype="f8"
            )
            v = np.asarray(
                self._series(ds, self.wind_vars[1], init_time, order), dtype="f8"
            )
            wind = np.sqrt(u**2 + v**2)
        track = diagnose_track(
            field,
            glat,
            glon,
            origin["lat"],
            origin["lon"],
            leads,
            wind=wind,
            center_half=self.center_half,
            early_half=self.early_half,
            early_hours=self.early_hours,
            intensity_half=self.intensity_half,
        )
        return match_errors(track, leads, init_time, analyses, tz_shift=self.tz_shift)

    def _diagnose_all(self, init_times: List[datetime], lead_times: List[float]) -> None:
        """逐 (起报, 台风) 链式诊断 + 配对，把结果打包成批次。"""
        ds = self.forecast_bundle.payload
        order, leads, glat, glon = self._lead_axes(ds)

        have_wind = all(name in ds for name in self.wind_vars)
        if not have_wind:
            log.warning(
                "预报场缺 %s，强度项（fcst_vmax / wind_err）将全为空", "/".join(self.wind_vars)
            )

        storm_ids = [str(value) for value in np.asarray(self.babj["storm"].values)]
        for tcid in storm_ids:
            if self.storm_ids and tcid not in self.storm_ids:
                continue
            index = storm_ids.index(tcid)
            analyses = self._storm_analyses(index)
            for init_time in init_times:
                init_bjt = init_time + timedelta(hours=self.tz_shift)
                origin, seed_offset_h = self._seed(analyses, init_bjt)
                if origin is None:
                    self.skipped_no_analysis.append(f"{tcid}@{init_bjt}")
                    continue
                # 种子比下限还晚 = 第一个时效那条分析场缺了（台风当天刚编号，
                # 或者刚消散）。正常场次不记，否则每个场次都要刷一行。
                if seed_offset_h > self.seed_min_offset + 1e-6:
                    self.seed_gaps.append(f"{tcid}@{init_bjt}+{seed_offset_h:g}h")
                rows = self._chain_rows(
                    ds, glat, glon, order, leads, analyses, init_time, origin, have_wind
                )
                key = ("init", init_time.isoformat(), "storm", tcid)
                tcname = str(self.babj["tcname"].values[index])
                self.batches[key] = self._build_batch(
                    key, init_time, tcid, tcname, leads, rows, origin, seed_offset_h
                )
                self.sample_list.append(
                    Sample(
                        key=key,
                        coordinates={
                            "storm": tcid,
                            "init_time": init_time.isoformat(),
                            "sample_unit": "storm",
                        },
                        payload={"batch_key": key},
                    )
                )
        if self.seed_gaps:
            head = "、".join(self.seed_gaps[:5])
            log.info(
                "台风链式诊断：%d 个场次起报后第一个时效那条没有实况，种子落到了更晚的"
                "一条（%s%s）——这些场次的 meta 里 seed 不是 init+%gh。台风当天的第一条"
                "定位常在 20:00（12 UTC），落在这一类的多是生成首日与消散收尾。",
                len(self.seed_gaps),
                head,
                "…" if len(self.seed_gaps) > 5 else "",
                self.seed_min_offset,
            )

    @staticmethod
    def _series(ds, name: str, init_time: datetime, order) -> np.ndarray:
        """取某个起报下单个变量的整条时效序列，按 lead 升序。"""
        selected = ds[name].sel(init_time=np.datetime64(init_time))
        return np.asarray(selected.values)[order]

    def _storm_analyses(self, index: int) -> Dict[datetime, Dict[str, float]]:
        """把第 index 号台风的实况列拆成 ``{北京时: {lat, lon, pmin_hpa, vmax_ms}}``。

        形状与旧链路的 ``read_babj_analyses`` 返回值一致——``match_errors``
        直接吃这个字典。
        """
        lat = self.babj["lat"].values[:, index]
        analyses: Dict[datetime, Dict[str, float]] = {}
        for k in range(lat.size):
            if not np.isfinite(lat[k]):
                continue
            moment = self.babj["time"].values[k].astype("datetime64[us]").item()
            analyses[moment] = {
                "lat": float(lat[k]),
                "lon": float(self.babj["lon"].values[k, index]),
                "pmin_hpa": float(self.babj["pmin"].values[k, index]),
                "vmax_ms": float(self.babj["vmax"].values[k, index]),
            }
        return analyses

    def _seed(
        self, analyses: Dict[datetime, Dict[str, float]], init_bjt: datetime
    ) -> Tuple[Optional[Dict[str, float]], float]:
        """取链式诊断的起始中心，返回 ``(实况记录, 相对起报时刻的偏移小时)``。

        取哪一条由 ``babj_seed_moment`` 说了算——**计划层切块前筛起报日用的是
        同一个函数**，两边口径必须是一份，否则会出现"计划层剔了某天、协议其实
        能跑"的静默缺场次。这里只负责把时刻翻成记录。
        """
        moment = babj_seed_moment(
            analyses, init_bjt, self.seed_min_offset, self._step_hours
        )
        if moment is None:
            return None, 0.0
        return analyses[moment], (moment - init_bjt).total_seconds() / 3600.0

    def _build_batch(
        self,
        key: Tuple,
        init_time: datetime,
        tcid: str,
        tcname: str,
        leads: np.ndarray,
        rows: List[Dict[str, Any]],
        origin: Dict[str, float],
        seed_offset_h: float = 0.0,
    ) -> EvaluationBatch:
        """把一条诊断路径 + 对应实况打包成 ``(lead_time, storm)`` 的批次。

        预报侧给**诊断出来的路径**（lat/lon/pmin/vmax），观测侧给**配对上的
        实况**；两者的差就是路径误差，由 ``track_error`` 指标算——协议只管
        配对，不算分。对不上实况的时效观测侧为 NaN，``valid_mask`` 为假。
        """
        def column(name: str) -> np.ndarray:
            return np.array([row[name] for row in rows], dtype="f8")[:, None]

        coords = {"lead_time": leads, "storm": [tcid]}
        forecast_ds = xr.Dataset(
            {
                "lat": (("lead_time", "storm"), column("fcst_lat")),
                "lon": (("lead_time", "storm"), column("fcst_lon")),
                "pmin": (("lead_time", "storm"), column("fcst_pmin_hpa")),
                "vmax": (("lead_time", "storm"), column("fcst_vmax_ms")),
            },
            coords=coords,
        )
        obs_ds = xr.Dataset(
            {
                "lat": (("lead_time", "storm"), column("obs_lat")),
                "lon": (("lead_time", "storm"), column("obs_lon")),
                "pmin": (("lead_time", "storm"), column("obs_pmin_hpa")),
                "vmax": (("lead_time", "storm"), column("obs_vmax_ms")),
            },
            coords=coords,
        )
        matched = np.isfinite(column("obs_lat"))
        valid_mask = xr.DataArray(
            matched, dims=("lead_time", "storm"), coords=coords, name="valid_mask"
        )
        return EvaluationBatch(
            forecast=forecast_ds,
            observation=obs_ds,
            sample_keys=[{"init": init_time.isoformat(), "storm": tcid}],
            valid_mask=valid_mask,
            sample_dim="storm",
            alignment={
                "method": "storm_track",
                # 台风中文名只走 alignment：它不进长表（不在 _COORD_FIELDS 里），
                # 但 typhoon_cases writer 拼 typhoon.csv 时要写进 tcname 列。
                "tcname": tcname,
                "tz_shift_hours": self.tz_shift,
                "init_bjt": (init_time + timedelta(hours=self.tz_shift)).isoformat(),
                "init_position": [origin["lat"], origin["lon"]],
                # 起始中心相对起报时刻的偏移小时（见 babj_seed_moment）：正常是
                # seed_min_offset（6h），更大说明那一条分析场缺了。只在 meta 里
                # 标出来，不参与任何计算。
                "seed_offset_hours": seed_offset_h,
                "search": {
                    "center_half_deg": self.center_half,
                    "early_half_deg": self.early_half,
                    "early_hours": self.early_hours,
                    "intensity_half_deg": self.intensity_half,
                },
            },
            protocol_id=self.name,
        )

    # ------------------------------------------------------------ 遍历

    def samples(self, context: PipelineContext) -> Iterator[Sample]:
        yield from self.sample_list

    def build_batch(
        self, context: PipelineContext, sample: Sample
    ) -> Optional[EvaluationBatch]:
        return self.batches.get(sample.payload["batch_key"])

    def defaults(self, context: PipelineContext) -> Dict[str, Any]:
        return {"sample_unit": "storm", "variable": self.forecast_var, "unit": "km"}

    def summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "init_times": [value.isoformat() for value in getattr(self, "init_times", [])],
            "tz_shift_hours": self.tz_shift,
            "search": {
                "center_half_deg": self.center_half,
                "early_half_deg": self.early_half,
                "early_hours": self.early_hours,
                "intensity_half_deg": self.intensity_half,
            },
            "n_sessions": len(self.sample_list),
        }
        if self.skipped_no_analysis:
            # 起报时刻对不上实况（报文里没有那一刻的分析场）——常见于预报从 +6h
            # 起编、而报文的分析场只到 00/12 UTC，报出来免得看着像少跑了。
            summary["skipped_no_analysis"] = list(self.skipped_no_analysis)
        for name, bundle in (
            ("forecast", self.forecast_bundle),
            ("observation", self.observation_bundle),
        ):
            if bundle is None:
                continue
            summary[f"{name}_input_files"] = list(bundle.provenance.input_files)
            summary[f"{name}_reader"] = (
                f"{bundle.provenance.reader_id}@{bundle.provenance.reader_version}"
            )
        return summary


class TyphoonTrackEnsProtocol(TyphoonTrackProtocol):
    """集合台风的路径与强度检验：逐成员链式诊断，再按两种口径聚合。

    与确定性的 ``TyphoonTrackProtocol`` 只在**预报怎么读、批次怎么攒**上不同：
    起报时刻、时效切分校验、BABJ 配对、起链种子口径全部继承，链式诊断本身也走
    同一个 ``_chain_rows``——两条链的逐位同源靠这个保证。

    **成员在协议内串行。** 每个成员单独读一次预报场（把索引字典筛成单成员再读，
    见 ``_read_member``），诊断完就丢掉，峰值内存与确定性链同级（单成员一个起报
    ≈0.75G）。全部成员一次读进来是这个数的成员数倍，放不下。代价是每个工作块
    耗时 ×成员数；要更快只能把成员并进工作块，那要动 execution 的分块与合并。

    **两种聚合口径**（与旧链路 ``core/tc_ref.py`` 的 ``ens_agg`` 对齐）：

    - 方案 A：先算各成员自己的误差，再对成员平均；
    - 方案 B：先把各成员位置平均成集合平均位置，再与实况求误差。

    两者只在**路径类**三项（track_err / at / ct）上不同——三角不等式使然。
    强度类两项代数上恒等：``mean_m(pmin_m − obs) ≡ mean_m(pmin_m) − obs``，
    实况对成员是常数。批次里预报侧保留成员轴，聚合全在 ``track_error_ens`` 里做。
    """

    name = "typhoon_track_ens"

    def __init__(self, spec: PipelineSpec):
        super().__init__(spec)
        self.members_found: List[str] = []
        self.members_completed: List[str] = []
        self.failed_members: Dict[str, str] = {}
        self.forecast_input_files: List[str] = []
        self.forecast_reader_id = ""

    # ------------------------------------------------------------ 准备

    def prepare(self, context: PipelineContext) -> None:
        self._prepare_common(context)
        self._diagnose_all_ens(self.init_times, self.lead_times)

    def _read_member(self, request: DataRequest, index: DataIndex, member: str):
        """只读一个成员的预报场，返回挤压掉成员轴之后的 payload。

        ``GriddedReader.read`` 是从 ``index.available[0]`` 那个
        ``{(起报, 成员, 时效): 路径}`` 字典还原记录的，所以**把字典筛成单成员再读
        就等于只读这个成员**——io 层一行不用改，也不必把全部成员同时读进来。

        读回来带一条长度 1 的成员轴（布局声明了 ``member_dim``），这里挤掉：后面
        整条确定性代码路径（``_series`` 按 lead 取序列、``diagnose_track``）原样
        能用，不必处处判成员轴在不在。
        """
        entries = index.available[0]
        subset = {key: path for key, path in entries.items() if key[1] == member}
        if not subset:
            raise ConfigError(f"集合成员 {member} 在预报索引里没有任何文件")
        bundle = self.forecast.reader.read(
            request, DataIndex(source_id=index.source_id, available=[subset])
        )
        if not self.forecast_reader_id:
            self.forecast_reader_id = (
                f"{bundle.provenance.reader_id}@{bundle.provenance.reader_version}"
            )
        self.forecast_input_files.extend(bundle.provenance.input_files)
        payload = bundle.payload
        if "member" in payload.dims:
            payload = payload.isel(member=0, drop=True)
        return payload

    def _diagnose_all_ens(
        self, init_times: List[datetime], lead_times: List[float]
    ) -> None:
        """逐成员读 + 逐成员链式诊断，再按 (起报, 台风) 聚合成批次。"""
        request = self._forecast_request(init_times, lead_times)
        index = self.forecast.catalog.discover(request)
        if not index.available:
            raise ConfigError("集合预报索引为空，一个文件都没发现")
        members = sorted({key[1] for key in index.available[0]})
        if not members:
            raise ConfigError(
                "预报索引里一个集合成员都没有。集合链要求预报根目录下按起报分目录、"
                "每个起报目录下再放 member_*/；指到确定性目录（没有 member_ 子目录）"
                "会走到这里。"
            )
        self.members_found = list(members)

        # 与成员无关的东西先算一遍：实况与起链种子。种子只由报文和起报时刻决定，
        # 每个成员都一样，放进成员循环里重算是浪费。
        storm_ids = [str(value) for value in np.asarray(self.babj["storm"].values)]
        sessions: Dict[Tuple[str, datetime], Dict[str, Any]] = {}
        for tcid in storm_ids:
            if self.storm_ids and tcid not in self.storm_ids:
                continue
            storm_position = storm_ids.index(tcid)
            analyses = self._storm_analyses(storm_position)
            for init_time in init_times:
                init_bjt = init_time + timedelta(hours=self.tz_shift)
                origin, seed_offset_h = self._seed(analyses, init_bjt)
                if origin is None:
                    self.skipped_no_analysis.append(f"{tcid}@{init_bjt}")
                    continue
                # 种子比下限还晚 = 第一个时效那条分析场缺了（判断与确定性链同款）
                if seed_offset_h > self.seed_min_offset + 1e-6:
                    self.seed_gaps.append(f"{tcid}@{init_bjt}+{seed_offset_h:g}h")
                # 首时次 at/ct 的定向基准：起报时刻实况（旧集合链路口径）。
                # 注意与种子（origin）不是同一条记录：种子是"起报后第一条能
                # 配上时效的实况"，通常晚 6h 以上；这里要的是起报时刻本身那条，
                # 取不到就 None（首时次 at/ct 会留空）。
                sessions[(tcid, init_time)] = {
                    "origin": origin,
                    "init_obs": analyses.get(init_bjt),
                    "seed_offset_h": seed_offset_h,
                    "analyses": analyses,
                    "tcname": str(self.babj["tcname"].values[storm_position]),
                }

        # {(台风, 起报): {成员: 该成员的逐时效行}}
        collected: Dict[Tuple[str, datetime], Dict[str, List[Dict[str, Any]]]] = {
            key: {} for key in sessions
        }
        for done, member in enumerate(members, start=1):
            try:
                payload = self._read_member(request, index, member)
            except Exception as error:  # 单个成员坏掉不该毁掉整个场次
                self.failed_members[member] = str(error)
                log.warning("集合成员 %s 读取失败，跳过: %s", member, error)
                continue
            order, leads, glat, glon = self._lead_axes(payload)
            have_wind = all(name in payload for name in self.wind_vars)
            if not have_wind and done == 1:
                log.warning(
                    "预报场缺 %s，强度项（fcst_vmax / wind_err）将全为空",
                    "/".join(self.wind_vars),
                )
            for (tcid, init_time), session in sessions.items():
                collected[(tcid, init_time)][member] = self._chain_rows(
                    payload,
                    glat,
                    glon,
                    order,
                    leads,
                    session["analyses"],
                    init_time,
                    session["origin"],
                    have_wind,
                )
            self.members_completed.append(member)
            log.info("集合台风诊断：成员 %d/%d 完成（%s）", done, len(members), member)

        if not self.members_completed:
            detail = next(iter(self.failed_members.values()), "没有可用成员")
            raise ConfigError(f"集合预报没有一个成员读成功，无法出结果: {detail}")

        for (tcid, init_time), per_member in collected.items():
            if not per_member:
                continue
            session = sessions[(tcid, init_time)]
            key = ("init", init_time.isoformat(), "storm", tcid)
            self.batches[key] = self._build_batch_ens(
                key,
                init_time,
                tcid,
                session["tcname"],
                sorted(per_member),
                per_member,
                session["origin"],
                session["seed_offset_h"],
                session["init_obs"],
            )
            self.sample_list.append(
                Sample(
                    key=key,
                    coordinates={
                        "storm": tcid,
                        "init_time": init_time.isoformat(),
                        "sample_unit": "storm",
                    },
                    payload={"batch_key": key},
                )
            )

        if self.seed_gaps:
            log.info(
                "台风链式诊断：%d 个场次起报后第一个时效那条没有实况，种子落到了更晚的"
                "一条（%s%s）——这些场次的 meta 里 seed 不是 init+%gh。",
                len(self.seed_gaps),
                "、".join(self.seed_gaps[:5]),
                "…" if len(self.seed_gaps) > 5 else "",
                self.seed_min_offset,
            )
        if self.failed_members:
            log.warning(
                "集合台风：%d/%d 个成员失败，已从聚合里剔除（%s）",
                len(self.failed_members),
                len(self.members_found),
                "、".join(str(name) for name in sorted(self.failed_members)[:5]),
            )

    def _build_batch_ens(
        self,
        key: Tuple,
        init_time: datetime,
        tcid: str,
        tcname: str,
        members: List[str],
        per_member: Dict[str, List[Dict[str, Any]]],
        origin: Dict[str, float],
        seed_offset_h: float,
        init_obs: Optional[Dict[str, float]],
    ) -> EvaluationBatch:
        """打包成 ``(lead_time, storm, member)`` 的批次。

        预报侧保留**成员轴**：成员各自的诊断位置是原始数据，集合平均位置与两种
        误差口径都由 ``track_error_ens`` 去算，协议不预先聚合。观测侧给
        ``(lead_time, storm)``——实况对成员是同一份，由指标自己广播。
        ``init_obs`` 是起报时刻那条实况（可为 None）：指标用它给首时次的
        at/ct 定向（旧集合链路口径，见 ``track_error_ens``）。
        """
        def member_rows(field: str) -> np.ndarray:
            """(lead, member)：各成员在该字段上的整条时效序列。"""
            return np.stack(
                [
                    np.array([row[field] for row in per_member[member]], dtype="f8")
                    for member in members
                ],
                axis=1,
            )

        def obs_column(field: str) -> np.ndarray:
            """(lead, storm)：实况。取第一个成员的行——实况不随成员变。"""
            return np.array(
                [row[field] for row in per_member[members[0]]], dtype="f8"
            )[:, None]

        leads = np.array(
            [row["lead_h"] for row in per_member[members[0]]], dtype="f8"
        )
        shared = {"lead_time": leads, "storm": [tcid]}
        forecast_ds = xr.Dataset(
            {
                name: (("lead_time", "storm", "member"), member_rows(field)[:, None, :])
                for name, field in (
                    ("lat", "fcst_lat"),
                    ("lon", "fcst_lon"),
                    ("pmin", "fcst_pmin_hpa"),
                    ("vmax", "fcst_vmax_ms"),
                )
            },
            coords={**shared, "member": list(members)},
        )
        observation_ds = xr.Dataset(
            {
                name: (("lead_time", "storm"), obs_column(field))
                for name, field in (
                    ("lat", "obs_lat"),
                    ("lon", "obs_lon"),
                    ("pmin", "obs_pmin_hpa"),
                    ("vmax", "obs_vmax_ms"),
                )
            },
            coords=dict(shared),
        )
        valid_mask = xr.DataArray(
            np.isfinite(obs_column("obs_lat")),
            dims=("lead_time", "storm"),
            coords=dict(shared),
            name="valid_mask",
        )
        return EvaluationBatch(
            forecast=forecast_ds,
            observation=observation_ds,
            sample_keys=[{"init": init_time.isoformat(), "storm": tcid}],
            valid_mask=valid_mask,
            sample_dim="storm",
            alignment={
                "method": "storm_track_ensemble",
                # 台风中文名只走 alignment：它不进长表（不在 _COORD_FIELDS 里），
                # 但 typhoon_cases writer 拼 typhoon.csv 时要写进 tcname 列。
                "tcname": tcname,
                "tz_shift_hours": self.tz_shift,
                "init_bjt": (init_time + timedelta(hours=self.tz_shift)).isoformat(),
                "init_position": [origin["lat"], origin["lon"]],
                # 首时次 at/ct 的定向基准：**起报时刻实况**（旧集合链路口径——
                # run_tc 的集合分支 / recompute_ens_track_error.py 用它当"前一个
                # 位置"）。与上面 init_position（种子）是两条不同的记录：种子
                # 通常晚 6h 以上。取不到起报时刻实况时为 None，首时次 at/ct 留空。
                "init_obs_position": (
                    [init_obs["lat"], init_obs["lon"]] if init_obs else None
                ),
                # 起始中心相对起报时刻的偏移小时（见 babj_seed_moment）：正常是
                # seed_min_offset（6h），更大说明那一条分析场缺了。只写进 meta。
                "seed_offset_hours": seed_offset_h,
                "members": list(members),
                "n_members": len(members),
                "search": {
                    "center_half_deg": self.center_half,
                    "early_half_deg": self.early_half,
                    "early_hours": self.early_hours,
                    "intensity_half_deg": self.intensity_half,
                },
            },
            protocol_id=self.name,
        )

    # ------------------------------------------------------------ 报告

    def summary(self) -> Dict[str, Any]:
        summary = super().summary()
        summary.update(
            {
                "aggregation": {
                    "A": "先算各成员自己的误差再对成员平均（mean member error by lead）",
                    "B": "先把成员位置平均成集合平均位置再求误差"
                    "（ensemble mean position error by lead）",
                },
                "n_members_found": len(self.members_found),
                "n_members_completed": len(self.members_completed),
                "members": list(self.members_found),
                "failed_members": dict(self.failed_members),
                "forecast_reader": self.forecast_reader_id,
                # 输入文件数是一个成员的 51 倍（51 成员 × 60 时效 ≈ 三千条），
                # 全量写进 summary 会跟着 ChunkOutcome 一起 pickle 到磁盘，每个
                # 工作块平白多几百 KB。这里只取样前若干条，**总数照样写**。
                "forecast_input_files": list(self.forecast_input_files[:20]),
                "n_forecast_input_files": len(self.forecast_input_files),
            }
        )
        return summary


def _variable_list(params: Dict[str, Any], side: str) -> List[str]:
    names = params.get("variables")
    if names:
        return [str(name) for name in names]
    if params.get("variable"):
        return [str(params["variable"])]
    raise ConfigError(f"{side} 数据源必须声明 variable 或 variables")
