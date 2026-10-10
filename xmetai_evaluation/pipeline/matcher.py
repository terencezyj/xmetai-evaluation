"""
Matcher：预报与观测配对

按照 README 第 5.3 节，Matcher 负责：
1. 时间对齐（valid_time = init_time + lead_time）
2. 空间对齐（插值到共同网格）
3. 生成共同有效掩码
4. 计算权重
5. 构建 EvaluationBatch
"""

from dataclasses import replace
from typing import Dict, Any, List, Optional
import logging
import xarray as xr
import numpy as np

from xmetai_evaluation.core.contracts import (
    DataBundle,
    EvaluationBatch,
)
from xmetai_evaluation.core.errors import AlignmentError
from xmetai_evaluation.transforms.regrid import (
    align_lon_to,
    align_to_grid,
    compute_latitude_weights,
)

log = logging.getLogger(__name__)

#: 已经告警过的「缺变量」组合。缺哪些变量是**数据源和配置**决定的静态事实，
#: 但 match() 每块都调一次——不去重就是每个块 5568 行一模一样的 WARNING，
#: 真正要看的告警会被淹掉。同一条消息只打一次（进程各存一份，fork 下每段一次）。
_WARNED_MISSING: set = set()

#: 由分量合成的派生变量：名字 -> (u 分量, v 分量)。风速 = sqrt(u²+v²)，m/s。
DERIVED_VARIABLES: Dict[str, Any] = {
    "ws10m": ("u10m", "v10m"),
    "ws850": ("u850", "v850"),
    "ws200": ("u200", "v200"),
}


def add_derived_variables(ds: xr.Dataset, wanted: List[str]) -> xr.Dataset:
    """按需合成派生变量（风速 = sqrt(u²+v²)）。

    只合成**被显式请求**且分量齐备的名字——不主动发明没人要的变量。
    """
    derived: Dict[str, Any] = {}
    for name, (u_name, v_name) in DERIVED_VARIABLES.items():
        if name in ds or name not in wanted:
            continue
        if u_name not in ds or v_name not in ds:
            continue
        field = np.sqrt(ds[u_name] ** 2 + ds[v_name] ** 2)
        field.attrs["units"] = "m/s"
        derived[name] = field
    return ds.assign(**derived) if derived else ds


def narrow_batch(batch: EvaluationBatch, variable: str) -> Optional[EvaluationBatch]:
    """把批次收窄到单个变量（按变量路由指标时用）。

    收窄后**必须重算 valid_mask**：批量掩码是所有变量的交集，十几个变量里
    只要有一个缺测，就会把其它变量的有效点一起砍掉。

    预报或观测里没有该变量时返回 None，调用方据此跳过这个 run。
    """

    def pick(source):
        if source is None:
            return None
        if isinstance(source, xr.Dataset):
            return source[variable] if variable in source.data_vars else None
        return source

    forecast = pick(batch.forecast)
    observation = pick(batch.observation)
    if forecast is None or observation is None:
        return None

    reference = pick(batch.reference)
    members = pick(batch.members)

    return replace(
        batch,
        forecast=forecast,
        observation=observation,
        reference=reference,
        members=members,
        valid_mask=forecast.notnull() & observation.notnull(),
    )


def _station_latitudes(mask: xr.DataArray) -> np.ndarray:
    """一维站点掩码上逐点的纬度。

    站点纬度是 ``grid_to_station`` 变换挂在插值结果上的 ``station_lat``
    （见 ``transforms/interpolation.py``，那里有测试守着这个约定）。
    ``lat`` 是备选：观测侧读进来的站点纬度就叫这个名字，收窄后两个都有。
    """
    for name in ("station_lat", "lat"):
        coord = mask.coords.get(name)
        if coord is not None and coord.dims == mask.dims:
            return np.asarray(coord.values, dtype="f8")
    raise AlignmentError(
        "站点分纬度带评估需要掩码上带站点纬度坐标（station_lat），"
        f"当前掩码维度 {tuple(mask.dims)}、可用坐标 {sorted(mask.coords)}"
    )


def restrict_to_latitude_band(
    batch: EvaluationBatch, lat_min: float, lat_max: float
) -> EvaluationBatch:
    """把批次的 valid_mask 收缩到 [lat_min, lat_max] 内（闭区间）。

    分纬度带评估用。数据一个点不裁、只挡掩码：标量指标（rmse/bias/
    acc/activity/ts_score）全部按 valid_mask 加权或筛点，带外的点自然
    不进任何统计量。要完整场的指标（谱/FSS）在执行层就被跳过 region
    样本，不会走到这里。

    两种掩码各走一条：
    - 格点批次 ``("lat", "lon")``：按掩码自己的 ``lat`` 坐标挡；
    - 站点批次一维 ``("station",)``：按站点纬度挡（见
      :func:`_station_latitudes`）。站点点位是离散的，闭区间下带外的站
      整站在掩码里被挡掉，站上的预报/观测值一个不裁。

    **闭区间**是刻意的：与 :func:`protocols.parse_regions` 的带定义一致，
    改成一端开会让格点与站点两类产物的分带口径不一致。

    必须在 ``narrow_batch`` **之后**调用：收窄会按单变量重算 valid_mask，
    先叠的带掩码会被重算覆盖掉。
    """
    mask = batch.valid_mask
    if mask is None:
        raise AlignmentError("分纬度带评估需要有效掩码，当前批次没有 valid_mask")

    if mask.dims[-2:] == ("lat", "lon") and "lat" in mask.coords:
        lat = mask.coords["lat"].values
        inside = (lat >= lat_min) & (lat <= lat_max)
        band = xr.DataArray(
            np.broadcast_to(inside[:, None], mask.shape[-2:]),
            coords={"lat": lat, "lon": mask.coords["lon"].values},
            dims=["lat", "lon"],
        )
    elif mask.ndim == 1:
        lats = _station_latitudes(mask)
        band = xr.DataArray(
            np.asarray((lats >= lat_min) & (lats <= lat_max)),
            dims=list(mask.dims),
        )
    else:
        raise AlignmentError(
            "分纬度带评估的掩码只能是格点 (lat, lon) 或一维站点 (station,)，"
            f"当前掩码维度是 {tuple(mask.dims)}"
        )
    return replace(batch, valid_mask=mask & band)


class Matcher:
    """
    预报与观测配对器

    负责时间、空间对齐，生成 EvaluationBatch。
    """

    def __init__(self, ensemble_reduction: str = "mean"):
        """
        Args:
            ensemble_reduction: 集合处理方式（'mean', 'median', 'none'）
        """
        self.ensemble_reduction = ensemble_reduction
        #: 经度口径是否做过折算，写进 batch.alignment 供排查"结果全 NaN"用
        self._lon_convention = "native"
        #: 空间对齐方式，同样写进 batch.alignment
        self._spatial = "forecast_regridded_to_observation"

    def match(
        self,
        forecast: DataBundle,
        observation: DataBundle,
        reference: Optional[DataBundle] = None,
        variables: Optional[List[str]] = None,
        sample_by: str = "valid_time",
    ) -> List[EvaluationBatch]:
        """
        配对预报与观测

        Args:
            forecast: 预报数据（可能包含 init_time, lead_time, member 维度）
            observation: 观测数据（包含 valid_time 维度）
            reference: 可选的参考数据（如气候态）
            variables: 要评测的变量列表（None 表示全部）
            sample_by: 批次的口径。``valid_time``（缺省）一个有效时刻一个批次，
                多个起报够到同一时刻时只留最新起报；``init_lead`` 每个
                (起报, 时效) 各出一个批次，同一个有效时刻有几个起报就有几个批次。

        Returns:
            EvaluationBatch 列表
        """
        forecast_ds = forecast.payload if isinstance(forecast.payload, xr.Dataset) else forecast.payload.to_dataset()
        observation_ds = observation.payload if isinstance(observation.payload, xr.Dataset) else observation.payload.to_dataset()

        # 经度折算：观测在 -180..180 而预报在 0..360 时，不折算 interp 会全 NaN
        lon_before = (
            np.asarray(observation_ds["lon"].values)
            if "lon" in observation_ds.coords
            else None
        )
        observation_ds = align_lon_to(observation_ds, forecast_ds["lon"].values)
        if lon_before is not None and "lon" in observation_ds.coords:
            self._lon_convention = (
                "native"
                if np.array_equal(lon_before, np.asarray(observation_ds["lon"].values))
                else "observation_wrapped_to_forecast_circle"
            )

        # 派生变量必须在变量求交**之前**合成：ws* 任何文件里都没有，
        # 放到求交之后会被当成"源里没有这个变量"直接丢掉。
        wanted = (
            list(variables)
            if variables is not None
            else list(set(forecast_ds.data_vars) | set(observation_ds.data_vars))
        )
        forecast_ds = add_derived_variables(forecast_ds, wanted)
        observation_ds = add_derived_variables(observation_ds, wanted)

        # 确定变量
        if variables is None:
            variables = list(set(forecast_ds.data_vars) & set(observation_ds.data_vars))

        # 缺变量按参考实现语义跳过并告警：一个源少几个要素不该让整轮评测挂掉
        missing = [
            name
            for name in variables
            if name not in forecast_ds.data_vars or name not in observation_ds.data_vars
        ]
        if missing:
            message = ", ".join(missing)
            if message not in _WARNED_MISSING:
                _WARNED_MISSING.add(message)
                log.warning("以下变量在预报或观测中不存在，已跳过: %s", message)
            variables = [name for name in variables if name not in missing]

        if not variables:
            raise AlignmentError("No common variables between forecast and observation")

        # 只保留共同变量
        forecast_ds = forecast_ds[variables]
        observation_ds = observation_ds[variables]

        # 计算 valid_time（如果预报有 init_time 和 lead_time）
        #
        # 配对记录是**与展平后的 valid_time 轴同序同长的位置列表**，不是按
        # valid_time 索引的字典。字典口径下同一个有效时刻被多个 (起报, 时效)
        # 够到时，后写的会把先写的挤掉，只留下最新起报——那是 valid_time 口径
        # 要的语义，但 init_lead 口径要每个 (起报, 时效) 各算各的，键一撞就
        # 整批丢掉（15 天只剩得下头几个时效）。按位置取就没有这回事。
        flatten_times: List[Any] = []
        sample_records: List[Dict[str, Any]] = []
        if "init_time" in forecast_ds.dims and "lead_time" in forecast_ds.dims:
            for init_time in forecast_ds.coords["init_time"].values:
                for lead_time in forecast_ds.coords["lead_time"].values:
                    flatten_times.append(
                        init_time + np.timedelta64(int(lead_time), "h")
                    )
                    sample_records.append(
                        {
                            "init_time": str(init_time),
                            "lead_h": float(lead_time),
                        }
                    )
            forecast_ds = self._add_valid_time(forecast_ds)

        # 集合处理
        if "member" in forecast_ds.dims and self.ensemble_reduction == "mean":
            forecast_ds = forecast_ds.mean(dim="member")

        # 空间对齐：两个源本就是同一套全球网格时（lat 可能反向、lon 可能差半圈），
        # 按索引搬运即可，一格都不用插。
        #
        # 绝不能轻易落到 xarray.interp：它会把 (时效 × 要素) 一次性展平，
        # 60 时效 × 16 要素 = 6200 万个点逐个算双线性权重，结果数组 8GB 起步。
        # 参考实现（vfc/regr_ens.py 的 _align_info / _apply_target_align）从头到尾
        # 只做 a[:, ::-1, :] 和 np.roll，从不插值。
        aligned = align_to_grid(
            observation_ds, forecast_ds["lat"].values, forecast_ds["lon"].values
        )
        if aligned is not None:
            observation_ds = aligned
            self._spatial = "grids_identical"
            log.debug(
                "预报与观测同网格（%d×%d），按索引对齐，跳过插值",
                forecast_ds.sizes.get("lat", 0),
                forecast_ds.sizes.get("lon", 0),
            )
            forecast_regridded = forecast_ds
        else:
            self._spatial = "forecast_regridded_to_observation"
            log.warning(
                "预报与观测不是同一套网格，退回双线性插值（很慢）。"
                "预报 lat %s..%s lon %s..%s；实况 lat %s..%s lon %s..%s",
                forecast_ds["lat"].values[0], forecast_ds["lat"].values[-1],
                forecast_ds["lon"].values[0], forecast_ds["lon"].values[-1],
                observation_ds["lat"].values[0], observation_ds["lat"].values[-1],
                observation_ds["lon"].values[0], observation_ds["lon"].values[-1],
            )
            forecast_regridded = forecast_ds.interp(
                lat=observation_ds.coords["lat"],
                lon=observation_ds.coords["lon"],
                method="linear",
            )

        # 时间对齐：找到共同的 valid_time
        if "valid_time" in forecast_regridded.dims and "valid_time" in observation_ds.dims:
            if sample_by == "init_lead":
                # 逐 (起报, 时效) 出样本时**不能**取交集再重排：按值取交集会把
                # 同一个有效时刻的多个起报折叠成一个位置，与配对记录（按展平
                # 顺序排的位置列表）错位。这里只按位置筛掉实况里没有的有效
                # 时刻，顺序原样保留。
                available = set(observation_ds.coords["valid_time"].values)
                kept = [
                    position
                    for position, valid_time in enumerate(
                        forecast_regridded.coords["valid_time"].values
                    )
                    if valid_time in available
                ]
                if not kept:
                    raise AlignmentError(
                        "No common valid_time between forecast and observation"
                    )
                forecast_regridded = forecast_regridded.isel(valid_time=kept)
                observation_ds = observation_ds.sel(
                    valid_time=forecast_regridded.coords["valid_time"].values
                )
                sample_records = [sample_records[position] for position in kept]
            else:
                common_times = list(
                    set(forecast_regridded.coords["valid_time"].values) &
                    set(observation_ds.coords["valid_time"].values)
                )
                if not common_times:
                    raise AlignmentError("No common valid_time between forecast and observation")

                forecast_regridded = forecast_regridded.sel(valid_time=common_times)
                observation_ds = observation_ds.sel(valid_time=common_times)

        # 计算权重（纬度余弦权重）
        weights = compute_latitude_weights(observation_ds)

        # 为每个时次创建 EvaluationBatch
        batches = []

        if "valid_time" in forecast_regridded.dims:
            moments = list(forecast_regridded.coords["valid_time"].values)
            if sample_by == "init_lead":
                if len(moments) != len(sample_records):
                    log.warning(
                        "配对记录 %d 条与有效时刻 %d 个对不上，按短的来",
                        len(sample_records), len(moments),
                    )
                # 一个位置 = 一个 (起报, 时效)：按位置切，配对记录同序同长。
                #
                # 用**标量**位置索引，不能用 `isel(valid_time=[i])`：列表索引会
                # 保留一个长度为 1 的 valid_time 维，而 valid_time 分支的
                # `sel(valid_time=标量)` 会把它降掉，参考场 `_reference_for` 也是
                # 标量 sel。三方形状必须一致，否则 ActivityRatio 这类逐格指标
                # 直接抛「形状不一致：fc=(1,721,1440) … clim=(721,1440)」。
                picks = [
                    (
                        moments[position],
                        forecast_regridded.isel(valid_time=position),
                        observation_ds.isel(valid_time=position),
                        sample_records[position],
                    )
                    for position in range(min(len(moments), len(sample_records)))
                ]
            else:
                # 一个值 = 一个有效时刻；同值的多个起报只留最新那个（与旧字典同）
                records_by_time = dict(zip(flatten_times, sample_records))
                picks = [
                    (
                        valid_time,
                        forecast_regridded.sel(valid_time=valid_time),
                        observation_ds.sel(valid_time=valid_time),
                        records_by_time.get(valid_time, {}),
                    )
                    for valid_time in moments
                ]
            for valid_time, forecast_slice, observation_slice, record in picks:
                batch = self._create_batch(
                    forecast_slice,
                    observation_slice,
                    weights,
                    valid_time,
                    reference,
                    record,
                )
                batches.append(batch)
                if len(batches) % 20 == 0:
                    log.info("配对进度 %d/%d", len(batches), len(moments))
        else:
            # 单时次
            batch = self._create_batch(
                forecast_regridded,
                observation_ds,
                weights,
                None,
                reference,
            )
            batches.append(batch)

        return batches

    def _add_valid_time(self, forecast_ds: xr.Dataset) -> xr.Dataset:
        """添加 valid_time 坐标"""
        # valid_time = init_time + lead_time
        init_times = forecast_ds.coords["init_time"].values
        lead_times = forecast_ds.coords["lead_time"].values

        # 创建 valid_time 坐标
        valid_times = []
        for init_time in init_times:
            for lead_time in lead_times:
                valid_time = init_time + np.timedelta64(int(lead_time), 'h')
                valid_times.append(valid_time)

        # 重构数据集
        # 将 (init_time, lead_time) 展平为 valid_time
        forecast_flat = []
        for init_time in init_times:
            for lead_time in lead_times:
                ds_slice = forecast_ds.sel(init_time=init_time, lead_time=lead_time)
                valid_time = init_time + np.timedelta64(int(lead_time), 'h')
                ds_slice = ds_slice.expand_dims(valid_time=[valid_time])
                forecast_flat.append(ds_slice)

        # xarray 的弃用预告是**成对**的：不写 coords 它问 coords，写了 coords 它
        # 转头问 compat（"Cannot specify both coords='different' and
        # compat='override'"）。两个都写死才是逐字复现今天的默认行为，
        # 两条警告才会一起消失。这函数**每个批次调一次**，落一个就是几百行
        # 一模一样的东西把日志糊死——真正要看的进度和告警全被埋掉。
        forecast_ds = xr.concat(
            forecast_flat, dim="valid_time", coords="different", compat="equals"
        )

        return forecast_ds

    def _create_batch(
        self,
        forecast: xr.Dataset,
        observation: xr.Dataset,
        weights: xr.DataArray,
        valid_time: Optional[np.datetime64],
        reference: Optional[DataBundle],
        sample_record: Optional[Dict[str, Any]] = None,
    ) -> EvaluationBatch:
        """创建单个 EvaluationBatch"""
        # 计算共同有效掩码
        forecast_valid = ~forecast.isnull()
        observation_valid = ~observation.isnull()

        # 对所有变量取交集
        valid_mask = xr.DataArray(True, coords=observation.coords)
        for var in forecast.data_vars:
            valid_mask = valid_mask & forecast_valid[var] & observation_valid[var]

        # Sample key
        sample_keys = [
            {
                **(sample_record or {}),
                "valid_time": str(valid_time) if valid_time is not None else "unknown",
            }
        ]

        # 对齐记录
        alignment = {
            "spatial": self._spatial,
            "temporal": "valid_time_matched",
            "units": "assumed_consistent",
            "lon_convention": self._lon_convention,
        }

        return EvaluationBatch(
            forecast=forecast,
            observation=observation,
            sample_keys=sample_keys,
            valid_mask=valid_mask,
            reference=reference,
            weights=weights,
            sample_dim="grid",
            alignment=alignment,
        )
