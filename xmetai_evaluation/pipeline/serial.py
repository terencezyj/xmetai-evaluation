from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import xarray as xr

from xmetai_evaluation.core.errors import ConfigError, EvaluationError
from xmetai_evaluation.io.climatology_reader import ClimatologyCatalog, ClimatologyReader
from xmetai_evaluation.io.gridded import GriddedCatalog, GriddedReader
from xmetai_evaluation.io.layouts import CRA_LAYOUT, FENGQING_LAYOUT
from xmetai_evaluation.metrics.acc import ACC
from xmetai_evaluation.metrics.bias import Bias
from xmetai_evaluation.metrics.ensemble import CRPS, SpreadError
from xmetai_evaluation.metrics.rmse import RMSE
from xmetai_evaluation.metrics.specialized import ActivityRatio, PowerSpectrum
from xmetai_evaluation.output import ResultStore, RunContext
from xmetai_evaluation.pipeline.matcher import (
    Matcher,
    add_derived_variables,
    narrow_batch,
)
from xmetai_evaluation.transforms.regrid import (
    align_lon_to,
    align_to_grid,
    compute_ensemble_mean,
)


def run_serial_evaluation(cfg: Any) -> list:
    """从 cfg 到 scores.csv 的完整 Fengqing 网格数据流，返回指标结果列表。"""
    print("active test: 2026年10月10日20:20:51")

    # 1. 细化统一配置：确定数据源、变量、指标和配对口径。
    forecast_cfg = cfg.forecast_reader
    observation_cfg = cfg.observation_reader
    reference_cfg = cfg.reference_reader or {}
    if forecast_cfg.get("type") != "fengqing" or observation_cfg.get("type") != "cra":
        raise ConfigError("此串行入口只支持 Fengqing 预报与 CRA40 实况")
    forecast_vars = _variables(forecast_cfg, "预报")
    observation_vars = _variables(observation_cfg, "实况")
    metric_options = cfg.metric_options or {}
    metrics = _build_metrics(metric_options)
    options = getattr(cfg, "options", None) or {}
    sample_by = str(options.get("sample_by", "valid_time"))
    if sample_by not in ("valid_time", "init_lead"):
        raise ConfigError(f"不支持的 sample_by: {sample_by}")
    if options.get("regions"):
        raise ConfigError("此 Fengqing 串行入口尚不支持 regions")
    read_leads = [float(value) for value in forecast_cfg.get("lead_times") or []]
    needs_reference = any(metric.needs_reference() for metric in metrics)

    # 2. 加载数据：预报的有效时刻决定实况和气候态的读取范围。
    init_times = _init_times(cfg)
    raw_forecast = _load_gridded(forecast_cfg, init_times, read_leads or None)
    valid_times = _valid_times(init_times, forecast_cfg, raw_forecast)
    observation = _load_gridded(observation_cfg, valid_times)
    reference = None
    if needs_reference and reference_cfg:
        reference = _load_climatology(reference_cfg, observation_vars, valid_times)

    # 3. 处理数据：派生变量、集合均值、时间和网格配对。
    forecast = _prepare_forecast(raw_forecast, forecast_vars)
    batches = _pair_samples(
        forecast, observation, raw_forecast, reference, metrics, cfg
    )

    # 4. 计算指标，并合并同一时刻、变量的统计状态。
    states, coordinates, processed, skipped = _calculate_metrics(
        batches, metrics, metric_options, sample_by
    )
    defaults = _result_defaults(forecast_vars, observation)
    scores = _merge_results(states, coordinates, metrics, defaults)

    # 5. 导出结果与运行记录。
    _write_results(cfg, scores, forecast, observation, batches, defaults, processed, skipped)
    return scores


def _variables(source_cfg: dict, label: str) -> list[str]:
    names = source_cfg.get("variables") or (
        [source_cfg["variable"]] if source_cfg.get("variable") else []
    )
    if not names:
        raise ConfigError(f"{label}数据源需要 variable 或 variables")
    return [str(name) for name in names]


def _parse_date(value: str) -> datetime:
    for fmt in ("%Y%m%d", "%Y%m%d%H", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            pass
    raise ConfigError(f"无效日期: {value}")


def _init_times(cfg: Any) -> list[datetime]:
    """优先用配置里的起报时刻，否则从预报文件目录发现。"""
    source = cfg.forecast_reader
    times = [datetime.fromisoformat(str(value)) for value in source.get("init_times") or []]
    if times:
        times.sort()
    else:
        start = _parse_date(cfg.start_date)
        end = _parse_date(cfg.end_date or cfg.start_date)
        dates = []
        current = start
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        root = source.get("root_dir") or source.get("root")
        if not root:
            raise ConfigError("Fengqing 数据源需要 root_dir")
        step = source.get("step_hours")
        layout = (
            FENGQING_LAYOUT.with_step_hours(float(step)) if step is not None else FENGQING_LAYOUT
        )
        source_id = source.get("source_id", "fengqing")
        reader = GriddedReader(source_id=source_id, layout=layout)
        catalog = GriddedCatalog(Path(root), layout)
        request = SimpleNamespace(
            source_id=source_id,
            variables=_variables(source, "预报"),
            init_times=dates,
            lead_times=None,
        )
        times = list(reader.available_init_times(catalog.discover(request)))
    if cfg.limit:
        times = times[: cfg.limit]
    if not times:
        raise EvaluationError("评测时段内没有可用的 Fengqing 起报")
    return times


def _load_gridded(source: dict, times: list[datetime], leads: list[float] | None = None):
    """按数据源配置查找并读取 Fengqing 或 CRA40 网格文件。"""
    source_type = source["type"]
    root = source.get("root_dir") or source.get("root")
    if not root:
        raise ConfigError(f"{source_type} 数据源需要 root_dir")
    if source_type == "fengqing":
        step = source.get("step_hours")
        layout = (
            FENGQING_LAYOUT.with_step_hours(float(step)) if step is not None else FENGQING_LAYOUT
        )
    else:
        layout = CRA_LAYOUT
    source_id = source.get("source_id", source_type)
    reader = GriddedReader(source_id=source_id, layout=layout)
    catalog = GriddedCatalog(Path(root), layout)
    request = SimpleNamespace(
        source_id=source_id,
        variables=_variables(source, "预报" if source_type == "fengqing" else "实况"),
        init_times=times,
        lead_times=leads,
    )
    return reader.read(request, catalog.discover(request))


def _load_climatology(source: dict, variables: list[str], times: list[datetime]):
    """按有效时刻读取气候态文件。"""
    if source.get("type") != "climatology":
        raise ConfigError("此串行入口只支持 climatology 参考场")
    root = source.get("root_dir") or source.get("root")
    if not root:
        raise ConfigError("climatology 数据源需要 root_dir")
    source_id = source.get("source_id", "climatology")
    reader = ClimatologyReader(source_id=source_id, engine=str(source.get("engine", "cfgrib")))
    catalog = ClimatologyCatalog(Path(root))
    request = SimpleNamespace(
        source_id=source_id, variables=variables, init_times=times, lead_times=None
    )
    return reader.read(request, catalog.discover(request))


def _prepare_forecast(raw_forecast, variables: list[str]):
    """派生变量先逐成员计算，再对集合成员求平均。"""
    if isinstance(raw_forecast.payload, xr.Dataset) and "member" in raw_forecast.payload.dims:
        raw_forecast.payload = add_derived_variables(raw_forecast.payload, variables)
    return compute_ensemble_mean(raw_forecast)


def _valid_times(init_times: list[datetime], source: dict, forecast) -> list[datetime]:
    """用抽样时效计算实况和气候态需要覆盖的有效时刻。"""
    sample_leads = [float(value) for value in source.get("sample_leads") or []]
    read_leads = [float(value) for value in source.get("lead_times") or []]
    leads = sample_leads or read_leads or [
        float(value) for value in forecast.payload["lead_time"].values
    ]
    return sorted({init + timedelta(hours=lead) for init in init_times for lead in leads})


def _pair_samples(forecast, observation, raw_forecast, reference, metrics: list, cfg: Any):
    """配对预报和实况，并为需要的指标附上成员场和气候态。"""
    options = getattr(cfg, "options", None) or {}
    sample_by = str(options.get("sample_by", "valid_time"))
    forecast_vars = _variables(cfg.forecast_reader, "预报")
    observation_vars = _variables(cfg.observation_reader, "实况")
    member_vars = _member_variables(metrics, cfg.metric_options or {}, forecast_vars)
    batches = Matcher(ensemble_reduction=str(options.get("ensemble_reduction", "mean"))).match(
        forecast, observation, variables=forecast_vars, sample_by=sample_by
    )
    batches.sort(
        key=lambda batch: (
            batch.sample_keys[0].get("valid_time", ""),
            batch.sample_keys[0].get("init_time", ""),
        )
    )
    for batch in batches:
        if member_vars:
            batch.members = _members_for(batch, raw_forecast, member_vars)
        if reference is not None:
            batch.reference = _reference_for(batch, reference, observation_vars)
    return batches


def _calculate_metrics(batches: list, metrics: list, metric_options: dict, sample_by: str):
    """逐配对样本计算指标，只收集可合并的统计状态。"""
    states = {}
    coordinates = {}
    processed = skipped = 0
    for batch in batches:
        record = batch.sample_keys[0]
        sample_coordinates = {
            "valid_time": record.get("valid_time", ""),
            "init_time": record.get("init_time", ""),
            "lead_h": record.get("lead_h", ""),
            "sample_unit": batch.sample_dim,
        }
        sample_key = (
            (
                sample_coordinates["valid_time"],
                sample_coordinates["init_time"],
                sample_coordinates["lead_h"],
            )
            if sample_by == "init_lead"
            else sample_coordinates["valid_time"]
        )
        calculated = False
        for index, metric in enumerate(metrics):
            for variable in metric_options[metric.name].get("variables") or [None]:
                target = narrow_batch(batch, variable) if variable else batch
                if target is None:
                    continue
                metric.validate(target)
                state = metric.accumulate(target)
                key = (index, sample_key, variable)
                states.setdefault(key, []).append(state)
                coordinates.setdefault(
                    key,
                    {
                        **sample_coordinates,
                        **({"variable": variable} if variable else {}),
                    },
                )
                calculated = True
        if calculated:
            processed += 1
        else:
            skipped += 1
    if not states:
        raise EvaluationError("Fengqing 评测没有成功处理任何配对样本")
    return states, coordinates, processed, skipped


def _result_defaults(forecast_vars: list[str], observation) -> dict:
    """为单变量结果补齐变量、样本类型和单位。"""
    single = len(forecast_vars) == 1
    return {
        "variable": forecast_vars[0] if single else "",
        "sample_unit": "grid",
        "unit": str(observation.semantic.units.get(forecast_vars[0], "")) if single else "",
    }


def _merge_results(states: dict, coordinates: dict, metrics: list, defaults: dict) -> list:
    """合并相同结果键的状态，再计算最终分数和结果坐标。"""
    scores = []
    for key, values in states.items():
        metric = metrics[key[0]]
        score = metric.finalize(metric.merge(values))
        score.coordinates = {**defaults, **coordinates[key]}
        score.product_kind = score.product_kind or getattr(metric, "PRODUCT_KIND", "")
        score.unit = score.unit or defaults["unit"]
        score.protocol_id = score.protocol_id or "grid_valid_time"
        scores.append(score)
    return scores


def _write_results(
    cfg: Any,
    scores: list,
    forecast,
    observation,
    batches: list,
    defaults: dict,
    processed: int,
    skipped: int,
) -> None:
    """统一写出指标结果和运行记录。"""
    forecast_id = cfg.forecast_reader.get("source_id", "fengqing")
    observation_id = cfg.observation_reader.get("source_id", "cra")
    manifest = {
        "description": cfg.description,
        "n_processed_batches": processed,
        "n_skipped_batches": skipped,
        "forecast_source": forecast_id,
        "observation_source": observation_id,
        "execution": {"mode": "serial", "chunking": False},
        "valid_times": sorted({batch.sample_keys[0]["valid_time"] for batch in batches}),
        "forecast_input_files": list(forecast.provenance.input_files),
        "observation_input_files": list(observation.provenance.input_files),
        "forecast_reader": f"{forecast.provenance.reader_id}@{forecast.provenance.reader_version}",
        "observation_reader": (
            f"{observation.provenance.reader_id}@{observation.provenance.reader_version}"
        ),
    }
    ResultStore(
        Path(cfg.output_dir),
        RunContext(cfg.name, forecast_id, observation_id, "grid_valid_time"),
        defaults=defaults,
    ).write(
        scores,
        manifest=manifest,
        resolved_config=asdict(cfg),
        writers=(
            cfg.writers
            if cfg.writers is not None
            else (["csv_long", "spectrum"] if "spectrum" in cfg.metric_options else ["csv_long"])
        ),
    )


def _build_metrics(options: dict) -> list:
    """cfg.metric_options 明确指定要运行的指标及其参数。"""
    constructors = {
        "rmse": lambda p: RMSE(p),
        "bias": lambda p: Bias(p),
        "acc": lambda p: ACC({"climatology_path": None, **p}),
        "activity": lambda p: ActivityRatio(p),
        "crps": lambda p: CRPS(p),
        "spread_error": lambda p: SpreadError(ddof=int(p.pop("ddof", 1)), params=p),
        "spectrum": lambda p: PowerSpectrum(
            max_wavenumber=int(p.pop("max_wavenumber", 30)),
            lat_order=p.pop("lat_order", None),
            params=p,
        ),
    }
    if not options:
        raise ConfigError("Fengqing 串行评测需要在 metric_options 中列出要算的指标")
    unknown = set(options) - set(constructors)
    if unknown:
        raise ConfigError(f"Fengqing 串行评测不支持指标: {sorted(unknown)}")
    return [constructors[name](dict(params or {})) for name, params in options.items()]


def _member_variables(metrics: list, metric_options: dict, forecast_vars: list[str]) -> list[str]:
    names = []
    for metric in metrics:
        if metric.needs_members():
            for name in metric_options[metric.name].get("variables") or forecast_vars:
                if name not in names:
                    names.append(name)
    return names


def _members_for(batch, raw_forecast, variables: list[str]):
    record = batch.sample_keys[0]
    payload = raw_forecast.payload
    available = [name for name in variables if name in payload.data_vars]
    if not available or "member" not in payload.dims:
        return None
    try:
        field = payload[available].sel(
            init_time=np.datetime64(record["init_time"]),
            lead_time=float(record["lead_h"]),
        )
    except (KeyError, ValueError):
        return None
    if "member" not in field.dims:
        return None
    target = batch.observation
    aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
    return aligned if aligned is not None else field.interp(lat=target["lat"], lon=target["lon"])


def _reference_for(batch, reference, variables: list[str]):
    record = batch.sample_keys[0]
    available = [name for name in variables if name in reference.payload.data_vars]
    if not available:
        return None
    try:
        field = reference.payload[available].sel(valid_time=np.datetime64(record["valid_time"]))
    except (KeyError, ValueError):
        return None
    target = batch.observation
    field = add_derived_variables(align_lon_to(field, target["lon"].values), variables)
    aligned = align_to_grid(field, target["lat"].values, target["lon"].values)
    return aligned if aligned is not None else field.interp(lat=target["lat"], lon=target["lon"])
