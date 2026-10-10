"""
网格插值 Transform

将预报场插值到观测场的网格，支持：
- 双线性插值（默认）
- 最近邻插值
- 保守重映射（需要额外库）

按照 README 第 13.3 节：Transform 是可复用的数据运算，输入和输出是 DataBundle。
"""

from typing import Any, Optional

import numpy as np
import xarray as xr

from xmetai_evaluation.core.contracts import DataBundle
from xmetai_evaluation.core.errors import ContractError


def align_lon_to(ds: xr.Dataset, target_lon: Any) -> xr.Dataset:
    """把 ``ds`` 的经度折算到 ``target_lon`` 所在的那一圈。

    不同数据源的经度原点会差 180°（如 ERA5 zarr 用 -180..180，预报文件用 0..360）。
    xarray 的 ``interp`` 在目标坐标超出源坐标范围时**直接返回 NaN**，不报错，
    所以不折算就会静默配对出一整片空场。经度是周期坐标，折算等价且不丢信息。

    折算后必须 ``sortby``：``[-180..180)`` 取模到 ``[0..360)`` 会变成降序，
    而 ``spectrum`` / ``zonal_spectrum`` 的 rFFT 依赖经度升序。
    """
    if "lon" not in ds.coords:
        return ds
    lon = np.asarray(ds["lon"].values)
    target = np.asarray(target_lon)
    if lon.size == 0 or target.size == 0:
        return ds
    lon_min = float(lon.min())
    target_min = float(target.min())
    if lon_min < 0.0 <= target_min:
        return ds.assign_coords(lon=lon % 360.0).sortby("lon")
    if lon_min >= 0.0 > target_min:
        return ds.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds


def align_to_grid(
    ds: xr.Dataset, target_lat: Any, target_lon: Any, atol: float = 1e-6
) -> Optional[xr.Dataset]:
    """把 ``ds`` 按**索引**搬到 ``target_lat`` / ``target_lon`` 这套网格上。

    只认「同一套网格，可能上下颠倒（lat 升序 vs 降序）或差半圈（经度原点不同）」
    这一种情况：lat 要么逐点相等、要么逐点反转；lon 要么相等、要么存在一个环形
    位移 ``k`` 能把源经度对齐到目标经度。对不上返回 ``None``，由调用方决定是否
    退回插值。

    这里一个插值都不做，是参考实现（``vfc/regr_ens.py:84-118``）的做法：两边本
    就是同一套全球网格时，``a[:, ::-1, :]`` 和 ``np.roll`` 是精确搬运——既没有
    双线性权重误差，也不会把 (时效 × 要素) 展平成 GB 级的中间数组。
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        return None
    lat = np.asarray(ds["lat"].values, dtype="f8")
    lon = np.asarray(ds["lon"].values, dtype="f8")
    want_lat = np.asarray(target_lat, dtype="f8")
    want_lon = np.asarray(target_lon, dtype="f8")

    if lat.shape != want_lat.shape or lon.shape != want_lon.shape or lon.size == 0:
        return None

    if np.allclose(lat, want_lat, rtol=0.0, atol=atol):
        flip = False
    elif np.allclose(lat, want_lat[::-1], rtol=0.0, atol=atol):
        flip = True
    else:
        return None

    # 位移量先用首元素定位候选，再整体比对确认：均匀网格上是 O(n)，
    # 逐个 k 去试是 O(n²)（1440² 也还好，但没必要）。
    roll = next(
        (
            int(k)
            for k in np.flatnonzero(np.isclose(lon, want_lon[0], rtol=0.0, atol=atol))
            if np.allclose(np.roll(lon, int(k)), want_lon, rtol=0.0, atol=atol)
        ),
        None,
    )
    if roll is None:
        return None

    if flip:
        ds = ds.isel(lat=slice(None, None, -1))
    if roll:
        ds = ds.roll(lon=roll, roll_coords=True)
    # 坐标按目标逐点写回：下游掩码/权重是按标签对齐的，浮点尾数级别的差异
    # 留在坐标里会让本可命中的配对悄悄落空。
    return ds.assign_coords(lat=want_lat, lon=want_lon)


def compute_ensemble_mean(
    bundle: DataBundle, member_dim: str = "member", skipna: bool = False
) -> DataBundle:
    """
    计算集合均值

    Args:
        bundle: 包含集合成员的数据
        member_dim: 成员维度名（默认 'member'）
        skipna: 是否跳过缺测成员。默认 False：任一成员缺测则整点缺测，
            与参考实现 ``members.mean(axis=0)`` 的传播语义一致，避免"缺员还当有效"。

    Returns:
        集合均值 DataBundle（不含 member 维度）
    """
    ds = bundle.payload if isinstance(bundle.payload, xr.Dataset) else bundle.payload.to_dataset()

    if member_dim not in ds.dims:
        # 已经是确定性场，直接返回
        return bundle

    # 计算均值
    mean_ds = ds.mean(dim=member_dim, skipna=skipna)

    # 更新元数据
    new_semantic = bundle.semantic
    new_semantic.member_count = None  # 不再有成员

    new_provenance = bundle.provenance
    new_provenance.transform_chain = bundle.provenance.transform_chain + [
        f"ensemble_mean_over_{member_dim}"
    ]

    return DataBundle(
        payload=mean_ds,
        kind=bundle.kind,
        source_id=bundle.source_id,
        standard_vars=bundle.standard_vars,
        semantic=new_semantic,
        provenance=new_provenance,
        quality=bundle.quality,
    )


class EnsembleMeanTransform:
    """集合均值变换：输入和输出都是 DataBundle。

    连续场评测在配对前先降到集合均值；确定性场原样返回。
    """

    def __init__(self, member_dim: str = "member"):
        self.member_dim = member_dim

    def transform(self, bundle: DataBundle) -> DataBundle:
        return compute_ensemble_mean(bundle, member_dim=self.member_dim)


def compute_latitude_weights(ds: xr.Dataset) -> xr.DataArray:
    """
    计算纬度余弦权重

    Args:
        ds: 包含 'lat' 坐标的数据集

    Returns:
        权重 DataArray，维度与 (lat, lon) 一致
    """
    if "lat" not in ds.coords:
        raise ContractError("Dataset must have 'lat' coordinate")

    lat = ds.coords["lat"]
    weights = np.cos(np.deg2rad(lat))

    # 扩展到 (lat, lon)
    if "lon" in ds.coords:
        weights, _ = xr.broadcast(weights, ds.coords["lon"])

    return weights
