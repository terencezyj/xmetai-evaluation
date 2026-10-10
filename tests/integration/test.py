# -*- coding: utf-8 -*-
"""FDP 要素检验（Fengqing × CRA40）真实数据档：与标准脚本产物对表。

跑两条真配置，产物与 ``D:\\xmetai-evalation\\fdp\\`` 下两个标准脚本的输出逐列比：

- ``ensemble_verifier.py``           → ``GOLDEN_ENSEMBLE``
- ``activity_spectrum_verifier.py``  → ``GOLDEN_ACTIVITY`` / ``GOLDEN_SPECTRUM``

标准答案是**内联**在下面的（三份加起来 4.6 KB，不值得单独存个数据目录）。数值是
从标准脚本落下的三个 CSV 原样抄进来的，没做任何四舍五入；那三份 CSV 还在
``tmp_data/ref_spectrum/``（服务器上是 ``DATA_ROOT/ref_spectrum/``），是出处。

fixture 在 ``DATA_ROOT``（见下），是一份**裁剪过的最小样例**：起报 20260820、
时效 6/12/18；预报 21 成员、实况 3 个时刻、气候态是编的。目录不齐（或没装
netCDF4 / cfgrib）时整个模块 skip；平时用 ``pytest -m realdata`` 选中才跑。
本文件名是 ``test.py``，需显式运行：
``python -m pytest tests/integration/test.py -m realdata -v -s``。
也可以直接运行本文件，入口会逐步执行 ens、single 评测与断言，便于调试。
产物保存在项目的 ``output/ens`` 和 ``output/single``。

对表容差 ``REL_TOL``：实测最差 6.8e-6。那点差全出在单位换算——框架
``io/layouts.py:212`` 的 z500 写 ``scale=1.0/GRAVITY``（乘 float32 倒数），标准脚本
是 ``z / GRAVITY``（除），``float32(1/9.80665)`` 比 ``1/9.80665`` 单向偏低 2.4e-8，
于是整场带一个系统性尺度偏移。布局是 weather 系列共用的，不为一万分之一去动它，
所以容差放到 1e-4（比实测最差宽 15 倍），只拦真回归。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.realdata
if __name__ != "__main__":
    pytest.importorskip("netCDF4")
    pytest.importorskip("cfgrib")

#: 最小 fixture 根：预报 fengqing/ + 实况 cra_root/ + 气候态 cli_fake/ 三件套都
#: 在它下面。换机器/换位置改这一行（配置里的 ``_DATA_ROOT`` 指的是同一处）。
DATA_ROOT = Path("/workspace/data/shenzw/fdp_test_data")
OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output"

#: 起报时刻与时效——配置里写死的，golden 也是照这个口径出的。INIT_ISO 是
#: ``scores.csv`` 里 ``init_time`` 列的写法（格式照抄，别改成别的形式）。
INIT_ISO = "2026-08-20T00:00:00.000000"
LEADS = (6, 12, 18)

#: 0.25° 全球网格点数；每个 (metric, lead) 格应当正好覆盖这么多点。
GRID_POINTS = 721 * 1440

#: 对表容差，见模块 docstring。
REL_TOL = 1e-4

#: ens 配置对表的四个误差族指标，golden 列名与框架 metric 名恰好同名。
ENS_COLUMNS = ("crps", "spread", "rmse", "spread_error_ratio")

#: 活跃度三列：golden 列名 -> 框架 metric 名。golden 没有 activity_bias（那是
#: fc_activity - obs_activity 的派生量），只在覆盖面里查一下存在。
ACTIVITY_COLUMNS = {
    "activity_ratio": "activity_ratio",
    "fc_activity": "activity_forecast",
    "obs_activity": "activity_observation",
}


# --------------------------------------------------------------------------
# 标准答案（内联；来源见模块 docstring）
# --------------------------------------------------------------------------

#: ``ensemble_verifier.py`` 的 ``ensemble_verification_2026082000.csv``。
GOLDEN_ENSEMBLE = {
    6: {
        "crps": 3.030698164066889,
        "spread": 2.350909948348999,
        "rmse": 4.886978626251221,
        "spread_error_ratio": 0.4810559096208635,
    },
    12: {
        "crps": 3.144235071187685,
        "spread": 2.8866982460021973,
        "rmse": 5.096440315246582,
        "spread_error_ratio": 0.566414608519266,
    },
    18: {
        "crps": 3.19647544233606,
        "spread": 3.446047067642212,
        "rmse": 5.442727565765381,
        "spread_error_ratio": 0.6331470803936176,
    },
}

#: ``activity_spectrum_verifier.py`` 的 ``activity_ratio_2026082000.csv``。
GOLDEN_ACTIVITY = {
    6: {
        "activity_ratio": 0.9984731078147888,
        "fc_activity": 265.7129821777344,
        "obs_activity": 266.11932373046875,
    },
    12: {
        "activity_ratio": 0.9968142509460448,
        "fc_activity": 264.0156555175781,
        "obs_activity": 264.85943603515625,
    },
    18: {
        "activity_ratio": 0.9973190426826476,
        "fc_activity": 261.57635498046875,
        "obs_activity": 262.2795104980469,
    },
}

#: ``activity_spectrum_verifier.py`` 的 ``power_spectrum_2026082000.csv``，**按起报
#: 时效平均后**的 P1..P30（标准脚本每小时一行，框架的 ``diagnostics/spectrum_z500.csv``
#: 是时效维平均后的单条曲线，所以先平均再比）。k=0 那列框架不存，丢掉。
GOLDEN_SPECTRUM = {
    "forecast": (
        602057836131054.9,
        1.2488027576399048e17,
        807979311525485.4,
        1.6420130552952886e16,
        1010865679762588.9,
        2749974009958590.5,
        883489351224695.6,
        2199338762714431.2,
        384668495464970.7,
        1710851339258180.8,
        226638010045345.75,
        807108072970235.0,
        118580330260528.05,
        422046234790953.8,
        82967250180868.1,
        294094593373972.56,
        41559749690681.13,
        219543220651043.66,
        31973249144773.906,
        171883638838190.03,
        24354744430155.02,
        110617312461897.38,
        12245064642216.703,
        109701306054021.64,
        13317854388573.172,
        112036754775453.05,
        7307808410876.227,
        86425389406370.06,
        4899362090951.778,
        68379130231360.77,
    ),
    "observation": (
        590475791453331.9,
        1.256451034130574e17,
        801732460696704.5,
        1.6325055978468918e16,
        995167719105104.0,
        2683966357194221.0,
        880020156681322.6,
        2117717428008450.8,
        381894680673279.8,
        1686591012104204.0,
        220602211506857.66,
        794462269191437.6,
        121316389771244.62,
        417930748933442.2,
        86794081875033.77,
        295278230464185.06,
        41906289281946.836,
        215244837824897.66,
        33155361335004.8,
        167933403979950.34,
        26263735777208.625,
        107676127259620.23,
        12876831557649.65,
        108578799548685.48,
        14338496641093.963,
        110641032963001.56,
        7801803305996.925,
        86926841288547.14,
        5214079539673.633,
        67860116820273.62,
    ),
}

_missing = [sub for sub in ("fengqing", "cra_root", "cli_fake") if not (DATA_ROOT / sub).is_dir()]
if _missing and __name__ != "__main__":
    pytest.skip(
        f"fixture 不存在: {DATA_ROOT}（缺 {', '.join(_missing)}/）", allow_module_level=True
    )


# --------------------------------------------------------------------------
# 跑链
# --------------------------------------------------------------------------


def _run(config_name: str, output_dir: Path) -> Path:
    """按真实配置走清晰的串行流程，只覆盖产物目录。

    数据根**不覆盖**——配置里的 ``_DATA_ROOT`` 指的就是这份 fixture。
    每次从零读取和计算，不使用原执行器的切块、并发与 resume 状态。
    指标直接由 cfg.metric_options 选择，不读取 cfg.pipeline 对应的模板。
    """
    from xmetai_evaluation.configs.base import load_config
    from xmetai_evaluation.pipeline.serial import run_serial_evaluation

    cfg = load_config(config_name)
    assert not isinstance(cfg, list), f"{config_name} 应导出单个 cfg"
    cfg.output_dir = str(output_dir)
    run_serial_evaluation(cfg)
    return Path(output_dir)


@pytest.fixture(scope="session")
def ens():
    """ens 配置的全部指标串行跑一次，整模块共用。"""
    from tests import runner

    root = _run("fdp_rmse_ens_fengqing", OUTPUT_ROOT / "ens")
    return root, runner.read_scores(root)


@pytest.fixture(scope="session")
def single():
    """single 配置的确定性指标串行跑一次，整模块共用。"""
    from tests import runner

    root = _run("fdp_rmse_single_fengqing", OUTPUT_ROOT / "single")
    return root, runner.read_scores(root)


# --------------------------------------------------------------------------
# 取数与比对
# --------------------------------------------------------------------------


def _value(scores: pd.DataFrame, metric: str, lead: int) -> float:
    """取 ``(metric, z500, lead)`` 那一格的 value。

    ``region`` 必须为空：档案里混着 tropics 行，不过滤就会取到别处的数。

    ens 配置的 ``rmse`` 会出两行：一行来自 rmse 指标，另一行是 spread_error
    输出的集合均值 RMSE。允许浮点累加顺序带来的微小差异，然后取第一行。
    """
    rows = scores[
        (scores["metric"] == metric)
        & (scores["variable"] == "z500")
        & (scores["lead_h"] == float(lead))
        & scores["region"].isna()
    ]
    assert len(rows) >= 1, f"scores.csv 里没有 {metric} @ {lead}h"
    values = rows["value"].to_numpy(dtype=float)
    spread = (values.max() - values.min()) / abs(values[0])
    assert spread <= REL_TOL, (
        f"{metric} @ {lead}h 的两行互相矛盾（相对差 {spread:.3e}）: {values!r}"
    )
    return float(values[0])


def _assert_rel(actual: float, expected: float, label: str) -> None:
    """相对差断言；``expected`` 是标准脚本的数。"""
    relative = abs(actual - expected) / abs(expected)
    assert relative <= REL_TOL, (
        f"{label}: 框架 {actual!r} vs 标准脚本 {expected!r}，相对差 {relative:.3e}"
    )


def _assert_spectrum(out_dir: Path) -> None:
    """``diagnostics/spectrum_z500.csv`` 的 P1..P30 两列对表。

    框架那份 CSV 只写到 6 位有效数字，比不出比 1e-5 更细的东西——``REL_TOL`` 罩得住。
    """
    ours = pd.read_csv(out_dir / "diagnostics" / "spectrum_z500.csv")
    assert sorted(ours["wavenumber"]) == list(range(1, 31)), "诊断谱的波数不是 1..30"
    for kind, column in (("forecast", "pred_mean"), ("observation", "obs_mean")):
        expected = GOLDEN_SPECTRUM[kind]  # P1 在 [0]
        for k in range(1, 31):
            actual = float(ours.loc[ours["wavenumber"] == k, column].iloc[0])
            _assert_rel(actual, expected[k - 1], f"谱 P{k}（{kind}）")


def _assert_activity(scores: pd.DataFrame) -> None:
    for lead in LEADS:
        for column, metric in ACTIVITY_COLUMNS.items():
            _assert_rel(
                _value(scores, metric, lead),
                GOLDEN_ACTIVITY[lead][column],
                f"活跃度 {column} @ {lead}h",
            )


# --------------------------------------------------------------------------
# 用例
# --------------------------------------------------------------------------


def test_ens_error_parity(ens):
    """ens：crps / spread / rmse / spread-error 比，四指标 × 三时效逐格对表。"""
    _, scores = ens
    for lead in LEADS:
        for column in ENS_COLUMNS:
            _assert_rel(
                _value(scores, column, lead),
                GOLDEN_ENSEMBLE[lead][column],
                f"ens {column} @ {lead}h",
            )


def test_ens_activity_parity(ens):
    """ens：活跃度比三列（ratio / forecast / observation）对表。"""
    _assert_activity(ens[1])


def test_ens_spectrum_parity(ens):
    """ens：功率谱 P1..P30（预报 + 实况两列）对表。"""
    _assert_spectrum(ens[0])


def test_single_parity(single, ens):
    """single：rmse + 活跃度 + 谱对表，且与 ens 的确定性指标逐位相同。

    single 的 metric_options 没有 crps / spread_error，所以不产生集合指标；
    rmse 与谱是同一批样本上同一套公式，实测与 ens **逐位相同**（``array_equal``），
    这里把这个性质钉住——它保证"加不加 crps 段"不影响确定性指标。
    """
    out, scores = single
    ens_out, ens_scores = ens

    for lead in LEADS:
        ours = _value(scores, "rmse", lead)
        _assert_rel(ours, GOLDEN_ENSEMBLE[lead]["rmse"], f"single rmse @ {lead}h")
        assert ours == _value(ens_scores, "rmse", lead), f"single/ens 的 rmse @ {lead}h 不逐位相同"

    assert "crps" not in set(scores["metric"]), "single 配置不该出 crps"
    _assert_activity(scores)
    _assert_spectrum(out)

    ours_spectrum = pd.read_csv(out / "diagnostics" / "spectrum_z500.csv")
    ens_spectrum = pd.read_csv(ens_out / "diagnostics" / "spectrum_z500.csv")
    for column in ("pred_mean", "obs_mean"):
        assert np.array_equal(ours_spectrum[column], ens_spectrum[column]), (
            f"single/ens 的诊断谱 {column} 不逐位相同"
        )


def test_scores_coverage(ens):
    """兜底：样本数、状态、以及 golden 之外那几个指标的完整性。

    golden 里没有 acc / bias / spectrum_power_ratio / activity_bias（标准脚本不算
    这些），对不了表；这里只保证它们出得来、覆盖全网格——漏算或静默出假数的回归
    至少能被这条拦住。
    """
    _, scores = ens
    assert set(scores["variable"]) == {"z500"}, "本配置只评 z500"
    assert scores["region"].isna().all(), "不该出现分区行"
    assert (scores["status"] == "success").all()
    assert set(scores["lead_h"]) == {float(lead) for lead in LEADS}

    # 逐格指标覆盖整张网格；``spectrum_power_ratio`` 是谱的汇总量（group=summary），
    # 一个时效一个样本，别按全网格去要求它。
    per_grid = scores[scores["metric"] != "spectrum_power_ratio"]
    assert (per_grid["n_valid"] == GRID_POINTS).all()
    summary = scores[scores["metric"] == "spectrum_power_ratio"]
    assert (summary["group"] == "summary").all() and (summary["n_valid"] == 1).all()

    # 时段与 fixture 对得上：起报 2026082000，有效时刻 0820 的 06/12/18。
    assert set(scores["init_time"]) == {INIT_ISO}
    assert set(scores["valid_time"]) == {f"2026-08-20T{lead:02d}:00:00.000000" for lead in LEADS}

    required = {
        "rmse",
        "bias",
        "acc",
        "crps",
        "spread",
        "spread_error_ratio",
        "activity_ratio",
        "activity_forecast",
        "activity_observation",
        "activity_bias",
    }
    assert required <= set(scores["metric"]), f"缺指标: {sorted(required - set(scores['metric']))}"

    # acc 用的是 FDP/WeatherBench2 的 uncentered 口径（配置里 centered=False），
    # 在编的气候态上会顶到接近 1；只要求它有值、不是常数零场兜底出来的。
    for lead in LEADS:
        assert 0.0 < _value(scores, "acc", lead) <= 1.0


def main() -> None:
    """直接调试：加载配置、串行评测、读取文件、逐项核对结果。"""
    if _missing:
        raise FileNotFoundError(f"fixture 不存在: {DATA_ROOT}（缺 {', '.join(_missing)}/）")

    # 直接运行本文件时，确保导入项目源码中的 serial.py。
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from xmetai_evaluation.configs.base import load_config
    from xmetai_evaluation.pipeline.serial import run_serial_evaluation

    # 1. 集合评测：串行入口直接按 cfg.metric_options 运行全部指标。
    ens_cfg = load_config("fdp_rmse_ens_fengqing")
    ens_output = OUTPUT_ROOT / "ens"
    ens_cfg.output_dir = str(ens_output)
    run_serial_evaluation(ens_cfg)

    # 2. 读取实际产物，与标准答案逐项对比。
    ens_scores = pd.read_csv(ens_output / "scores.csv")
    ens_result = (ens_output, ens_scores)
    test_ens_error_parity(ens_result)
    test_ens_activity_parity(ens_result)
    test_ens_spectrum_parity(ens_result)
    test_scores_coverage(ens_result)

    # 3. 单模型评测：检查它与集合评测的确定性结果一致。
    single_cfg = load_config("fdp_rmse_single_fengqing")
    single_output = OUTPUT_ROOT / "single"
    single_cfg.output_dir = str(single_output)
    run_serial_evaluation(single_cfg)
    single_scores = pd.read_csv(single_output / "scores.csv")
    test_single_parity((single_output, single_scores), ens_result)

    print(f"对表测试通过；产物位于 {ens_output} 和 {single_output}", flush=True)


if __name__ == "__main__":
    main()
