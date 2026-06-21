"""
공정능력분석 & 통계적공정관리(SPC) 대시보드
산업공학 과제 - 강의록(공정능력분석, 통계적공정관리) 기반 단일 파일 Streamlit 앱

실행 방법: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy.stats import shapiro, boxcox, norm
from scipy.special import gamma
from scipy import stats as sstats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="공정능력 · SPC 분석 대시보드", page_icon="📊", layout="wide")


# =====================================================================
# 1. 불편화 상수 (Unbiased Constants)
#    강의록 표(N=2~25)를 딕셔너리로 내장. 범위를 벗어나면 근사식 사용.
# =====================================================================

# 공정능력분석용: c4, d2, d3, d4
_CAP_TABLE = {
    2: dict(c4=0.797885, d2=1.128, d3=0.8525, d4=0.954),
    3: dict(c4=0.886227, d2=1.693, d3=0.8884, d4=1.588),
    4: dict(c4=0.921318, d2=2.059, d3=0.8798, d4=1.978),
    5: dict(c4=0.939986, d2=2.326, d3=0.8641, d4=2.257),
    6: dict(c4=0.951533, d2=2.534, d3=0.8480, d4=2.472),
    7: dict(c4=0.959369, d2=2.704, d3=0.8332, d4=2.645),
    8: dict(c4=0.965030, d2=2.847, d3=0.8198, d4=2.791),
    9: dict(c4=0.969311, d2=2.970, d3=0.8078, d4=2.915),
    10: dict(c4=0.972659, d2=3.078, d3=0.7971, d4=3.024),
    11: dict(c4=0.975350, d2=3.173, d3=0.7873, d4=3.121),
    12: dict(c4=0.977559, d2=3.258, d3=0.7785, d4=3.207),
    13: dict(c4=0.979406, d2=3.336, d3=0.7704, d4=3.285),
    14: dict(c4=0.980971, d2=3.407, d3=0.7630, d4=3.356),
    15: dict(c4=0.982316, d2=3.472, d3=0.7562, d4=3.422),
    16: dict(c4=0.983484, d2=3.532, d3=0.7499, d4=3.482),
    17: dict(c4=0.984506, d2=3.588, d3=0.7441, d4=3.538),
    18: dict(c4=0.985410, d2=3.640, d3=0.7386, d4=3.591),
    19: dict(c4=0.986214, d2=3.689, d3=0.7335, d4=3.640),
    20: dict(c4=0.986934, d2=3.735, d3=0.7287, d4=3.686),
    21: dict(c4=0.987583, d2=3.778, d3=0.7242, d4=3.730),
    22: dict(c4=0.988170, d2=3.819, d3=0.7199, d4=3.771),
    23: dict(c4=0.988705, d2=3.858, d3=0.7159, d4=3.811),
    24: dict(c4=0.989193, d2=3.895, d3=0.7121, d4=3.847),
    25: dict(c4=0.989640, d2=3.931, d3=0.7084, d4=3.883),
}

# 관리도용: A2, A3, D3, D4, B3, B4 (d2는 위 테이블 재사용)
_CTRL_TABLE = {
    2: dict(A2=1.880, A3=2.659, D3=0, D4=3.267, B3=0, B4=3.267),
    3: dict(A2=1.023, A3=1.954, D3=0, D4=2.574, B3=0, B4=2.568),
    4: dict(A2=0.729, A3=1.628, D3=0, D4=2.282, B3=0, B4=2.266),
    5: dict(A2=0.577, A3=1.427, D3=0, D4=2.114, B3=0, B4=2.089),
    6: dict(A2=0.483, A3=1.287, D3=0, D4=2.004, B3=0.030, B4=1.970),
    7: dict(A2=0.419, A3=1.182, D3=0.076, D4=1.924, B3=0.118, B4=1.882),
    8: dict(A2=0.373, A3=1.099, D3=0.136, D4=1.864, B3=0.185, B4=1.815),
    9: dict(A2=0.337, A3=1.032, D3=0.184, D4=1.816, B3=0.239, B4=1.761),
    10: dict(A2=0.308, A3=0.975, D3=0.223, D4=1.777, B3=0.284, B4=1.716),
    11: dict(A2=0.285, A3=0.927, D3=0.256, D4=1.744, B3=0.321, B4=1.679),
    12: dict(A2=0.266, A3=0.886, D3=0.283, D4=1.717, B3=0.354, B4=1.646),
    13: dict(A2=0.249, A3=0.850, D3=0.307, D4=1.693, B3=0.382, B4=1.618),
    14: dict(A2=0.235, A3=0.817, D3=0.328, D4=1.672, B3=0.406, B4=1.594),
    15: dict(A2=0.223, A3=0.789, D3=0.347, D4=1.653, B3=0.428, B4=1.572),
    16: dict(A2=0.212, A3=0.763, D3=0.363, D4=1.637, B3=0.448, B4=1.552),
    17: dict(A2=0.203, A3=0.739, D3=0.378, D4=1.622, B3=0.466, B4=1.534),
    18: dict(A2=0.194, A3=0.718, D3=0.391, D4=1.608, B3=0.482, B4=1.518),
    19: dict(A2=0.187, A3=0.698, D3=0.403, D4=1.597, B3=0.497, B4=1.503),
    20: dict(A2=0.180, A3=0.680, D3=0.415, D4=1.585, B3=0.510, B4=1.490),
    21: dict(A2=0.173, A3=0.663, D3=0.425, D4=1.575, B3=0.523, B4=1.477),
    22: dict(A2=0.167, A3=0.647, D3=0.434, D4=1.566, B3=0.534, B4=1.466),
    23: dict(A2=0.162, A3=0.633, D3=0.443, D4=1.557, B3=0.545, B4=1.455),
    24: dict(A2=0.157, A3=0.619, D3=0.451, D4=1.548, B3=0.555, B4=1.445),
    25: dict(A2=0.153, A3=0.606, D3=0.459, D4=1.541, B3=0.565, B4=1.435),
}


def calc_unbiased_const(name, n):
    """공정능력분석용 불편화 상수. 테이블 범위를 벗어나면 근사식 사용."""
    n = max(2, round(n))
    if name == "c4":
        if n in _CAP_TABLE:
            return _CAP_TABLE[n]["c4"]
        return (np.sqrt(2) / np.sqrt(n - 1)) * (gamma(n / 2) / gamma((n - 1) / 2))
    if name == "d2":
        if n in _CAP_TABLE:
            return _CAP_TABLE[n]["d2"]
        if n < 51:
            nn = min(max(n, 2), 25)
            return _CAP_TABLE[nn]["d2"]
        return 3.4873 + 0.0250141 * n - 0.00009823 * n ** 2
    if name == "d3":
        if n in _CAP_TABLE:
            return _CAP_TABLE[n]["d3"]
        nn = min(max(n, 2), 25)
        return _CAP_TABLE[nn]["d3"]
    if name == "d4":
        if n in _CAP_TABLE:
            return _CAP_TABLE[n]["d4"]
        nn = min(max(n, 2), 25)
        return _CAP_TABLE[nn]["d4"]
    return None


def unbiased_coefficient(name, m):
    """관리도 작성용 불편화 상수 (A2, A3, D3, D4, B3, B4). 범위 벗어나면 근사식."""
    m_round = max(2, round(m))
    if m_round in _CTRL_TABLE and name in _CTRL_TABLE[m_round]:
        return _CTRL_TABLE[m_round][name]

    d2_val = calc_unbiased_const("d2", m_round)
    d3_val = calc_unbiased_const("d3", m_round)
    c4_val = calc_unbiased_const("c4", m_round)
    if name == "A2":
        return 3 / (d2_val * np.sqrt(m_round))
    if name == "A3":
        return 3 / (c4_val * np.sqrt(m_round))
    if name == "D3":
        return max(0, 1 - 3 * (d3_val / d2_val))
    if name == "D4":
        return 1 + 3 * (d3_val / d2_val)
    if name == "B3":
        return max(0, 1 - 3 * np.sqrt(1 - c4_val ** 2) / c4_val)
    if name == "B4":
        return 1 + 3 * np.sqrt(1 - c4_val ** 2) / c4_val
    return np.nan


# =====================================================================
# 2. 공정능력분석 (Process Capability Analysis)
# =====================================================================

def normality_test(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return {"stat": np.nan, "p": np.nan, "is_normal": None}
    stat, p = shapiro(values)
    return {"stat": stat, "p": p, "is_normal": bool(p >= alpha)}


def qq_plot_data(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    z = sstats.zscore(values)
    (x, y), _ = sstats.probplot(z, dist="norm")
    return x, y


def process_capability(df, group_col, value_col, USL, LSL):
    """강의록의 process_capability(data, LSL, USL) 함수와 동일한 로직."""
    data = df[[group_col, value_col]].dropna()

    mean_sg = data.groupby(group_col)[value_col].mean()
    sigma_sg = data.groupby(group_col)[value_col].std()
    sizes_sg = data.groupby(group_col)[value_col].size()

    x_bar = data[value_col].mean()
    sigma_hat = data[value_col].std(ddof=1)
    n_total = len(data)
    sigma_overall = sigma_hat / calc_unbiased_const("c4", n_total)

    valid_groups = sigma_sg[sizes_sg >= 2]
    valid_sizes = sizes_sg[sizes_sg >= 2]
    k = len(valid_groups)

    if k == 0:
        sorted_vals = data[value_col].values
        mr = np.abs(np.diff(sorted_vals))
        mr_bar = np.nanmean(mr) if len(mr) else np.nan
        sigma_within = mr_bar / 1.128
    else:
        sigma_p = np.sqrt(np.sum(valid_groups ** 2) / k)
        total_n = int(valid_sizes.sum())
        d_value = total_n - k + 1
        sigma_within = sigma_p / calc_unbiased_const("c4", d_value)

    Cp = (USL - LSL) / (6 * sigma_within)
    Cpk = min((USL - x_bar) / (3 * sigma_within), (x_bar - LSL) / (3 * sigma_within))
    Pp = (USL - LSL) / (6 * sigma_overall)
    Ppk = min((USL - x_bar) / (3 * sigma_overall), (x_bar - LSL) / (3 * sigma_overall))
    Cpu = (USL - x_bar) / (3 * sigma_within)
    Cpl = (x_bar - LSL) / (3 * sigma_within)

    return {
        "n": n_total, "k": len(mean_sg), "x_bar": x_bar,
        "sigma_hat": sigma_hat, "sigma_overall": sigma_overall, "sigma_within": sigma_within,
        "Cp": Cp, "Cpk": Cpk, "Pp": Pp, "Ppk": Ppk, "Cpu": Cpu, "Cpl": Cpl,
        "mean_sg": mean_sg, "sigma_sg": sigma_sg, "USL": USL, "LSL": LSL,
    }


def grade_capability(cpk):
    """강의록 5단계 공정능력 판정 등급표."""
    if cpk is None or (isinstance(cpk, float) and np.isnan(cpk)):
        return None
    if cpk >= 1.67:
        return dict(grade=0, label="매우 충분", color="#4ADE80", sigma="±5σ",
                     action="들쭉날쭉이 약간 커져도 걱정할 필요가 없다. 비용절감이나 관리의 간소화를 생각하도록 한다.")
    if cpk >= 1.33:
        return dict(grade=1, label="충분", color="#34D399", sigma="±4σ",
                     action="아주 이상적인 공정상황이므로 현재의 상태를 유지한다.")
    if cpk >= 1.00:
        return dict(grade=2, label="충분하지는 않지만 괜찮음", color="#FBBF24", sigma="±3σ",
                     action="공정관리를 확실하게 하여 관리상태를 유지할 것. Cp가 1에 가까워지면 불량발생 가능성이 있으므로 주의해야 한다.")
    if cpk >= 0.67:
        return dict(grade=3, label="모자람", color="#FB923C", sigma="±2σ",
                     action="불량품이 발생되고 있다. 전체 선별, 공정의 개선, 관리가 필요하다.")
    return dict(grade=4, label="매우 부족", color="#EF4444", sigma="±1σ",
                 action="품질이 전혀 만족스럽지 않다. 서둘러 현황조사, 원인규명, 품질개선 같은 긴급 대책을 펴야 한다.")


def estimate_defect_rate(x_bar, sigma_within, USL, LSL):
    z_usl = (USL - x_bar) / sigma_within
    z_lsl = (LSL - x_bar) / sigma_within
    p_above = 1 - norm.cdf(z_usl)
    p_below = norm.cdf(z_lsl)
    total = p_above + p_below
    return {"p_above_usl": p_above, "p_below_lsl": p_below, "total": total, "ppm": total * 1e6}


def boxcox_transform(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if np.any(values <= 0):
        return None, None, "Box-Cox 변환은 양수 데이터에만 적용 가능합니다."
    transformed, lam = boxcox(values)
    return transformed, lam, None


# =====================================================================
# 3. 통계적공정관리(SPC) 관리도
# =====================================================================

def xbar_r_chart(df, group_col, value_col):
    grouped = df.groupby(group_col)[value_col]
    xbar = grouped.mean()
    r = grouped.apply(lambda s: s.max() - s.min())
    n_avg = grouped.size().mean()
    xbar_bar, r_bar = xbar.mean(), r.mean()
    A2v, D3v, D4v = unbiased_coefficient("A2", n_avg), unbiased_coefficient("D3", n_avg), unbiased_coefficient("D4", n_avg)

    x_chart = pd.DataFrame({"group": xbar.index, "value": xbar.values,
                             "CL": xbar_bar, "UCL": xbar_bar + A2v * r_bar, "LCL": xbar_bar - A2v * r_bar})
    r_chart = pd.DataFrame({"group": r.index, "value": r.values,
                             "CL": r_bar, "UCL": D4v * r_bar, "LCL": D3v * r_bar})
    return x_chart, r_chart, {"xbar_bar": xbar_bar, "r_bar": r_bar, "n_avg": n_avg}


def xbar_s_chart(df, group_col, value_col):
    grouped = df.groupby(group_col)[value_col]
    xbar = grouped.mean()
    s = grouped.std()
    n_avg = grouped.size().mean()
    xbar_bar, s_bar = xbar.mean(), s.mean()
    A3v, B3v, B4v = unbiased_coefficient("A3", n_avg), unbiased_coefficient("B3", n_avg), unbiased_coefficient("B4", n_avg)

    x_chart = pd.DataFrame({"group": xbar.index, "value": xbar.values,
                             "CL": xbar_bar, "UCL": xbar_bar + A3v * s_bar, "LCL": xbar_bar - A3v * s_bar})
    s_chart = pd.DataFrame({"group": s.index, "value": s.values,
                             "CL": s_bar, "UCL": B4v * s_bar, "LCL": B3v * s_bar})
    return x_chart, s_chart, {"xbar_bar": xbar_bar, "s_bar": s_bar, "n_avg": n_avg}


def i_mr_chart(df, group_col, value_col, window=2):
    sg = df[[group_col, value_col]].dropna().reset_index(drop=True)
    values = sg[value_col].values
    n = len(values)
    w = window

    mr_i = np.full(n, np.nan)
    for i in range(w - 1, n):
        window_vals = values[i - w + 1: i + 1]
        mr_i[i] = window_vals.max() - window_vals.min()

    mr_bar = np.nanmean(mr_i)
    xbar = values.mean()
    D3v, D4v, d2v = unbiased_coefficient("D3", w), unbiased_coefficient("D4", w), calc_unbiased_const("d2", w)

    i_chart = pd.DataFrame({"group": sg[group_col].values, "value": values,
                             "CL": xbar, "LCL": xbar - 3 * mr_bar / d2v, "UCL": xbar + 3 * mr_bar / d2v})
    mr_chart = pd.DataFrame({"group": sg[group_col].values, "value": mr_i,
                              "CL": mr_bar, "LCL": D3v * mr_bar, "UCL": D4v * mr_bar})
    return i_chart, mr_chart, {"xbar": xbar, "mr_bar": mr_bar, "window": w}


def np_chart(df, group_col, sample_col, count_col):
    data = df[[group_col, sample_col, count_col]].dropna()
    k = len(data)
    np_bar = data[count_col].sum() / k
    p_bar = data[count_col].sum() / data[sample_col].sum()
    sigma = np.sqrt(np_bar * (1 - p_bar))
    chart = pd.DataFrame({"group": data[group_col].values, "value": data[count_col].values,
                           "CL": np_bar, "UCL": np_bar + 3 * sigma, "LCL": max(0, np_bar - 3 * sigma)})
    return chart, {"np_bar": np_bar, "p_bar": p_bar}


def p_chart(df, group_col, sample_col, count_col):
    data = df[[group_col, sample_col, count_col]].dropna()
    p_bar = data[count_col].sum() / data[sample_col].sum()
    sigma = np.sqrt(p_bar * (1 - p_bar) / data[sample_col])
    chart = pd.DataFrame({"group": data[group_col].values,
                           "value": (data[count_col] / data[sample_col]).values,
                           "CL": p_bar, "UCL": np.minimum(1, p_bar + 3 * sigma).values,
                           "LCL": np.maximum(0, p_bar - 3 * sigma).values})
    return chart, {"p_bar": p_bar}


def c_chart(df, group_col, count_col):
    data = df[[group_col, count_col]].dropna()
    c_bar = data[count_col].mean()
    sigma = np.sqrt(c_bar)
    chart = pd.DataFrame({"group": data[group_col].values, "value": data[count_col].values,
                           "CL": c_bar, "UCL": c_bar + 3 * sigma, "LCL": max(0, c_bar - 3 * sigma)})
    return chart, {"c_bar": c_bar}


def u_chart(df, group_col, sample_col, count_col):
    data = df[[group_col, sample_col, count_col]].dropna()
    u_bar = data[count_col].sum() / data[sample_col].sum()
    sigma = np.sqrt(u_bar / data[sample_col])
    chart = pd.DataFrame({"group": data[group_col].values,
                           "value": (data[count_col] / data[sample_col]).values,
                           "CL": u_bar, "UCL": (u_bar + 3 * sigma).values,
                           "LCL": np.maximum(0, u_bar - 3 * sigma).values})
    return chart, {"u_bar": u_bar}


def find_out_of_control(chart_df):
    mask = (chart_df["value"] > chart_df["UCL"]) | (chart_df["value"] < chart_df["LCL"])
    return chart_df.loc[mask.fillna(False), "group"].tolist()


def nelson_rules(chart_df):
    """Nelson's Rule 일부 구현: Rule1(이탈), Rule2(9점 연속 한쪽), Rule3(6점 연속 추세)."""
    n = len(chart_df)
    values = chart_df["value"].values.astype(float)
    cl, ucl, lcl = chart_df["CL"].values, chart_df["UCL"].values, chart_df["LCL"].values
    flags = [[] for _ in range(n)]

    for i in range(n):
        if pd.notna(values[i]) and (values[i] > ucl[i] or values[i] < lcl[i]):
            flags[i].append("Rule1: 관리한계 이탈")

    for i in range(8, n):
        window = values[i - 8: i + 1]
        if np.all(~np.isnan(window)):
            if np.all(window > cl[i]):
                flags[i].append("Rule2: 9점 연속 중심선 위")
            elif np.all(window < cl[i]):
                flags[i].append("Rule2: 9점 연속 중심선 아래")

    for i in range(5, n):
        window = values[i - 5: i + 1]
        if np.all(~np.isnan(window)):
            diffs = np.diff(window)
            if np.all(diffs > 0):
                flags[i].append("Rule3: 6점 연속 증가")
            elif np.all(diffs < 0):
                flags[i].append("Rule3: 6점 연속 감소")
    return flags


def current_status_summary(chart_df, flags, recent_n=5):
    """가장 최근 시점 기준 '현재 공정 상태'를 한눈에 판단할 수 있는 요약 진단.
    - 최근 시점이 관리상태인지(정상/이탈)
    - 최근 recent_n개 구간의 추세(상승/하강/안정)
    - 최근 recent_n개 중 이상신호 발생 횟수
    """
    n = len(chart_df)
    last = chart_df.iloc[-1]
    last_flags = flags[-1] if flags else []
    is_last_ooc = bool(last_flags) or (pd.notna(last["value"]) and (last["value"] > last["UCL"] or last["value"] < last["LCL"]))

    recent = chart_df.tail(min(recent_n, n))
    recent_vals = recent["value"].dropna().values
    if len(recent_vals) >= 3:
        diffs = np.diff(recent_vals)
        if np.all(diffs > 0):
            trend = "상승"
        elif np.all(diffs < 0):
            trend = "하강"
        else:
            trend = "안정"
    else:
        trend = "데이터 부족"

    recent_flag_count = sum(1 for f in flags[-min(recent_n, n):] if f)

    return {
        "is_ooc": is_last_ooc,
        "last_group": last["group"],
        "last_value": last["value"],
        "trend": trend,
        "recent_flag_count": recent_flag_count,
        "recent_n": min(recent_n, n),
    }


def render_status_strip(summary):
    """현재 공정 상태 요약을 카드 3개로 시각화."""
    status_color = "#EF4444" if summary["is_ooc"] else "#4ADE80"
    status_text = "이탈 / 이상" if summary["is_ooc"] else "정상 (관리상태)"

    trend_icon = {"상승": "▲", "하강": "▼", "안정": "■", "데이터 부족": "·"}.get(summary["trend"], "·")
    trend_color = "#FBBF24" if summary["trend"] in ("상승", "하강") else "#8B98A9"

    flag_color = "#EF4444" if summary["recent_flag_count"] > 0 else "#4ADE80"

    st.markdown(f"""
    <div class="status-strip">
        <div class="status-card" style="border-color:{status_color}">
            <div class="status-label">현재(최근) 상태</div>
            <div class="status-value" style="color:{status_color}">{status_text}</div>
            <div class="status-sub">최근 부분군 {summary['last_group']} · 값 {summary['last_value']:.3f}</div>
        </div>
        <div class="status-card" style="border-color:{trend_color}">
            <div class="status-label">최근 추세</div>
            <div class="status-value" style="color:{trend_color}">{trend_icon} {summary['trend']}</div>
            <div class="status-sub">최근 {summary['recent_n']}개 구간 기준</div>
        </div>
        <div class="status-card" style="border-color:{flag_color}">
            <div class="status-label">최근 이상신호</div>
            <div class="status-value" style="color:{flag_color}">{summary['recent_flag_count']}건</div>
            <div class="status-sub">최근 {summary['recent_n']}개 구간 중 Nelson's Rule 발생 횟수</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =====================================================================
# 4. 샘플 데이터 (강의록 예제)
# =====================================================================

def sample_pvc_viscosity():
    data = np.array([
        [3576.27, 3630.12, 3576.27, 3630.12, 3355.69, 3363.62],
        [3504.17, 3514.52, 3747.43, 3666.15, 3709.25, 3317.28],
        [3440.11, 3494.35, 3962.93, 3514.30, 3273.57, 3336.20],
        [3638.33, 3719.84, 3617.47, 3450.17, 3378.70, 3475.50],
        [3661.94, 3485.53, 3499.43, 3605.53, 3390.29, 3519.26],
    ])
    df_wide = pd.DataFrame(data, columns=[f"pl_{i+1}" for i in range(6)])
    df_long = df_wide.melt(var_name="prod_line", value_name="viscocity")
    return {"df": df_long, "group_col": "prod_line", "value_col": "viscocity",
            "USL": 4000.0, "LSL": 3000.0,
            "desc": "PVC 점도(viscosity) 공정능력분석 예제 — 6개 생산라인 × 5개 표본, 규격 3,500±500"}


def generate_wafer_thickness(seed=42, num_sg=30, sg_size=4, target=40, tolerance=2, sg_std=2, mean_shift=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_sg):
        shift = rng.uniform(-mean_shift, mean_shift)
        for v in rng.normal(loc=target + shift, scale=sg_std, size=sg_size):
            rows.append({"Lot": i + 1, "Thickness": v})
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "value_col": "Thickness",
            "USL": target + tolerance, "LSL": target - tolerance,
            "desc": f"3D 적층형 D램 웨이퍼 두께 — {num_sg}개 로트 × {sg_size}개 표본, 목표 {target}㎛, 허용오차 ±{tolerance}㎛"}


def generate_individual_data(seed=42, num_sg=30, target=40, tolerance=2, sg_std=2, mean_shift=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_sg):
        shift = rng.uniform(-mean_shift, mean_shift)
        rows.append({"Lot": i + 1, "Thickness": rng.normal(loc=target + shift, scale=sg_std)})
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "value_col": "Thickness",
            "USL": target + tolerance, "LSL": target - tolerance,
            "desc": f"I-MR 관리도 예제 — {num_sg}개 로트 × 1개 표본, 목표 {target}, 허용오차 ±{tolerance}"}


def generate_defectives_np(seed=7, num_sg=25, sample_size=300, p=0.02):
    rng = np.random.default_rng(seed)
    rows = [{"Lot": i + 1, "sample_size": sample_size, "Defectives": rng.binomial(sample_size, p)} for i in range(num_sg)]
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "sample_col": "sample_size", "count_col": "Defectives",
            "desc": f"LED 전구 불량개수 — {num_sg}개 로트, 표본크기 {sample_size} 동일 (NP 관리도)"}


def generate_defectives_p(seed=7, num_sg=25, sample_size=300, sample_variation=80, p=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_sg):
        n_i = sample_size + rng.integers(-sample_variation, sample_variation + 1)
        rows.append({"Lot": i + 1, "sample_size": n_i, "Defectives": rng.binomial(n_i, p)})
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "sample_col": "sample_size", "count_col": "Defectives",
            "desc": f"LED 전구 불량률 — {num_sg}개 로트, 표본크기 변동 있음 (P 관리도)"}


def generate_defects_c(seed=11, num_sg=20, sample_size=500, lam=50):
    rng = np.random.default_rng(seed)
    rows = [{"Lot": i + 1, "sample_size": sample_size, "Defects": rng.poisson(lam)} for i in range(num_sg)]
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "sample_col": "sample_size", "count_col": "Defects",
            "desc": f"충전기 결함수 — {num_sg}개 로트, 표본크기 {sample_size} 동일 (C 관리도)"}


def generate_defects_u(seed=11, num_sg=20, sample_size=500, sample_variation=100, lam=50):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_sg):
        n_i = sample_size + rng.integers(-sample_variation, sample_variation + 1)
        rate = lam / sample_size
        rows.append({"Lot": i + 1, "sample_size": n_i, "Defects": rng.poisson(rate * n_i)})
    return {"df": pd.DataFrame(rows), "group_col": "Lot", "sample_col": "sample_size", "count_col": "Defects",
            "desc": f"충전기 결함수 — {num_sg}개 로트, 표본크기 변동 있음 (U 관리도)"}


# =====================================================================
# 5. Plotly 차트 (다크 테마)
# =====================================================================
PANEL_BG, GRID, TEXT_DIM = "#161D26", "#2A3441", "#8B98A9"
GOOD, WARN, BAD, INFO, PINK = "#4ADE80", "#FBBF24", "#EF4444", "#38BDF8", "#F472B6"
FONT_BODY, FONT_DISPLAY, FONT_MONO = "Inter, sans-serif", "Space Grotesk, sans-serif", "JetBrains Mono, monospace"


def _base_layout(fig, height=380, title=None):
    fig.update_layout(
        paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
        font=dict(family=FONT_BODY, color=TEXT_DIM, size=12), height=height,
        margin=dict(l=50, r=30, t=50 if title else 20, b=40),
        title=dict(text=title, font=dict(family=FONT_DISPLAY, color="#E8EDF2", size=15)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor=PANEL_BG, font_family=FONT_MONO, font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(size=11))
    return fig


def chart_histogram(values, x_bar, sigma_within, USL, LSL, bins=20):
    fig = go.Figure()
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_width = edges[1] - edges[0]
    fig.add_trace(go.Bar(x=centers, y=counts, width=bin_width * 0.92, name="빈도", marker_color=INFO, opacity=0.75))
    x_curve = np.linspace(min(values.min(), LSL), max(values.max(), USL), 200)
    y_curve = norm.pdf(x_curve, x_bar, sigma_within) * len(values) * bin_width
    fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode="lines", name="정규분포 적합", line=dict(color=PINK, width=2.5)))
    for spec_val, label in [(LSL, "LSL"), (USL, "USL")]:
        fig.add_vline(x=spec_val, line_dash="dash", line_color=BAD, annotation_text=label, annotation_font_color=BAD)
    fig.update_xaxes(title_text="측정값")
    fig.update_yaxes(title_text="빈도")
    return _base_layout(fig, title="분포 히스토그램 & 정규분포 적합")


def chart_qq(x, y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="표본", marker=dict(color=INFO, size=7)))
    lo, hi = min(min(x), min(y)) - 0.3, max(max(x), max(y)) + 0.3
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="기준선", line=dict(color=BAD, width=2)))
    fig.update_xaxes(title_text="이론 분위수")
    fig.update_yaxes(title_text="표본 분위수")
    return _base_layout(fig, title="Q-Q Plot (정규성 확인)")


def chart_subgroup_spread(mean_sg, sigma_sg, USL, LSL):
    groups = [str(g) for g in mean_sg.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=groups, y=mean_sg.values, mode="markers+lines", name="부분군 평균",
        marker=dict(color=PINK, size=9),
        error_y=dict(type="data", array=sigma_sg.values, visible=True, color=TEXT_DIM, thickness=1.3),
        line=dict(color=PINK, width=1.5, dash="dot"),
    ))
    fig.add_hline(y=LSL, line_dash="dash", line_color=BAD, annotation_text="LSL", annotation_font_color=BAD)
    fig.add_hline(y=USL, line_dash="dash", line_color=BAD, annotation_text="USL", annotation_font_color=BAD)
    fig.update_xaxes(title_text="부분군")
    fig.update_yaxes(title_text="측정값")
    return _base_layout(fig, title="부분군별 평균 및 산포")


def chart_control(chart_df, title, y_label="측정값"):
    fig = go.Figure()
    groups = chart_df["group"].astype(str)
    fig.add_trace(go.Scatter(x=groups, y=chart_df["UCL"], mode="lines", name="UCL", line=dict(color=BAD, dash="dash", width=1.3)))
    fig.add_trace(go.Scatter(x=groups, y=chart_df["CL"], mode="lines", name="CL", line=dict(color=GOOD, dash="dashdot", width=1.3)))
    fig.add_trace(go.Scatter(x=groups, y=chart_df["LCL"], mode="lines", name="LCL", line=dict(color=BAD, dash="dash", width=1.3)))

    is_ooc = ((chart_df["value"] > chart_df["UCL"]) | (chart_df["value"] < chart_df["LCL"])).fillna(False)
    colors = [BAD if o else INFO for o in is_ooc]
    sizes = [10 if o else 7 for o in is_ooc]
    line_widths = [1.3 if o else 0 for o in is_ooc]

    fig.add_trace(go.Scatter(
        x=groups, y=chart_df["value"], mode="lines+markers", name=y_label,
        line=dict(color=INFO, width=1.5),
        marker=dict(color=colors, size=sizes, line=dict(color="white", width=line_widths)),
        connectgaps=False,
    ))

    last_idx = chart_df.index[-1]
    for col, color in [("UCL", BAD), ("CL", GOOD), ("LCL", BAD)]:
        val = chart_df.loc[last_idx, col]
        if pd.notna(val):
            fig.add_annotation(x=groups.iloc[-1], y=val, text=f"{col}={val:.4f}", showarrow=False,
                                xanchor="left", xshift=42, font=dict(color=color, size=11, family=FONT_MONO))
    fig.update_xaxes(title_text="부분군 / 로트")
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(margin=dict(r=80))
    return _base_layout(fig, title=title)


def chart_boxcox(original, transformed, lam):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["원본 데이터 (비대칭)", f"Box-Cox 변환 (λ={lam:.3f})"])
    fig.add_trace(go.Histogram(x=original, marker_color="#FB923C", opacity=0.8, nbinsx=20, name="원본"), row=1, col=1)
    fig.add_trace(go.Histogram(x=transformed, marker_color=INFO, opacity=0.8, nbinsx=20, name="변환"), row=1, col=2)
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=340, title="Box-Cox 변환 전후 비교")


# =====================================================================
# 6. 커스텀 CSS
# =====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #0F1419; }
section[data-testid="stSidebar"] { background-color: #161D26; border-right: 1px solid #2A3441; }

/* 사이드바 내부 텍스트 가독성 강화 */
section[data-testid="stSidebar"] * { color: #E8EDF2; }
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p { color: #E8EDF2 !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #9AA7B5 !important; }
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: #E8EDF2 !important; font-weight: 500; }
section[data-testid="stSidebar"] hr { border-color: #2A3441; }

/* 셀렉트박스, 텍스트입력, 숫자입력, 텍스트영역 */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #1B2330 !important; color: #E8EDF2 !important; border-color: #2A3441 !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span { color: #E8EDF2 !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background-color: #1B2330 !important; color: #E8EDF2 !important; border-color: #2A3441 !important;
}
section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] button { color: #E8EDF2 !important; }

/* 라디오 버튼 라벨 */
section[data-testid="stSidebar"] div[role="radiogroup"] label p { color: #E8EDF2 !important; }

/* 드롭다운 펼쳐진 메뉴 (BaseWeb popover, body에 렌더링됨) */
ul[role="listbox"] { background-color: #1B2330 !important; }
ul[role="listbox"] li { color: #E8EDF2 !important; }
ul[role="listbox"] li:hover { background-color: #2A3441 !important; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
.brand-box { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.brand-mark { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 12px;
    letter-spacing: 0.05em; background: rgba(74,222,128,0.12); color: #4ADE80;
    border: 1px solid rgba(74,222,128,0.35); padding: 6px 9px; border-radius: 6px; }
.brand-title { font-family: 'Space Grotesk', sans-serif; font-size: 14px; font-weight: 600; color: #E8EDF2; }
.brand-sub { font-size: 10.5px; color: #5C6878; font-family: 'JetBrains Mono', monospace; }
.signal-card { display: flex; align-items: center; gap: 16px; background: #161D26;
    border: 1px solid #2A3441; border-left-width: 5px; border-radius: 12px;
    padding: 16px 22px; margin-bottom: 18px; }
.signal-dot { width: 16px; height: 16px; border-radius: 50%; flex-shrink: 0; }
.signal-grade { font-family: 'Space Grotesk', sans-serif; font-size: 15.5px; font-weight: 600; margin-bottom: 2px; color: #E8EDF2;}
.signal-action { font-size: 12.5px; color: #8B98A9; }
.signal-sigma { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #5C6878; margin-left: auto; flex-shrink: 0; }
.metric-card { background: #161D26; border: 1px solid #2A3441; border-radius: 10px; padding: 14px 16px; text-align: center; }
.metric-label { font-family: 'Space Grotesk', sans-serif; font-size: 12.5px; color: #8B98A9; font-weight: 600; }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 700; margin-top: 4px; }
.metric-sub { font-size: 10.5px; color: #5C6878; margin-top: 2px; }
.notice-good { background: rgba(74,222,128,0.10); border: 1px solid rgba(74,222,128,0.3); color: #4ADE80;
    padding: 10px 14px; border-radius: 8px; font-size: 12.8px; }
.notice-warn { background: rgba(251,191,36,0.10); border: 1px solid rgba(251,191,36,0.3); color: #FBBF24;
    padding: 10px 14px; border-radius: 8px; font-size: 12.8px; }
.notice-bad { background: rgba(239,68,68,0.10); border: 1px solid rgba(239,68,68,0.3); color: #EF4444;
    padding: 10px 14px; border-radius: 8px; font-size: 12.8px; }
div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; }

/* 현재 공정 상태 요약 패널 */
.status-strip { display: flex; gap: 14px; margin-bottom: 6px; flex-wrap: wrap; }
.status-card { flex: 1; min-width: 160px; background: #161D26; border: 1px solid #2A3441;
    border-radius: 10px; padding: 14px 16px; }
.status-label { font-family: 'Space Grotesk', sans-serif; font-size: 11.5px; color: #8B98A9;
    font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; }
.status-value { font-family: 'JetBrains Mono', monospace; font-size: 20px; font-weight: 700; margin-top: 4px; }
.status-sub { font-size: 11px; color: #5C6878; margin-top: 2px; }

</style>
""", unsafe_allow_html=True)


# =====================================================================
# 7. 사이드바 — 데이터 입력
# =====================================================================
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-mark">SPC</div>
        <div>
            <div class="brand-title">공정능력 · 관리도 분석</div>
            <div class="brand-sub">Process Capability &amp; SPC Dashboard</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 1 · 관리도 종류")
    chart_family = st.radio("데이터 유형", ["계량형 (연속값)", "계수형 (불량/결점)"], label_visibility="collapsed")
    is_count = chart_family.startswith("계수형")

    st.markdown("##### 2 · 데이터 입력")
    data_mode = st.radio("입력 방식", ["샘플 데이터", "파일 업로드", "직접 입력"], label_visibility="collapsed")

    df = None
    group_col = value_col = sample_col = count_col = None
    USL = LSL = None

    if data_mode == "샘플 데이터":
        if not is_count:
            sample_choice = st.selectbox("예제 선택", ["PVC 점도 (6라인×5)", "웨이퍼 두께 Xbar-R (30로트×4)", "웨이퍼 두께 I-MR (30로트×1)"])
            if sample_choice.startswith("PVC"):
                s = sample_pvc_viscosity()
            elif "Xbar-R" in sample_choice:
                s = generate_wafer_thickness()
            else:
                s = generate_individual_data()
            df, group_col, value_col = s["df"], s["group_col"], s["value_col"]
            USL, LSL = s["USL"], s["LSL"]
            st.caption(s["desc"])
        else:
            sample_choice = st.selectbox("예제 선택", ["LED 불량개수 (NP, 25로트)", "LED 불량률 (P, 25로트)", "충전기 결함수 (C, 20로트)", "충전기 결점률 (U, 20로트)"])
            if "NP" in sample_choice:
                s = generate_defectives_np()
            elif sample_choice.startswith("LED 불량률"):
                s = generate_defectives_p()
            elif "C," in sample_choice:
                s = generate_defects_c()
            else:
                s = generate_defects_u()
            df = s["df"]
            group_col, sample_col, count_col = s["group_col"], s["sample_col"], s["count_col"]
            st.caption(s["desc"])

    elif data_mode == "파일 업로드":
        uploaded = st.file_uploader("CSV 또는 Excel 파일", type=["csv", "xlsx", "xls"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"파일을 읽을 수 없습니다: {e}")

    else:  # 직접 입력
        st.caption("쉼표(,)로 구분된 데이터를 입력하세요. 첫 행은 헤더입니다.")
        default_text = (
            # 일부러 우측으로 치우친(비정규) 데이터를 기본값으로 넣어, 정규성 불만족 -> Box-Cox 변환 화면이
            # 사용자가 아무것도 안 해도 바로 보이도록 함
            "subgroup,value\n"
            "g1,7.398\ng1,9.962\ng1,5.617\ng1,3.033\ng1,4.690\n"
            "g2,2.741\ng2,7.847\ng2,28.225\ng2,4.517\ng2,3.973\n"
            "g3,12.059\ng3,10.558\ng3,8.211\ng3,2.914\ng3,7.176\n"
            "g4,14.810\ng4,1.927\ng4,4.676\ng4,1.104\ng4,2.035\n"
            "g5,1.171\ng5,5.841\ng5,2.080\ng5,9.692\ng5,8.643\n"
            "g6,6.129\ng6,0.596\ng6,4.312\ng6,7.039\ng6,8.276\n"
            "g7,1.600\ng7,4.583\ng7,2.777\ng7,3.291\ng7,21.347\n"
            "g8,3.295\ng8,7.153\ng8,17.893\ng8,4.122\ng8,6.608"
            if not is_count else
            "lot,sample_size,count\n1,300,8\n2,300,16\n3,300,13\n4,300,6\n5,300,8"
        )
        text = st.text_area("데이터", value=default_text, height=180)
        try:
            df = pd.read_csv(StringIO(text))
        except Exception as e:
            st.error(f"데이터 형식을 확인해주세요: {e}")

        # 위 기본 데이터(0.5~28 범위)에 맞춘 규격 기본값. 사용자가 직접 값을 입력하면 그 값이 우선됨.
        if not is_count and USL is None and LSL is None:
            USL, LSL = 30.0, 0.0

    if df is not None and data_mode != "샘플 데이터":
        st.markdown("##### 3 · 컬럼 매핑")
        cols = list(df.columns)
        if not is_count:
            group_col = st.selectbox("부분군 컬럼", cols, index=0)
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            value_col = st.selectbox("측정값 컬럼", numeric_cols if numeric_cols else cols, index=0)
        else:
            group_col = st.selectbox("로트 컬럼", cols, index=0)
            sample_col = st.selectbox("표본크기 컬럼 (없으면 동일 가정)", ["(없음)"] + cols)
            sample_col = None if sample_col == "(없음)" else sample_col
            remaining = [c for c in cols if c != group_col and c != sample_col]
            count_col = st.selectbox("불량/결점수 컬럼", remaining if remaining else cols)

    if not is_count and df is not None:
        st.markdown("##### 4 · 규격(Spec) 설정")
        spec_mode = st.radio("입력 방식", ["USL/LSL 직접 입력", "목표값(Target) ± 허용오차"], label_visibility="collapsed")

        if spec_mode == "USL/LSL 직접 입력":
            c1, c2 = st.columns(2)
            with c1:
                LSL = st.number_input("LSL (규격하한)", value=float(LSL) if LSL is not None else 0.0, format="%.4f")
            with c2:
                USL = st.number_input("USL (규격상한)", value=float(USL) if USL is not None else 1.0, format="%.4f")
            if USL is not None and LSL is not None and USL > LSL:
                st.caption(f"목표값(중심) {((USL+LSL)/2):.3f} · 허용한계 ±{((USL-LSL)/2):.3f}")
        else:
            default_target = float((USL + LSL) / 2) if (USL is not None and LSL is not None) else 0.0
            default_tol = float((USL - LSL) / 2) if (USL is not None and LSL is not None) else 1.0
            c1, c2 = st.columns(2)
            with c1:
                target = st.number_input("목표값 (Target)", value=default_target, format="%.4f")
            with c2:
                tolerance = st.number_input("허용오차 (Tolerance, ±)", value=default_tol, min_value=0.0, format="%.4f")
            LSL = target - tolerance
            USL = target + tolerance
            st.caption(f"LSL = {LSL:.4f} · USL = {USL:.4f} (목표값 ± 허용오차로 자동 산출)")

    st.markdown("---")
    st.caption("데이터를 바꾸면 모든 분석과 차트가 즉시 갱신됩니다.")


# =====================================================================
# 8. 메인 영역
# =====================================================================
st.markdown("## 공정 모니터링 대시보드")
st.caption(("공정능력분석 + 계량형 관리도" if not is_count else "계수형 관리도") + " · 데이터가 바뀌면 즉시 재분석됩니다")

if df is None or df.empty:
    st.info("왼쪽 사이드바에서 데이터를 입력하면 분석이 시작됩니다.")
    st.stop()


# ----------------- 계량형: 공정능력분석 + Xbar-R/S, I-MR -----------------
if not is_count:
    if value_col is None or group_col is None:
        st.warning("부분군 컬럼과 측정값 컬럼을 확인해주세요.")
        st.stop()

    df_work = df[[group_col, value_col]].copy()
    df_work[value_col] = pd.to_numeric(df_work[value_col], errors="coerce")
    df_work = df_work.dropna()

    if len(df_work) < 2:
        st.warning("유효한 측정값이 충분하지 않습니다.")
        st.stop()

    tab1, tab2 = st.tabs(["📐 공정능력분석", "📈 관리도 (SPC)"])

    with tab1:
        try:
            result = process_capability(df_work, group_col, value_col, USL, LSL)
        except Exception as e:
            st.error(f"공정능력 계산 중 오류: {e}")
            st.stop()

        grade = grade_capability(result["Cpk"])
        st.markdown(f"""
        <div class="signal-card" style="border-color:{grade['color']}">
            <div class="signal-dot" style="background:{grade['color']}; box-shadow:0 0 22px {grade['color']}"></div>
            <div>
                <div class="signal-grade">등급 {grade['grade']} · {grade['label']}</div>
                <div class="signal-action">{grade['action']}</div>
            </div>
            <div class="signal-sigma">{grade['sigma']} 수준</div>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(4)
        for col, label, val, sub in zip(
            cols, ["Cp", "Cpk", "Pp", "Ppk"],
            [result["Cp"], result["Cpk"], result["Pp"], result["Ppk"]],
            ["잠재능력(군내변동)", "실제능력(치우침 반영)", "장기 잠재능력", "장기 실제능력"],
        ):
            g = grade_capability(val)
            color = g["color"] if g else "#8B98A9"
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{val:.3f}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        norm_result = normality_test(df_work[value_col].values)
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(chart_histogram(df_work[value_col].values, result["x_bar"], result["sigma_within"], USL, LSL), width='stretch')
        with col_b:
            x, y = qq_plot_data(df_work[value_col].values)
            st.plotly_chart(chart_qq(x, y), width='stretch')
            badge = "notice-good" if norm_result["is_normal"] else "notice-warn"
            badge_text = "정규성 만족" if norm_result["is_normal"] else "정규성 불만족"
            st.markdown(f'<div class="{badge}">Shapiro-Wilk p = {norm_result["p"]:.4f} · {badge_text}</div>', unsafe_allow_html=True)

        if not norm_result["is_normal"]:
            with st.expander("정규성 불만족 → Box-Cox 변환 적용해보기"):
                transformed, lam, err = boxcox_transform(df_work[value_col].values)
                if err:
                    st.warning(err)
                else:
                    st.plotly_chart(chart_boxcox(df_work[value_col].values, transformed, lam), width='stretch')
                    sw_after = normality_test(transformed)
                    st.markdown(
                        f"변환 후 Shapiro-Wilk p = {sw_after['p']:.4f} "
                        f"({'정규성 만족' if sw_after['is_normal'] else '정규성 불만족'}). "
                        "공정능력지수 자체는 무차원 비율이므로 변환 전후 해석은 동일하나, "
                        "규격(USL/LSL)도 동일하게 변환해야 정확한 산출이 가능합니다."
                    )

        st.markdown("#### 기초 통계량 & 공정능력지수")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**기초 통계량**")
            st.table(pd.DataFrame({"값": [f"{result['x_bar']:.4f}", f"{result['sigma_hat']:.4f}", f"{result['sigma_within']:.4f}",
                                          f"{result['sigma_overall']:.4f}", str(result["n"]), str(result["k"])]},
                                   index=["전체 평균 (X̄)", "전체 표준편차", "σ_within (군내)", "σ_overall (전체)", "표본 수 (n)", "부분군 수 (k)"]))
        with c2:
            st.markdown("**공정능력지수**")
            st.table(pd.DataFrame({"값": [f"{result['Cp']:.4f}", f"{result['Cpk']:.4f}", f"{result['Pp']:.4f}",
                                          f"{result['Ppk']:.4f}", f"{result['Cpu']:.3f} / {result['Cpl']:.3f}"]},
                                   index=["Cp (단기 잠재)", "Cpk (단기 실제)", "Pp (장기 잠재)", "Ppk (장기 실제)", "Cpu / Cpl"]))
        with c3:
            defect = estimate_defect_rate(result["x_bar"], result["sigma_within"], USL, LSL)
            st.markdown("**추정 불량률 (정규분포 가정)**")
            st.table(pd.DataFrame({"값": [f"{defect['p_above_usl']*100:.4f}%", f"{defect['p_below_lsl']*100:.4f}%",
                                          f"{defect['total']*100:.4f}%", f"{defect['ppm']:.1f}"]},
                                   index=["USL 초과", "LSL 미만", "총 불량률", "PPM (백만개당)"]))

        st.markdown("#### 부분군별 평균 및 산포")
        st.plotly_chart(chart_subgroup_spread(result["mean_sg"], result["sigma_sg"], USL, LSL), width='stretch')

    with tab2:
        avg_size = df_work.groupby(group_col).size().mean()
        col1, col2 = st.columns([2, 1])
        with col1:
            chart_choice = st.selectbox("관리도 종류", ["자동 선택", "Xbar-R (부분군 2~10)", "Xbar-S (부분군 10 이상)", "I-MR (부분군 크기 1)"])
        with col2:
            mr_window = 3
            if "I-MR" in chart_choice or (chart_choice == "자동 선택" and avg_size <= 1):
                mr_window = st.selectbox("이동범위 윈도우", [2, 3, 4, 5], index=1)

        if chart_choice == "자동 선택":
            effective = "I-MR" if avg_size <= 1 else ("Xbar-S" if avg_size >= 10 else "Xbar-R")
        else:
            effective = chart_choice.split(" ")[0]

        st.info(f"**{effective}** 관리도 적용 중 (평균 부분군 크기: {avg_size:.1f})")

        if effective == "I-MR":
            primary, secondary, meta = i_mr_chart(df_work, group_col, value_col, window=mr_window)
            primary_label, secondary_label = "I (개별값)", "MR (이동범위)"
            cl_text = f"CL(X̄) = {meta['xbar']:.4f}"
        elif effective == "Xbar-S":
            primary, secondary, meta = xbar_s_chart(df_work, group_col, value_col)
            primary_label, secondary_label = "X̄ (부분군 평균)", "s (부분군 표준편차)"
            cl_text = f"CL(X̄) = {meta['xbar_bar']:.4f}"
        else:
            primary, secondary, meta = xbar_r_chart(df_work, group_col, value_col)
            primary_label, secondary_label = "X̄ (부분군 평균)", "R (부분군 범위)"
            cl_text = f"CL(X̄) = {meta['xbar_bar']:.4f}"

        flags = nelson_rules(primary)
        st.markdown("#### 현재 공정 상태 한눈에 보기")
        render_status_strip(current_status_summary(primary, flags))
        st.markdown("")

        st.caption(cl_text)
        st.plotly_chart(chart_control(primary, f"{effective} 관리도 — {primary_label}", primary_label), width='stretch')
        st.plotly_chart(chart_control(secondary, f"{effective} 관리도 — {secondary_label}", secondary_label), width='stretch')

        st.markdown("#### 이상 신호 탐지 (Nelson's Rule 일부 적용)")
        st.caption("Rule1: 관리한계 이탈 · Rule2: 9점 연속 한쪽 · Rule3: 6점 연속 추세")
        flagged = [(g, f) for g, f in zip(primary["group"], flags) if f]
        if not flagged:
            st.markdown('<div class="notice-good">탐지된 이상 신호가 없습니다. 공정이 관리상태(state of control)에 있는 것으로 판단됩니다.</div>', unsafe_allow_html=True)
        else:
            for g, f in flagged:
                st.markdown(f'<div class="notice-bad"><b>부분군 {g}</b> — {", ".join(f)}</div>', unsafe_allow_html=True)
                st.markdown("")

        if effective in ("Xbar-R", "Xbar-S"):
            with st.expander("이상치 제거 후 관리도 재작성"):
                ooc_groups = find_out_of_control(primary)
                if not ooc_groups:
                    st.success("이상점이 없으므로 현재 관리한계를 채택합니다.")
                else:
                    st.warning(f"이상점 발견 부분군: {ooc_groups}")
                    df_cleaned = df_work[~df_work[group_col].isin(ooc_groups)]
                    x2, r2, meta2 = (xbar_r_chart(df_cleaned, group_col, value_col) if effective == "Xbar-R"
                                      else xbar_s_chart(df_cleaned, group_col, value_col))
                    st.plotly_chart(chart_control(x2, f"이상치 제거 후 {effective} 관리도", primary_label), width='stretch')
                    st.write(f"이상 부분군 {len(ooc_groups)}개 제거: {len(df_work)}개 → {len(df_cleaned)}개 데이터")


# ----------------- 계수형: NP, P, C, U -----------------
else:
    if count_col is None or group_col is None:
        st.warning("로트 컬럼과 불량/결점수 컬럼을 확인해주세요.")
        st.stop()

    cols_needed = [group_col, count_col] + ([sample_col] if sample_col else [])
    df_work = df[cols_needed].copy()
    df_work[count_col] = pd.to_numeric(df_work[count_col], errors="coerce")
    if sample_col:
        df_work[sample_col] = pd.to_numeric(df_work[sample_col], errors="coerce")
    df_work = df_work.dropna()

    if len(df_work) < 2:
        st.warning("유효한 데이터가 충분하지 않습니다.")
        st.stop()

    has_sample = sample_col is not None
    sample_equal = has_sample and df_work[sample_col].nunique() == 1
    recommended = "C" if not has_sample else ("NP" if sample_equal else "P")

    col1, col2 = st.columns([2, 1])
    with col1:
        chart_choice = st.selectbox("관리도 종류", ["자동 선택", "NP (불량개수, 표본크기 동일)", "P (불량률, 표본크기 다름)",
                                                    "C (결점수, 표본크기 동일)", "U (단위당 결점수, 표본크기 다름)"])
    with col2:
        st.markdown(f"<br>**권장: {recommended} 관리도**", unsafe_allow_html=True)

    effective = recommended if chart_choice == "자동 선택" else chart_choice.split(" ")[0]
    sc = sample_col if sample_col else None

    if effective == "NP":
        if not sc:
            st.error("NP 관리도는 표본크기 컬럼이 필요합니다.")
            st.stop()
        chart, meta = np_chart(df_work, group_col, sc, count_col)
        unit_label, formula = "불량개수", f"np̄ = {meta['np_bar']:.3f}, p̄ = {meta['p_bar']:.4f}"
    elif effective == "P":
        if not sc:
            st.error("P 관리도는 표본크기 컬럼이 필요합니다.")
            st.stop()
        chart, meta = p_chart(df_work, group_col, sc, count_col)
        unit_label, formula = "불량률", f"p̄ = {meta['p_bar']:.4f}"
    elif effective == "U":
        if not sc:
            st.error("U 관리도는 표본크기 컬럼이 필요합니다.")
            st.stop()
        chart, meta = u_chart(df_work, group_col, sc, count_col)
        unit_label, formula = "단위당 결점수", f"ū = {meta['u_bar']:.4f}"
    else:
        chart, meta = c_chart(df_work, group_col, count_col)
        unit_label, formula = "결점수", f"c̄ = {meta['c_bar']:.3f}"

    st.info(f"**{effective}** 관리도 적용 중 · {formula}")

    flags = nelson_rules(chart)
    st.markdown("#### 현재 공정 상태 한눈에 보기")
    render_status_strip(current_status_summary(chart, flags))
    st.markdown("")

    st.plotly_chart(chart_control(chart, f"{effective} 관리도 — {unit_label}", unit_label), width='stretch')

    st.markdown("#### 이상 신호 탐지")
    st.caption("Rule1: 관리한계 이탈 · Rule2: 9점 연속 한쪽 · Rule3: 6점 연속 추세")
    flagged = [(g, f) for g, f in zip(chart["group"], flags) if f]
    if not flagged:
        st.markdown('<div class="notice-good">탐지된 이상 신호가 없습니다. 공정이 관리상태에 있는 것으로 판단됩니다.</div>', unsafe_allow_html=True)
    else:
        for g, f in flagged:
            st.markdown(f'<div class="notice-bad"><b>로트 {g}</b> — {", ".join(f)}</div>', unsafe_allow_html=True)
            st.markdown("")
