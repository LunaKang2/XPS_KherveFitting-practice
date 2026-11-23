"""Interactive F 1s fitting demo powered by the independent SciPy XPS PoC."""
from __future__ import annotations

import io
import json
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from xps_engine import (
    extract_zip_csv, fit_comparison, make_synthetic_f1s,
    parse_csv, serializable_report, zip_csv_names,
)

st.set_page_config(page_title="XPS F 1s | Fitting PoC", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .block-container { max-width: 1230px; padding-top: 2.4rem; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    .subtitle { color: #687080; font-size: 1rem; margin-top: -10px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)
st.title("XPS F 1s · Automated fitting PoC")
st.markdown('<div class="subtitle">Shirley background · product GL peaks · SciPy least-squares · 1 vs 2 components</div>', unsafe_allow_html=True)
st.info(
    "**연구용 독립 PoC** — KherveFitting/CasaXPS의 실행 결과가 아니며, "
    "피크 피팅만으로 화합물 존재를 확정하지 않습니다. F 1s에 맞춘 초기 피크 가정이 적용됩니다."
)

with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded = st.file_uploader("F 1s CSV 또는 CSV가 들어 있는 ZIP", type=["csv", "zip"],
                                help="CSV columns: Binding Energy (eV), Intensity (CPS)")
    selected_name = None
    is_demo = uploaded is None
    if is_demo:
        st.caption("업로드 전에는 합성(synthetic) 데모 데이터를 표시합니다. 실제 측정값이 아닙니다.")
        selected_name = "SYNTHETIC_F1s_demo.csv"
        raw = make_synthetic_f1s()
    else:
        uploaded_bytes = uploaded.getvalue()
        if uploaded.name.lower().endswith(".zip"):
            try:
                available = zip_csv_names(uploaded_bytes)
                selected_name = st.selectbox("ZIP 내부 CSV 파일 선택", options=available)
                raw = extract_zip_csv(uploaded_bytes, selected_name)
            except (ValueError, KeyError, zipfile.BadZipFile, RuntimeError) as exc:
                st.error(f"ZIP 파일 처리 실패: {exc}")
                st.stop()
        else:
            selected_name = uploaded.name
            raw = uploaded_bytes

    st.divider()
    st.header("2. 피팅 설정")
    additional_offset = st.number_input(
        "추가 BE offset (eV)", min_value=-10.0, max_value=10.0,
        value=0.0, step=0.01, format="%.2f",
        help="CSV에 이미 BE 보정이 포함됐다면 0으로 두세요. 여기에 입력한 값은 추가로 적용됩니다."
    )
    peak1 = st.number_input("첫 번째 피크 초기 중심 (eV)", value=685.10, step=0.1, format="%.2f")
    peak2 = st.number_input("두 번째 피크 초기 중심 (eV)", value=688.50, step=0.1, format="%.2f")
    center_window = st.slider("피크 중심 허용폭 (±eV)", min_value=0.3, max_value=3.0, value=1.5, step=0.1)
    show_model = st.radio("그래프에 표시할 피팅 모델", ["2개 피크", "1개 피크"], horizontal=True)
    st.caption("기본 FWHM 범위 0.6~4.0 eV · GL mixing 0~1 · Shirley baseline 고정")

try:
    x, y = parse_csv(raw)
    x = x + float(additional_offset)
    bg, results = fit_comparison(x, y, float(peak1), float(peak2), float(center_window))
except (ValueError, TypeError, FloatingPointError) as exc:
    st.error(f"데이터/피팅 설정 확인: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"피팅 과정에서 오류가 발생했습니다: {type(exc).__name__}: {exc}")
    st.stop()

if is_demo:
    st.warning("현재 **합성 데모 스펙트럼**을 보고 있습니다. 실제 실험 결과가 아닙니다.")

st.caption(f"**입력:** {selected_name} · 유효 포인트 {len(x):,}개 · BE 범위 {x[0]:.2f}–{x[-1]:.2f} eV · 추가 BE offset {additional_offset:+.2f} eV")

quality_rows = []
for result in results:
    q = result["metrics"]
    quality_rows.append({
        "모델": f'{result["peak_count"]} peak',
        "R²": q["r2"],
        "RMSE (CPS)": q["rmse"],
        "AICc ↓": q["aicc"],
        "BIC ↓": q["bic"],
        "최적화 성공": result["success"],
    })
quality = pd.DataFrame(quality_rows)
st.subheader("모델 비교")
st.dataframe(quality.style.format({"R²": "{:.5f}", "RMSE (CPS)": "{:,.1f}",
                                   "AICc ↓": "{:,.1f}", "BIC ↓": "{:,.1f}"}, na_rep="—"),
             hide_index=True, use_container_width=True)

if results[0]["metrics"]["aicc"] is not None and results[1]["metrics"]["aicc"] is not None:
    difference = results[0]["metrics"]["aicc"] - results[1]["metrics"]["aicc"]
    st.caption(
        f"ΔAICc (1-peak − 2-peak) = **{difference:+.2f}**. "
        "양수면 이 조건에서 2-peak 모델의 AICc가 더 낮습니다. "
        "이는 통계적 비교이지 화학적 피크 배정을 검증한 결과가 아닙니다."
    )

selected = results[0] if show_model == "1개 피크" else results[1]
fig, (ax, axr) = plt.subplots(2, 1, figsize=(10.0, 6.9), sharex=True,
                              gridspec_kw={"height_ratios": [3.25, 1]}, constrained_layout=True)
ax.scatter(x, y, s=8, c="#20252c", alpha=0.62, label="Measured intensity")
ax.plot(x, bg, lw=1.8, ls="--", c="#7f8796", label="Iterative Shirley (fixed)")
ax.plot(x, selected["fitted"], lw=2.3, c="#187a78", label=f'{selected["peak_count"]}-peak total fit')
for p in selected["peaks"]:
    ax.plot(x, bg + p["component"], lw=1.5, alpha=.88, label=f'Peak {p["peak_index"]} + background')
ax.set_ylabel("Intensity (CPS)")
ax.legend(loc="best", fontsize=8)
ax.grid(alpha=.15)
axr.plot(x, selected["residual"], lw=1.25, c="#ab541e", label="Measured - fit")
axr.axhline(0, lw=.85, c="#333333")
axr.set_xlabel("Binding Energy (eV)")
axr.set_ylabel("Residual (CPS)")
axr.legend(loc="best", fontsize=8)
axr.grid(alpha=.15)
ax.invert_xaxis()
st.subheader(f"{selected['peak_count']}-peak 피팅 및 residual")
st.pyplot(fig, use_container_width=True)
plot_data = io.BytesIO()
fig.savefig(plot_data, dpi=180, format="png", bbox_inches="tight")
plt.close(fig)

peak_rows = []
for p in selected["peaks"]:
    peak_rows.append({
        "피크": p["peak_index"], "중심 BE (eV)": p["center_eV"],
        "FWHM (eV)": p["fwhm_eV"], "높이 (CPS)": p["height_CPS"],
        "GL Lorentz fraction": p["mix"], "Area (CPS·eV)": p["area_CPS_eV_in_window"],
    })
st.subheader("선택 모델의 피크 파라미터")
st.dataframe(pd.DataFrame(peak_rows).style.format({
    "중심 BE (eV)": "{:.3f}", "FWHM (eV)": "{:.3f}",
    "높이 (CPS)": "{:,.0f}", "GL Lorentz fraction": "{:.3f}",
    "Area (CPS·eV)": "{:,.1f}",
}), hide_index=True, use_container_width=True)

if not all(r["success"] for r in results):
    st.warning("하나 이상의 모델에서 optimizer가 수렴 성공을 보고하지 않았습니다. 결과를 검토하세요.")
if any(r["near_any_bound"] for r in results):
    st.warning("하나 이상의 파라미터가 설정된 경계값에 가까워 결과가 제약조건에 민감할 수 있습니다.")

report = serializable_report(x, y, selected_name, raw, float(additional_offset),
                             float(peak1), float(peak2), float(center_window), results)
export_rows = []
for r in results:
    for p in r["peaks"]:
        export_rows.append({
            "peak_count": r["peak_count"], "r2": r["metrics"]["r2"],
            "rmse_CPS": r["metrics"]["rmse"], "aicc": r["metrics"]["aicc"],
            "bic": r["metrics"]["bic"], "peak_index": p["peak_index"],
            "center_eV": p["center_eV"], "fwhm_eV": p["fwhm_eV"],
            "height_CPS": p["height_CPS"], "mix": p["mix"],
            "area_CPS_eV_in_window": p["area_CPS_eV_in_window"],
        })
st.subheader("결과 다운로드")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("피팅 결과 JSON", data=json.dumps(report, ensure_ascii=False, indent=2),
                       file_name="F1s_fit_results.json", mime="application/json", use_container_width=True)
with c2:
    st.download_button("피크 요약 CSV", data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig"),
                       file_name="F1s_fit_summary.csv", mime="text/csv", use_container_width=True)
with c3:
    st.download_button("그래프 PNG", data=plot_data.getvalue(),
                       file_name="F1s_model_comparison.png", mime="image/png", use_container_width=True)

with st.expander("피팅 방법 · 과학적 해석 유의사항"):
    st.markdown("""
- **입력값:** 업로드한 CSV의 BE 및 CPS. BE offset이 0이면 원래 파일에 있는 보정값만 사용합니다.
- **배경:** 양 끝 5개 신호값의 중앙값을 경계값으로 쓰는 원본 PoC의 반복형 Shirley 근사. 전 구간에서 사전 계산한 배경을 피팅하는 동안 고정합니다.
- **모델:** 높이 정규화된 product GL; Gaussian/Lorentzian 기여율은 0–1 사이에서 최적화합니다. KherveFitting의 GL과 수치적으로 같다는 보장은 없습니다.
- **피팅:** 비선형 least-squares. 중심 위치는 사용자 초기값 ±허용폭, FWHM 0.6–4.0 eV. 피크 수는 1개/2개만 비교합니다.
- **평가지표:** R², RMSE, AICc, BIC. AICc/BIC는 **고정 배경**과 잔차의 독립·동분산 가정을 전제로 한 비교이며, 배경 불확실성을 포함하지 않습니다.
- **Peak area:** 측정된 **BE 범위 안에서만 적분**한 GL 성분 면적입니다. 절대 농도나 원소 조성비를 의미하지 않습니다.
- **화학적 배정:** F 1s 하나만으로 화합물을 확정하지 마세요. C 1s, O 1s, P 2p, Li 1s 등의 피크와 측정 조건, charge correction, 외부 레퍼런스를 함께 검토해야 합니다.
- **데이터 보안:** 실험 데이터를 제3자 클라우드에 올리기 전에 연구실의 데이터 공개 정책을 확인하세요.
""")
