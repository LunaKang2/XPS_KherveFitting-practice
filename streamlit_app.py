"""XPS multi-core research fitting v3: Excel → ROI Shirley → product GL → LSQ.

Research screening only; no verified chemical assignment or automatic charge correction.
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from xps_excel_parser import parse_input
from xps_multicore_engine import (
    fit_core, serialize_fit, model_warnings,
    FIT_PROFILES, DEFAULT_GL_LIMITS, RECOMMENDED_MAX_PEAKS,
)

st.set_page_config(page_title="AI Agent for Automated XPS Peak Fitting",page_icon='🔬',layout='wide')
st.markdown('''<style>
.block-container {max-width: 1280px; padding-top: 1.7rem;}
[data-testid="stMetricValue"] {font-size:1.3rem;}
</style>''',unsafe_allow_html=True)
st.title("An AI Agent for Automated XPS Peak Fitting in Battery Research")
st.caption('XLSX/CSV/ZIP → raw CPS → reviewed BE axis → ROI-specific iterative Shirley → product GL → nonlinear least-squares')
st.warning('**주의:** 화합물 AI 동정과 검증된 자동 charge correction은 아직 없습니다. 자동 피크 수(BIC)는 통계적 후보일 뿐이며, 피크의 화학적 의미·ROI·FWHM·GL 값은 연구자가 검토해야 합니다.')

with st.sidebar:
    st.header('1. 데이터')
    uploaded=st.file_uploader('XPS 측정 파일 업로드',type=['xlsx','csv','zip'],
            help='LSJ-9 파일은 C/F 1s의 기존 B열 이동 인식, 다른 core는 A열 사용')
    st.caption('이미 수정된 BE 축을 우선 사용합니다. 배경 및 기존 피팅 열은 실측 CPS로 쓰지 않습니다.')
if uploaded is None:
    st.info('LSJ-9 .xlsx 파일을 업로드하면 6개 core와 기존 BE 이동량을 인식합니다.')
    st.stop()

@st.cache_data(show_spinner=False)
def read_data(payload,filename):
    return parse_input(payload,filename)

try:
    payload=uploaded.getvalue()
    spectra=read_data(payload,uploaded.name)
except Exception as exc:
    st.error(f'파일 분석 실패: {type(exc).__name__}: {exc}')
    st.stop()
fit_cores=[core for core in spectra if core!='Survey']
if not fit_cores:
    st.error('피팅 가능한 Core Level이 없습니다.')
    st.stop()

with st.sidebar:
    st.divider()
    st.header('2. BE 보정 이력')
    st.success('추가 BE 이동량 기본값 = 0.00 eV')
    st.caption('이미 max-정렬된 LSJ-9 C 1s / F 1s는 B열을 사용합니다. 이것은 charge correction의 화학적 검증이 아닙니다.')
    with st.expander('선택적 C 1s 기준 오프셋 (승인 필요)'):
        proposed=0.0
        if 'C1s' in spectra:
            s=spectra['C1s']; max_be=float(s.x[np.argmax(s.y)])
            proposed=284.8-max_be
            st.write(f'C 1s 강도 최대 BE = {max_be:.3f} eV; 기준 후보 284.800 eV; ΔBE = {proposed:+.3f} eV')
        st.caption('강도 최대점이 C–C/C–H 기준 피크라고 가정하면 위험합니다. 이미 보정한 경우 다시 적용하지 마세요.')
        approved=st.checkbox('에너지 기준을 별도 검증했고 추가 이동을 승인함',value=False)
    global_offset=proposed if approved else 0.0
    st.divider()
    st.header('3. 자동 모델 검색')
    max_option=st.selectbox('최대 피크 / doublet 수',options=['원소별 기본값',1,2,3,4],index=0,
                            help='P 2p의 수는 doublet 그룹 수이며 최대 2입니다.')
    st.caption('일반: 피크 1–N개 후보 비교. 제약 내 최소 BIC 모델을 자동 추천하며 화학적 확정은 하지 않습니다.')

@st.cache_data(show_spinner=False)
def run_fit(core,x,y,max_n,offset,roi=None,fwhm=None,gl_limit=None,centers=None):
    return fit_core(np.asarray(x),np.asarray(y),core,max_species=max_n,
                    energy_offset_eV=offset,peak_window=roi,
                    fwhm_limits=fwhm,gl_limits=gl_limit,manual_centers=centers)

results={};errors={}
with st.spinner('Core Level별 model candidates 계산 중…'):
    for core in fit_cores:
        spec=spectra[core]
        count=RECOMMENDED_MAX_PEAKS.get(core,3) if max_option=='원소별 기본값' else int(max_option)
        if core=='P2p':count=min(count,2)
        try:
            results[core]=run_fit(core,tuple(spec.x),tuple(spec.y),count,round(global_offset,5))
        except (ValueError,RuntimeError,TypeError) as exc:
            errors[core]=f'{type(exc).__name__}: {exc}'

summary=[]
for core in fit_cores:
    spec=spectra[core]
    if core not in results:
        summary.append({'Core':core,'N':len(spec.x),'자동 추천':'ERROR','R²':np.nan,
                        'RMSE':np.nan,'BIC':np.nan,'경고':errors.get(core,'실패')})
        continue
    r=results[core];m=r['best'];q=m['metrics']
    summary.append({'Core':core,'N':len(spec.x),
                    '자동 추천':f"{m['n_species']} " + ('doublet' if core=='P2p' else 'peak'),
                    'R²':q['R2'],'RMSE':q['RMSE'],'BIC':q['BIC'],
                    '경고':'; '.join(r['warnings']) or '—'})
st.subheader('자동 피팅 요약')
st.caption(f'파일: {uploaded.name} · SHA-256: {hashlib.sha256(payload).hexdigest()[:15]}… · Survey 포함 {len(spectra)}개 구간')
st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True,
    column_config={'R²':st.column_config.NumberColumn(format='%.4f'),
                   'RMSE':st.column_config.NumberColumn(format='%.1f'),
                   'BIC':st.column_config.NumberColumn(format='%.1f')})
with st.expander('BE 데이터 열 및 기존 보정 이력 확인'):
    st.dataframe(pd.DataFrame([{'Core':core,'Sheet':sp.sheet,'BE 사용 열':sp.energy_column,
       '실측 CPS 열':sp.intensity_column,'파일 내 선행 BE 이동(eV)':sp.embedded_shift_eV,
       '새로 승인한 BE 이동(eV)':global_offset,'기록':sp.note}
       for core,sp in spectra.items()]),hide_index=True,use_container_width=True)

st.divider()
st.subheader('Core Level 리뷰 · ROI / 제약조건 / 모델 선택')
selected=st.selectbox('자세히 검토할 Core Level',options=fit_cores)
if selected not in results:
    st.error(errors[selected]);st.stop()
sp=spectra[selected]
base=results[selected]
with st.expander('Manual Review 설정: BE ROI, FWHM, GL 제한 및 초기 중심',expanded=False):
    roi_enabled=st.checkbox('ROI를 조정하여 Shirley와 피크를 다시 계산',value=False,key=f'roi_switch_{selected}')
    whole=(float(sp.x[0]+global_offset),float(sp.x[-1]+global_offset))
    if roi_enabled:
        roi_choice=st.slider('BE fit range (eV)',min_value=whole[0],max_value=whole[1],
           value=whole,step=0.05,key=f'roi_{selected}')
    else: roi_choice=None
    defaults=FIT_PROFILES.get(selected,(0.6,3.0))
    custom_limits=st.checkbox('FWHM 및 GL 제한값 직접 조정',value=False,key=f'limits_switch_{selected}')
    if custom_limits:
        c1,c2=st.columns(2)
        with c1:
            fwhm_choice=st.slider('FWHM 범위 (eV)',min_value=0.25,max_value=5.0,
              value=defaults,step=0.05,key=f'fwhm_{selected}')
        with c2:
            gl_choice=st.slider('GL Lorentz fraction 범위',min_value=0.0,max_value=1.0,
              value=DEFAULT_GL_LIMITS,step=0.05,key=f'gl_{selected}')
    else:
        fwhm_choice=None;gl_choice=None
    manual_mode=st.checkbox('초기 중심 BE를 지정하고 해당 개수만 다시 피팅',value=False,key=f'manual_{selected}')
    centers=None
    if manual_mode:
        best_centers=[p['center_eV'] for p in (base['best']['peaks'][::2] if selected=='P2p' else base['best']['peaks'])]
        seed=','.join(f'{v:.2f}' for v in best_centers)
        center_input=st.text_input('중심 BE 목록 (쉼표 구분, eV)',value=seed,key=f'centers_{selected}')
        try:
            centers=tuple(float(value.strip()) for value in center_input.split(',') if value.strip())
            if not centers:raise ValueError('최소 한 개의 중심 BE 필요')
        except ValueError:
            st.error('초기 피크 위치는 284.8, 286.2처럼 쉼표로 구분한 숫자를 입력하세요.')
            st.stop()
    st.caption('이 설정은 현재 선택한 Core Level에만 적용됩니다. 원본 Excel과 다른 core의 자동 피팅은 변경하지 않습니다.')

nmax=RECOMMENDED_MAX_PEAKS.get(selected,3) if max_option=='원소별 기본값' else int(max_option)
if selected=='P2p':nmax=min(nmax,2)
try:
    # Cache avoids repeated expensive fitting on unrelated widget changes.
    reviewed=run_fit(selected,tuple(sp.x),tuple(sp.y),nmax,round(global_offset,5),
          roi_choice,fwhm_choice,gl_choice,centers)
except (ValueError,RuntimeError,TypeError) as exc:
    st.error(f'선택한 설정으로 피팅 실패: {exc}');st.stop()

st.info(sp.note)
st.caption(f"ROI: {reviewed['fit_settings']['ROI_eV'][0]:.2f}–{reviewed['fit_settings']['ROI_eV'][1]:.2f} eV | "
           f"FWHM: {reviewed['fit_settings']['FWHM_bounds_eV']} eV | "
           f"GL: {reviewed['fit_settings']['GL_fraction_bounds']} | 추가 BE: {global_offset:+.2f} eV")

models=reviewed['candidates']
def choice_label(model):
    unit='doublet' if selected=='P2p' else 'peak'
    return f"{model['n_species']} {unit} · BIC {model['metrics']['BIC']:.1f}"
labels=['Auto: minimum BIC']+[choice_label(model) for model in models]
selection=st.selectbox('비교·표시할 모델 선택',options=labels,index=0)
chosen=reviewed['best'] if selection==labels[0] else models[labels.index(selection)-1]
selected_warnings=model_warnings(chosen,selected,reviewed['y']-reviewed['background'],reviewed['y'])
if selected_warnings:
    st.warning('**선택한 모델의 검토 항목:** '+' · '.join(selected_warnings))
else:
    st.success('현재 자동 진단 규칙에 해당하는 경고 없음. 화학적 타당성 검증을 뜻하지는 않습니다.')

model_table=[]
for m in models:
    q=m['metrics']
    model_table.append({'모델':m['model_kind'],'개수':m['n_species'],'총 peak':m['n_components'],
        'R²':q['R2'],'RMSE(CPS)':q['RMSE'],'AICc':q['AICc'],'BIC':q['BIC'],
        'ΔBIC':q['BIC']-min(item['metrics']['BIC'] for item in models),
        '잔차 lag-1':q['residual_lag1'],
        '수렴':m['optimizer_success'],'제약경계':m['near_bound'],
        '경고 개수':len(model_warnings(m,selected,reviewed['y']-reviewed['background'],reviewed['y']))})
st.markdown('#### 피크 개수별 모델 비교 (선택 가능)')
st.dataframe(pd.DataFrame(model_table).style.format({
    'R²':'{:.5f}','RMSE(CPS)':'{:,.1f}','AICc':'{:,.1f}',
    'BIC':'{:,.1f}','ΔBIC':'{:.1f}','잔차 lag-1':'{:.2f}'},na_rep='—'),
    hide_index=True,use_container_width=True)
st.caption('ΔBIC는 모델 간 통계적 차이이며 독립·동분산 잔차와 고정 Shirley를 가정합니다. lag-1이 크면 이 가정이 깨질 수 있습니다.')


def make_figure(core, result, model):
    x=result['x'];y=result['y'];bg=result['background']
    fig,(ax,axr)=plt.subplots(2,1,figsize=(10,6.8),sharex=True,
        gridspec_kw={'height_ratios':[3.2,1]},constrained_layout=True)
    ax.scatter(x,y,s=9,c='#202d3c',alpha=.65,label='Measured CPS')
    ax.plot(x,bg,'--',c='#7e8793',lw=1.6,label='Fixed Shirley (ROI)')
    ax.plot(x,model['fit'],c='#127568',lw=2.25,label='Total product GL fit')
    palette=['#b35b2e','#3c6baa','#9b4aa4','#777f19','#b98e20','#7a5195']
    for i,p in enumerate(model['peaks']):
        color=palette[(i//2 if core=='P2p' else i)%len(palette)]
        ax.plot(x,bg+p['curve'],lw=1.4,color=color,alpha=.91,
                label=f"{p['label']} {p['center_eV']:.2f} eV")
    ax.set_ylabel('Intensity (CPS)');ax.grid(alpha=.15)
    ax.legend(loc='best',fontsize=8,ncol=2);ax.set_title(f'{core} | {model["n_species"]} species | R²={model["metrics"]["R2"]:.4f}')
    axr.plot(x,model['residual'],lw=1.2,c='#b24c2e')
    axr.axhline(0,lw=.75,c='#333');axr.grid(alpha=.16)
    axr.set_xlabel('Binding Energy (eV)');axr.set_ylabel('Residual (CPS)')
    ax.invert_xaxis()
    return fig

fig=make_figure(selected,reviewed,chosen)
st.pyplot(fig,use_container_width=True)
plot=io.BytesIO();fig.savefig(plot,format='png',dpi=180,bbox_inches='tight');plt.close(fig)

st.markdown('#### 선택 모델 피크 파라미터')
selected_peaks=[{k:v for k,v in pk.items() if k!='curve'} for pk in chosen['peaks']]
st.dataframe(pd.DataFrame(selected_peaks).style.format({
        'center_eV':'{:.3f}','fwhm_eV':'{:.3f}',
        'height_CPS':'{:,.0f}','GL_lorentz_fraction':'{:.3f}',
        'area_CPS_eV':'{:,.1f}'},na_rep='—'),hide_index=True,use_container_width=True)
st.caption('GL = 0은 Gaussian 경계, GL = 1은 Lorentzian 경계에 대응합니다. 경계에 걸린 파라미터는 결과표에서 별도로 경고합니다.')

st.markdown('#### 연구자 검토 기록')
review_note=st.text_area('선택 이유 또는 추가 검증 항목 (선택)',
    placeholder='예: C1s 고BE 피크가 carbonate에 해당하는지 별도 reference 및 O1s와 교차 검증 필요',
    height=72)


def pack_report():
    buf=io.BytesIO()
    metadata={'source_file':uploaded.name,'sha256':hashlib.sha256(payload).hexdigest(),
       'analysis_label':'research preliminary / no validated chemical assignment',
       'energy_policy':'Previously shifted B when detected; no new offset by default',
       'user_approved_additional_offset_eV':float(global_offset),
       'reviewed_core':selected,'selected_model_species':chosen['n_species'],
       'selected_model_source':'auto_BIC' if selection==labels[0] else 'human-selected candidate',
       'researcher_review_note':review_note,'spectra':{},'errors':errors}
    all_peaks=[];all_models=[]
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for core,sp in spectra.items():
            metadata['spectra'][core]={'sheet':sp.sheet,'BE_column':sp.energy_column,
                'CPS_column':sp.intensity_column,'already_applied_shift_eV':sp.embedded_shift_eV,
                'notes':sp.note}
            if core not in results:continue
            res=reviewed if core==selected else results[core]
            selected_model=chosen if core==selected else res['best']
            metadata['spectra'][core]['fit']=serialize_fit(res)
            metadata['spectra'][core]['exported_model_species']=selected_model['n_species']
            metadata['spectra'][core]['exported_model'] = {
                'model_kind': selected_model['model_kind'],
                'n_species': selected_model['n_species'],
                'n_components': selected_model['n_components'],
                'metrics': selected_model['metrics'],
                'optimizer_success': selected_model['optimizer_success'],
                'boundary_details': selected_model['boundary_details'],
                'warnings': model_warnings(selected_model, core, res['y']-res['background'], res['y']),
                'peaks': [{k:v for k,v in pk.items() if k != 'curve'} for pk in selected_model['peaks']],
            }
            for m in res['candidates']:
                all_models.append({'core':core,'count':m['n_species'],**m['metrics'],
                    'converged':m['optimizer_success'],'at_parameter_bound':m['near_bound'],
                    'boundary_details':'; '.join(m['boundary_details'])})
            for pk in selected_model['peaks']:
                all_peaks.append({'core':core,**{k:v for k,v in pk.items() if k!='curve'}})
            curve=pd.DataFrame({'BE_eV':res['x'],'measured_CPS':res['y'],
                'Shirley_CPS':res['background'],'fit_CPS':selected_model['fit'],
                'residual_CPS':selected_model['residual']})
            archive.writestr(f'{core}/curve.csv',curve.to_csv(index=False))
            f=make_figure(core,res,selected_model)
            image=io.BytesIO();f.savefig(image,format='png',dpi=165,bbox_inches='tight');plt.close(f)
            archive.writestr(f'{core}/fit.png',image.getvalue())
        archive.writestr('report.json',json.dumps(metadata,ensure_ascii=False,indent=2))
        archive.writestr('all_peaks.csv',pd.DataFrame(all_peaks).to_csv(index=False))
        archive.writestr('model_candidates.csv',pd.DataFrame(all_models).to_csv(index=False))
    return buf.getvalue()

st.markdown('#### 결과 내보내기')
col_a,col_b,col_c=st.columns(3)
with col_a:
    st.download_button('모든 Core ZIP',data=pack_report(),
        file_name='XPS_v3_multicore_review.zip',mime='application/zip',use_container_width=True)
with col_b:
    st.download_button('선택 Core 피크 CSV',
        data=pd.DataFrame(selected_peaks).to_csv(index=False).encode('utf-8-sig'),
        file_name=f'{selected}_reviewed_peaks.csv',mime='text/csv',use_container_width=True)
with col_c:
    st.download_button('선택 Core 그래프 PNG',data=plot.getvalue(),
        file_name=f'{selected}_selected_model.png',mime='image/png',use_container_width=True)

with st.expander('해석·자동화 한계 및 설정 주의사항'):
    st.markdown('''
- **원소 해석 / AI**: 피크 수의 자동 결정은 화학적 결합상태 동정이 아닙니다. C 1s/O 1s/F 1s/P 2p/Li 1s 등을 함께 확인하세요.
- **기존 BE 보정**: LSJ-9의 B열 이동을 재보정하지 않습니다. 다른 데이터는 기준 성분·기존 보정 여부를 반드시 확인하세요.
- **Shirley**: 지정한 ROI 내에서 한 번 계산한 iterative Shirley를 고정하고 GL 파라미터만 least-squares로 최적화합니다. 반복적인 joint fitting은 구현하지 않았습니다.
- **FWHM/GL 범위**: 기본값은 잠정적 탐색 범위입니다. 시료의 기기 분해능, charging, core-level 물리 조건을 기준으로 검증해야 합니다.
- **P 2p**: 2p3/2:2p1/2 강도 2:1, splitting 0.86 eV, 같은 폭과 GL을 가정합니다. 계측·참조 데이터로 확인 필요.
- **모델 비교**: BIC/AICc는 동일 데이터·동일 ROI에서만 비교하세요. 서로 다른 ROI 간 BIC를 직접 비교하면 안 됩니다.
- **Residual 및 불확실성**: 구조적 residual, 파라미터 상관, 작은 성분, 중첩 성분의 검증이 필요합니다. R²만으로 타당성을 판단하지 마세요.
- **데이터 보안**: 실험 스펙트럼을 공용 클라우드에 올리기 전 연구실 정책을 확인하세요.
''')
