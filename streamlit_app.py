"""XPS multi-core fitting research prototype: XLSX -> Shirley -> product GL -> LSQ.

No inferred compound labels; no unapproved auto charge correction.
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

from xps_excel_parser import parse_input, CORE_ORDER
from xps_multicore_engine import fit_core, serialize_fit, RECOMMENDED_MAX_PEAKS

st.set_page_config(page_title='XPS Multi-Core Fitting | Research PoC',page_icon='🔬',layout='wide')
st.markdown('''<style>
.block-container {max-width: 1280px; padding-top: 1.7rem;}
[data-testid="stMetricValue"]{font-size:1.25rem;}
.small-note{font-size:.9rem;color:#6b7280;}
</style>''',unsafe_allow_html=True)
st.title('XPS Multi-Core Fitting Agent · Research PoC')
st.caption('Excel/CSV/ZIP → Core-level detection → optional, user-approved BE adjustment → Shirley → product GL → SciPy nonlinear least-squares')
st.warning('연구용 분석 프로토타입입니다. **화합물 자동 동정 AI는 아직 아닙니다.** 현재 버전은 BE 기준의 신뢰성을 자동으로 판정하지 않으며, 피크 개수와 성분 배정은 연구자가 검증해야 합니다.')

with st.sidebar:
    st.header('1 · 데이터')
    uploaded=st.file_uploader('XPS 측정 파일 업로드',type=['xlsx','csv','zip'],help='Excel은 C1s, F1s 등 Core Level별 시트를 자동 인식. XLS(구형)는 XLSX로 변환 후 업로드.')
    st.caption('현재 구현의 Excel 자동 인식은 Sheet 이름과 측정 BE/CPS 열을 사용합니다. 기존 Background/피팅 곡선은 raw intensity로 사용하지 않습니다.')
if uploaded is None:
    st.info('Core-level별 시트가 들어 있는 .xlsx 파일을 업로드하세요. 현재 LSJ-9 형식은 검증 완료되었습니다.')
    st.stop()

@st.cache_data(show_spinner=False)
def read_data(raw, name):
    return parse_input(raw,name)

try:
    payload=uploaded.getvalue()
    spectra=read_data(payload,uploaded.name)
except Exception as exc:
    st.error(f'파일 분석 실패: {type(exc).__name__}: {exc}');st.stop()

all_cores=list(spectra)
fit_cores=[name for name in all_cores if name!='Survey']
if not fit_cores:
    st.error('피팅 가능한 Core Level을 찾지 못했습니다.');st.stop()

with st.sidebar:
    st.divider()
    st.header('2 · BE 보정')
    st.success('기본값: 추가 BE correction = 0.00 eV')
    st.caption('이미 B열로 보정된 C 1s / F 1s는 B열을 선택합니다. 다른 Core는 원본 A열을 사용하며, 보정 이력은 별도 확인해야 합니다.')
    with st.expander('선택적 C 1s 기준 오프셋 제안 (검토 필수)'):
        proposed=0.
        if 'C1s' in spectra:
            s=spectra['C1s'];candidate=float(s.x[np.argmax(s.y)])
            proposed=284.8-candidate
            st.write(f'C 1s 최대 강도 BE: {candidate:.2f} eV / 명목 참조: 284.80 eV')
            st.write(f'산술적 오프셋 제안: **{proposed:+.2f} eV**')
            st.caption('최대 강도 = C–C/C–H 성분이라는 보장이 없습니다. 이미 적용된 보정을 감안하고, 화학적 기준을 검증한 경우에만 승인하세요.')
        approved=st.checkbox('위 C 1s 기준을 확인했고 전체 코어에 동일 오프셋을 추가 적용',value=False)
    global_offset=proposed if approved else 0.0
    if approved:st.warning(f'추가 글로벌 이동량 {global_offset:+.2f} eV 적용됨')
    st.divider()
    st.header('3 · 모델')
    max_override=st.selectbox('최대 피크 모델 개수',options=['원소별 기본값',1,2,3,4],index=0,
       help='P2p는 피크 개수가 아니라 doublet 그룹 개수 (최대 2).')
    st.caption('선형결합이 아닌 product GL·고정 Shirley·ordinary least-squares. BIC가 가장 낮은 모델 선택.')

@st.cache_data(show_spinner=False)
def run_fit_core(core,x_tuple,y_tuple,max_n,offset):
    return fit_core(np.asarray(x_tuple),np.asarray(y_tuple),core,
                    max_species=max_n,energy_offset_eV=offset)

results={};errors={}
with st.spinner('6개 Core Level에 대해 nonlinear least-squares fitting 중…'):
    for core in fit_cores:
        sp=spectra[core]
        nmax=RECOMMENDED_MAX_PEAKS.get(core,3) if max_override=='원소별 기본값' else int(max_override)
        if core=='P2p':nmax=min(nmax,2)
        try:
            results[core]=run_fit_core(core,tuple(sp.x),tuple(sp.y),nmax,round(global_offset,5))
        except Exception as exc:
            errors[core]=f'{type(exc).__name__}: {exc}'

metrics=[]
for core in fit_cores:
    if core not in results:
        metrics.append({'Core':core,'N':len(spectra[core].x),'모델':'ERROR','R²':None,'RMSE':None,'BIC':None,'주의':errors[core]})
        continue
    r=results[core];m=r['best'];q=m['metrics']
    metrics.append({'Core':core,'N':len(r['x']),'모델':f"{m['n_species']}개 " + ('doublet' if core=='P2p' else 'peak'),
       'R²':q['R2'],'RMSE':q['RMSE'],'BIC':q['BIC'],
       '주의':'; '.join(r['warnings']) or '—'})

st.subheader('검출된 Core Level 및 자동 피팅 요약')
st.caption(f'{uploaded.name} · SHA-256 {hashlib.sha256(payload).hexdigest()[:14]}… · Survey 포함 {len(all_cores)}개 시트/파일')
st.dataframe(pd.DataFrame(metrics),hide_index=True,use_container_width=True,
             column_config={'R²':st.column_config.NumberColumn(format='%.4f'),
                            'RMSE':st.column_config.NumberColumn(format='%.1f'),
                            'BIC':st.column_config.NumberColumn(format='%.1f')})
with st.expander('BE 데이터 열 자동 선택 확인'):
    st.dataframe(pd.DataFrame([{'Core':key,'Sheet':sp.sheet,'BE 사용 열':sp.energy_column,
        'Intensity 사용 열':sp.intensity_column,
        'Excel에서 감지한 사전 BE 이동 (eV)':sp.embedded_shift_eV,
        '추가 BE 이동 (eV)':global_offset,'처리 메모':sp.note}
        for key,sp in spectra.items()]),hide_index=True,use_container_width=True)

selected=st.selectbox('자세히 분석할 Core Level',options=fit_cores)
if selected not in results:
    st.error(errors[selected]);st.stop()
sp=spectra[selected];r=results[selected];best=r['best']
st.info(sp.note)
if r['warnings']:
    st.warning('**자동 검증 경고:** '+' · '.join(r['warnings']))

def make_figure(core,r):
    x=r['x'];y=r['y'];best=r['best'];bg=r['background']
    fig,(ax,axr)=plt.subplots(2,1,figsize=(10.2,7),sharex=True,
      gridspec_kw={'height_ratios':[3.3,1]},constrained_layout=True)
    ax.scatter(x,y,s=9,color='#263548',alpha=.68,label='Measured CPS')
    ax.plot(x,bg,'--',c='#7e8793',lw=1.7,label='Iterative Shirley (fixed)')
    ax.plot(x,best['fit'],c='#126e69',lw=2.4,label='Total GL fit')
    colors=['#ad5b30','#3a65a9','#a3479b','#777f19','#6b46a2']
    for i,p in enumerate(best['peaks']):
        ax.plot(x,bg+p['curve'],lw=1.45,alpha=.9,
                color=colors[(i//2 if core=='P2p' else i)%len(colors)],
                label=f"{p['label']} {p['center_eV']:.2f} eV")
    ax.grid(alpha=.15);ax.legend(fontsize=8,ncol=2)
    ax.set_ylabel('Intensity (CPS)')
    axr.plot(x,best['residual'],c='#a74827',lw=1.1)
    axr.axhline(0,color='#2d3748',lw=.8);axr.grid(alpha=.15)
    axr.set_ylabel('Residual');axr.set_xlabel('Binding Energy (eV)')
    ax.invert_xaxis();ax.set_title(f"{core} | Shirley + product GL | {best['model_kind']}")
    return fig

fig=make_figure(selected,r)
st.pyplot(fig,use_container_width=True)
plt.close(fig)

q=pd.DataFrame([{'모델':m['model_kind'],'구성종 수':m['n_species'],'피크 수':m['n_components'],
    'R²':m['metrics']['R2'],'RMSE (CPS)':m['metrics']['RMSE'],
    'AICc':m['metrics']['AICc'],'BIC':m['metrics']['BIC'],'수렴':m['optimizer_success'],
    '경계 도달':m['near_bound']} for m in r['candidates']])
st.markdown('#### 피크 개수별 모델 비교')
st.dataframe(q.style.format({'R²':'{:.5f}','RMSE (CPS)':'{:,.1f}','AICc':'{:,.1f}','BIC':'{:,.1f}'},na_rep='—'),
             hide_index=True,use_container_width=True)
peak_rows=[{k:v for k,v in p.items() if k!='curve'} for p in best['peaks']]
st.markdown('#### 선택 모델의 피크 파라미터')
st.dataframe(pd.DataFrame(peak_rows),hide_index=True,use_container_width=True)

st.markdown('#### 분석 결과 다운로드')

def pack_all():
    archive=io.BytesIO()
    report={'source_file':uploaded.name,'sha256':hashlib.sha256(payload).hexdigest(),
      'energy_policy':'Use user-adjusted B column when consistent shifted series; otherwise A. NO new correction by default.',
      'user_approved_global_offset_eV':global_offset,
      'spectra':{},'errors':errors}
    all_peaks=[];all_models=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,sp in spectra.items():
            report['spectra'][name]={'sheet':sp.sheet,'n_points':len(sp.x),'energy_column':sp.energy_column,
              'intensity_column':sp.intensity_column,'embedded_BE_shift_eV':sp.embedded_shift_eV,
              'note':sp.note}
            if name not in results: continue
            rr=results[name]
            report['spectra'][name]['fit']=serialize_fit(rr)
            m=rr['best']
            for p in m['peaks']:
                all_peaks.append({'core':name,**{k:v for k,v in p.items() if k!='curve'}})
            for c in rr['candidates']:
                all_models.append({'core':name,'peak_count':c['n_components'],'n_species':c['n_species'],
                  'model_kind':c['model_kind'],**c['metrics'],
                  'optimizer_success':c['optimizer_success'],'near_bound':c['near_bound']})
            curves=pd.DataFrame({'BE_eV':rr['x'],'measured_CPS':rr['y'],
                 'shirley_CPS':rr['background'],'fit_CPS':m['fit'],'residual_CPS':m['residual']})
            z.writestr(f'{name}/fitted_curve.csv',curves.to_csv(index=False))
            fig=make_figure(name,rr); png=io.BytesIO();fig.savefig(png,format='png',dpi=170)
            plt.close(fig);z.writestr(f'{name}/fit.png',png.getvalue())
        z.writestr('fit_report.json',json.dumps(report,indent=2,ensure_ascii=False))
        z.writestr('all_peaks.csv',pd.DataFrame(all_peaks).to_csv(index=False))
        z.writestr('all_models.csv',pd.DataFrame(all_models).to_csv(index=False))
    return archive.getvalue()

st.download_button('모든 Core Level 결과 (PNG/CSV/JSON) ZIP 다운로드',data=pack_all(),
                   file_name='XPS_multicore_fit_results.zip',mime='application/zip',use_container_width=True)
with st.expander('한계 및 후속 검증'):
    st.markdown('''
- **이미 처리한 BE를 다시 보정하지 않습니다.** B열의 일정 이동은 Excel에 이미 존재하는 이동량이며, 분석 엔진의 추가 이동량과 구분합니다.
- **추론이 아니라 수치 피팅입니다.** 모델의 BIC 최솟값이 화학적 성분 수를 확정하지 않습니다.
- **고정 Shirley**: 피팅 전에 만든 background는 최적화 동안 고정됩니다. BE 범위/끝점 선택과 측정된 배경에 따라 달라질 수 있습니다.
- **GL**: 원본 연구 PoC의 product GL 근사이며, KherveFitting/CasaXPS와 정확히 같지 않을 수 있습니다.
- **P 2p**: 0.86 eV 분리, 2:1 면적비, 같은 FWHM/GL을 가정하는 doublet 모델. 장비·시료에 맞게 검증해야 합니다.
- **Uncertainty**: 데이터 반복 측정, 잔차 구조, 변수 상관관계, 조건별 교차검증 없이 자동 화합물 동정이나 정량 분석에 사용하지 마세요.
- **기밀 데이터**: Streamlit Cloud 업로드 전에 연구실 데이터 공개 정책 확인.
''')
