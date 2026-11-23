"""Automated, *non-AI* least-squares fitting for XPS regions.

Purpose: reproducible baseline for a future chemistry-aware agent. No
chemical compound assignments are made. Uncertainty in binding energy
referencing is never overridden automatically.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from xps_engine import gl, shirley

# Fixed doublet physics: P 2p(1/2) = P 2p(3/2) + ~0.86eV; area ratio 1:2.
P2P_SPLIT_EV = 0.86
RECOMMENDED_MAX_PEAKS = {'C1s':4,'O1s':3,'F1s':3,'P2p':2,'Li1s':2,'N1s':3}


def _smooth(y):
    n=len(y)
    window=min(15,n if n%2 else n-1)
    return savgol_filter(y,window,3) if window>=7 else y.copy()


def _metrics(y,fitted,k):
    residual=y-fitted
    rss=float(np.dot(residual,residual));n=len(y)
    ss=float(np.dot(y-y.mean(),y-y.mean()))
    base=n*np.log(max(rss/n,np.finfo(float).tiny))
    return {'R2':1-rss/ss if ss>1e-12 else None,'RMSE':float(np.sqrt(rss/n)),
            'AICc':float(base+2*k+2*k*(k+1)/(n-k-1)) if n>k+1 else None,
            'BIC':float(base+k*np.log(n)),'RSS':rss}


def _build_peak_centers(x, net, selected, min_dist=0.65, max_dist=None):
    """Residual-guided candidate: avoid duplicated unresolved components."""
    ys=_smooth(net)
    lo=float(x[0])+0.3; hi=float(x[-1])-0.3
    options=np.argsort(ys)[::-1]
    for k in options:
        c=float(x[k])
        if not lo<c<hi: continue
        if all(abs(c-v)>=min_dist for v in selected):return c
    return None


def _fit_generic(x,y,bg,centers,core):
    n=len(centers); dx=float(np.median(np.diff(x)))
    fwhm_min=max(0.35,2.5*dx)
    fwhm_max=min(4.5,max(2.8,float(x[-1]-x[0])*0.35))
    net=np.maximum(y-bg,0)
    peak_amp=max(float(np.max(net)),10.)
    p0=[]; lo=[]; hi=[]
    for center in centers:
        idx=int(np.argmin(np.abs(x-center)))
        height=max(float(net[idx]),peak_amp*0.06)
        width=min(1.8,fwhm_max*0.85)
        cmin=max(float(x[0])+dx,center-1.3)
        cmax=min(float(x[-1])-dx,center+1.3)
        p0.extend([center,width,height,0.3])
        lo.extend([cmin,fwhm_min,0,0.0])
        hi.extend([cmax,fwhm_max,peak_amp*3.5,0.9])
    low=np.array(lo); high=np.array(hi)
    initial=np.clip(np.array(p0,float),low+1e-6,high-1e-6)
    def components(p):
        return [gl(x,*p[i:i+4]) for i in range(0,len(p),4)]
    def predict(p):return bg+sum(components(p))
    opt=least_squares(lambda p:predict(p)-y,initial,bounds=(low,high),
                      max_nfev=4500,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    comp=components(opt.x);fitted=predict(opt.x)
    peaks=[]
    for i,c in enumerate(comp):
        e,w,h,m=map(float,opt.x[4*i:4*i+4]);peaks.append({'label':f'Peak {i+1}',
            'center_eV':e,'fwhm_eV':w,'height_CPS':h,'GL_lorentz_fraction':m,
            'area_CPS_eV':float(np.trapezoid(c,x)),'curve':c})
    margin=np.minimum(opt.x-low,high-opt.x)/np.maximum(high-low,1e-12)
    return {'n_species':n,'n_components':n,'fit':fitted,'background':bg,
            'residual':y-fitted,'peaks':peaks,'metrics':_metrics(y,fitted,4*n),
            'optimizer_success':bool(opt.success),'near_bound':bool(np.any(margin<0.001)),
            'model_kind':'independent GL','message':str(opt.message)}


def _fit_p2p(x,y,bg,centers):
    """One or two chemically *unassigned* P 2p doublets, tied GL/FWHM and 2:1 area ratio."""
    net=np.maximum(y-bg,0)
    amp=max(float(net.max()),10.)
    dx=float(np.median(np.diff(x)))
    p0=[];lo=[];hi=[]
    for center in centers:
        center=np.clip(center,float(x[0])+1.0,float(x[-1])-1.3)
        p0.extend([center,1.55,amp*0.55/len(centers),0.3])
        lo.extend([max(float(x[0])+dx,center-1.5),max(0.5,2.5*dx),0.0,0.0])
        hi.extend([min(float(x[-1])-P2P_SPLIT_EV-dx,center+1.5),3.5,amp*3.0,0.9])
    low=np.asarray(lo);high=np.asarray(hi)
    initial=np.clip(p0,low+1e-6,high-1e-6)
    def components(p):
        res=[]
        for i in range(0,len(p),4):
            center,width,height,mix=p[i:i+4]
            res.extend([gl(x,center,width,height,mix),
                        gl(x,center+P2P_SPLIT_EV,width,height/2.,mix)])
        return res
    def predict(p):return bg+sum(components(p))
    opt=least_squares(lambda p:predict(p)-y,initial,bounds=(low,high),
                      max_nfev=5000,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    comps=components(opt.x); fitted=predict(opt.x)
    peaks=[]
    for i,co in enumerate(comps):
        j=i//2;center,width,height,mix=map(float,opt.x[j*4:j*4+4]); k=i%2
        peaks.append({'label':f'Doublet {j+1} (2p'+('3/2' if k==0 else '1/2')+')',
          'center_eV':center+(P2P_SPLIT_EV if k else 0),'fwhm_eV':width,
          'height_CPS':height/(2 if k else 1),'GL_lorentz_fraction':mix,
          'area_CPS_eV':float(np.trapezoid(co,x)),'curve':co})
    margin=np.minimum(opt.x-low,high-opt.x)/np.maximum(high-low,1e-12)
    return {'n_species':len(centers),'n_components':len(peaks),'fit':fitted,
        'background':bg,'residual':y-fitted,'peaks':peaks,
        'metrics':_metrics(y,fitted,4*len(centers)),
        'optimizer_success':bool(opt.success),'near_bound':bool(np.any(margin<0.001)),
        'model_kind':f'P2p constrained doublet (split={P2P_SPLIT_EV}eV, area ratio 2:1)',
        'message':str(opt.message)}


def fit_core(x, y, core, max_species=None, energy_offset_eV=0., peak_window=None):
    """Return ranked candidates, best by BIC, with no new BE correction unless requested.
    energy_offset_eV is an *additional*, explicitly approved offset.
    """
    x=np.asarray(x,float)+float(energy_offset_eV);y=np.asarray(y,float)
    if peak_window is not None:
        low,high=sorted(peak_window); mask=(x>=low)&(x<=high)
        x,y=x[mask],y[mask]
    if len(x)<35: raise ValueError(f'{core}: 최소 35개 데이터 포인트 필요')
    order=np.argsort(x);x=x[order];y=y[order]
    nmax=int(max_species or RECOMMENDED_MAX_PEAKS.get(core,3));nmax=min(max(nmax,1),5)
    if core=='P2p':nmax=min(nmax,2)
    bg=shirley(x,y)
    net=y-bg
    center=_build_peak_centers(x,net,[],min_dist=0.5)
    if center is None:raise ValueError('유효한 피크 후보 중심을 찾지 못함')
    if core=='P2p':
        # The primary maximum may be either member; assume 2p3/2 dominant lower BE.
        center=min(float(x[-1])-1.2,center)
    centers=[center]; candidates=[]
    for n in range(1,nmax+1):
        try:
            model=_fit_p2p(x,y,bg,centers) if core=='P2p' else _fit_generic(x,y,bg,centers,core)
            candidates.append(model)
        except ValueError as exc:
            break
        residual=model['residual']
        next_center=_build_peak_centers(x,np.maximum(residual,0),centers,min_dist=0.75)
        if next_center is None:break
        if core=='P2p':
            next_center=min(float(x[-1])-1.2,next_center)
            if abs(next_center-centers[0])<1.3:break
        centers.append(float(next_center))
    if not candidates:raise ValueError('피팅 최적화에 실패')
    valid=[m for m in candidates if m['optimizer_success'] and m['metrics']['BIC'] is not None]
    best=min(valid or candidates,key=lambda item:item['metrics']['BIC'])
    max_area=max([p['area_CPS_eV'] for p in best['peaks']] or [0])
    notes=[]
    if not best['optimizer_success']:notes.append('최적화 수렴 실패')
    if best['near_bound']:notes.append('일부 파라미터가 제약 경계에 도달')
    # A smooth nonrandom residual can reveal systematic model/background errors
    # despite a deceptively high global R².
    resid=best['residual']
    if len(resid)>25 and np.std(resid)>1e-9:
        lag1=float(np.corrcoef(resid[:-1],resid[1:])[0,1])
        if np.isfinite(lag1) and lag1>0.75:
            notes.append(f'잔차 자기상관이 큼 (lag-1={lag1:.2f}): 배경/피크 모델 재검토 필요')
    if any(p['area_CPS_eV']<0.04*max_area for p in best['peaks']):notes.append('매우 작은 성분: 과피팅 가능성')
    if core=='N1s' and float(np.max(net)) < 4*np.std(np.diff(y))/np.sqrt(2):
        notes.append('N 1s 신호 대비 잡음이 커 피크 해석 주의')
    if core=='P2p':notes.append('P 2p doublet 규칙 적용. 실제 시료와 장비 조건으로 splitting 확인 필요')
    return {'core':core,'x':x,'y':y,'background':bg,'candidates':candidates,
            'best':best,'warnings':notes,'additional_BE_shift_eV':float(energy_offset_eV)}


def serialize_fit(result):
    return {'core':result['core'],'energy_range_eV':[float(result['x'][0]),float(result['x'][-1])],
      'additional_BE_shift_eV':result['additional_BE_shift_eV'],
      'warnings':result['warnings'],
      'best_model':_serialize_model(result['best']),
      'model_candidates':[_serialize_model(m) for m in result['candidates']]}


def _serialize_model(m):
    return {'model_kind':m['model_kind'],'n_species':m['n_species'],
      'n_components':m['n_components'],'metrics':m['metrics'],
      'optimizer_success':m['optimizer_success'],'near_bound':m['near_bound'],
      'peaks':[{k:v for k,v in p.items() if k!='curve'} for p in m['peaks']]}
