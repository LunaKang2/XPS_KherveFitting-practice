"""Research-only XPS fitting: fixed iterative Shirley, product GL, ordinary LSQ.

Automatic component count is statistical *screening*, never a chemical assignment.
The supplied already-adjusted BE axes are used without further offset by default.
All default fit constraints are editable trial ranges, NOT validated reference values.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from xps_engine import gl, shirley

P2P_SPLIT_EV = 0.86  # starting doublet constraint, verify against acquisition/standards
RECOMMENDED_MAX_PEAKS = {'C1s':4, 'O1s':3, 'F1s':3, 'P2p':2, 'Li1s':2, 'N1s':3}
# Exploratory fit widths in eV; spectroscopy resolution and physical chemistry must be checked.
FIT_PROFILES = {
    'C1s': (0.65, 2.50),
    'F1s': (0.60, 2.70),
    'O1s': (0.65, 2.90),
    'P2p': (0.60, 2.80),
    'Li1s': (0.50, 3.00),
    'N1s': (0.65, 2.90),
}
DEFAULT_GL_LIMITS = (0.0, 0.85)


def _metrics(y, fitted, k):
    resid = y - fitted
    rss = float(np.dot(resid, resid))
    n = len(y)
    sst = float(np.dot(y-y.mean(), y-y.mean()))
    base = n * np.log(max(rss/n, np.finfo(float).tiny))
    aicc = float(base+2*k+2*k*(k+1)/(n-k-1)) if n>k+1 else None
    lag = None
    if n > 12 and np.std(resid) > 1e-12:
        z = np.corrcoef(resid[:-1], resid[1:])
        if np.isfinite(z[0, 1]):
            lag = float(z[0, 1])
    return {
        'R2': float(1-rss/sst) if sst>1e-12 else None,
        'RMSE': float(np.sqrt(rss/n)), 'RSS': rss,
        'AICc': aicc, 'BIC': float(base+k*np.log(n)),
        'residual_lag1': lag,
    }


def _smooth(y):
    n=len(y)
    window=min(15, n if n%2 else n-1)
    return savgol_filter(y,window,3) if window>=7 else np.array(y, copy=True)


def _next_peak_center(x, guide, taken, min_dist=0.80):
    smoothed=_smooth(np.asarray(guide,float))
    lo=float(x[0])+0.25
    hi=float(x[-1])-0.25
    # Refuse to manufacture a new peak from an everywhere-negative residual.
    if float(smoothed.max()) <= 0: return None
    for i in np.argsort(smoothed)[::-1]:
        c=float(x[i])
        if lo<c<hi and smoothed[i]>0 and all(abs(c-v)>=min_dist for v in taken):
            return c
    return None


def _limits(x, core, fwhm_limits, gl_limits):
    widths=FIT_PROFILES.get(core, (0.60, 3.0)) if fwhm_limits is None else tuple(map(float,fwhm_limits))
    mix=DEFAULT_GL_LIMITS if gl_limits is None else tuple(map(float,gl_limits))
    if not (0.2 <= widths[0] < widths[1] <= 8.0):
        raise ValueError('FWHM constraint must satisfy 0.2 <= min < max <= 8.0 eV')
    if not (0 <= mix[0] < mix[1] <= 1):
        raise ValueError('GL fraction must satisfy 0 <= low < high <= 1')
    dx=float(np.median(np.diff(x)))
    if dx <= 0: raise ValueError('BE spacing must be positive')
    minw=max(widths[0], 2.0*dx)
    if minw>=widths[1]: raise ValueError('FWHM constraint incompatible with energy resolution')
    return (float(minw),float(widths[1])),mix


def _bounds_warning(p, low, high, n, p2p):
    margin=np.minimum(p-low, high-p) / np.maximum(high-low,1e-12)
    labels=['BE', 'FWHM', 'height', 'GL mix']
    details=[]
    for j,val in enumerate(margin):
        if val < 0.003:
            details.append(f"{'Doublet' if p2p else 'Peak'} {j//4+1} {labels[j%4]} near {'lower' if p[j]-low[j] <= high[j]-p[j] else 'upper'} bound")
    return details


def _build_model(x,y,bg,centers,core,fwhm_limits=None,gl_limits=None):
    p2p=core=='P2p'
    (wlo,whi),(mlo,mhi)=_limits(x,core,fwhm_limits,gl_limits)
    net=np.maximum(y-bg,0)
    amp=max(float(net.max()),1.)
    dx=float(np.median(np.diff(x)))
    lo=[];hi=[];p0=[]
    for i,center in enumerate(centers):
        c=float(center)
        if p2p:
            c=np.clip(c,float(x[0])+0.2,float(x[-1])-P2P_SPLIT_EV-0.2)
        left=max(float(x[0])+dx, c-1.35)
        right=min(float(x[-1])-dx-(P2P_SPLIT_EV if p2p else 0), c+1.35)
        if right-left<0.15:
            raise ValueError('Peak seed too close to scan boundary')
        local_index=int(np.argmin(abs(x-c)))
        height=max(float(net[local_index])*0.8, amp*0.06)
        if p2p: height *= 0.7/len(centers)
        p0.extend([c, float(np.clip(1.6,wlo+1e-4,whi-1e-4)), height, 0.30])
        lo.extend([left,wlo,0.,mlo]);hi.extend([right,whi,amp*4.,mhi])
    low=np.asarray(lo);upper=np.asarray(hi)
    original=np.clip(np.asarray(p0),low+1e-8,upper-1e-8)
    def components(params):
        peaks=[]
        for k in range(len(centers)):
            c,w,h,m=params[4*k:4*k+4]
            peaks.append(gl(x,c,w,h,m))
            if p2p:
                peaks.append(gl(x,c+P2P_SPLIT_EV,w,h/2,m))
        return peaks
    def predict(params):
        # Sum with a real zero array, not bg as a mutable accumulator.
        return bg+np.sum(np.asarray(components(params)),axis=0)
    starts=[original]
    # A second basin helps identify local solutions for overlapping C/F regions.
    if core in ('C1s','F1s') and len(centers)>1:
        variant=original.copy()
        variant[1::4] = np.clip(2.1,wlo+1e-4,whi-1e-4)
        variant[3::4] = np.clip(0.63,mlo+1e-4,mhi-1e-4)
        starts.append(variant)
    attempts=[]
    for start in starts:
        attempts.append(least_squares(
            lambda p:predict(p)-y,start,bounds=(low,upper),
            method='trf',max_nfev=5500,xtol=1e-9,ftol=1e-9,gtol=1e-9,
        ))
    success=[a for a in attempts if a.success]
    opt=min(success or attempts,key=lambda a:float(np.dot(a.fun,a.fun)))
    comp=components(opt.x);fit=predict(opt.x)
    peaks=[]
    for j,curve in enumerate(comp):
        k=j//2 if p2p else j
        center,width,height,mix=map(float,opt.x[4*k:4*k+4])
        if p2p and j%2: center+=P2P_SPLIT_EV; height/=2
        label=(f'Doublet {k+1} 2p'+('1/2' if j%2 else '3/2')) if p2p else f'Peak {j+1}'
        peaks.append({'label':label,'center_eV':center,'fwhm_eV':width,
                      'height_CPS':height,'GL_lorentz_fraction':mix,
                      'area_CPS_eV':float(np.trapezoid(curve,x)),'curve':curve})
    bound_details=_bounds_warning(opt.x,low,upper,len(centers),p2p)
    max_area=max([pk['area_CPS_eV'] for pk in peaks]+[1e-6])
    tiny=[pk['label'] for pk in peaks if pk['area_CPS_eV']<0.04*max_area]
    close=[]
    # Centers separated by substantially less than the narrower FWHM may be unresolvable.
    independent=peaks[::2] if p2p else peaks
    for a in range(len(independent)):
        for b in range(a+1,len(independent)):
            pa,pb=independent[a],independent[b]
            if abs(pa['center_eV']-pb['center_eV'])<0.5*min(pa['fwhm_eV'],pb['fwhm_eV']):
                close.append(f"{pa['label']}/{pb['label']}")
    metrics=_metrics(y,fit,4*len(centers))
    return {
        'n_species':len(centers),'n_components':len(comp),'fit':fit,
        'background':bg,'residual':y-fit,'peaks':peaks,'metrics':metrics,
        'optimizer_success':bool(opt.success),'near_bound':bool(bound_details),
        'boundary_details':bound_details,'tiny_components':tiny,'overlapping_components':close,
        'model_kind': ('P2p constrained doublet(s)' if p2p else 'independent product GL'),
        'message':str(opt.message),'starts_tried':len(starts),
    }


def _model_warnings(m,core,net,y):
    notes=[]
    if not m['optimizer_success']: notes.append('LSQ 최적화 수렴 실패')
    if m['boundary_details']: notes.append('제약 경계 도달: '+', '.join(m['boundary_details']))
    if m['tiny_components']: notes.append('작은 피크 면적: '+', '.join(m['tiny_components']))
    if m['overlapping_components']: notes.append('피크 분리도 부족 가능성: '+', '.join(m['overlapping_components']))
    lag=m['metrics']['residual_lag1']
    if lag is not None and lag>0.75:
        notes.append(f'잔차 자기상관 lag-1={lag:.2f} (배경/피크 모델 검토)')
    if core=='N1s' and float(max(net)) < 4*np.std(np.diff(y))/np.sqrt(2):
        notes.append('N1s 신호 대 잡음 비율 검토 필요')
    if core=='P2p':
        notes.append('P2p 0.86 eV splitting, 2:1 intensity/area model constraint: 확인 필수')
    return notes


def fit_core(x,y,core,max_species=None,energy_offset_eV=0.,peak_window=None,
             fwhm_limits=None,gl_limits=None,manual_centers=None):
    """Fit auto 1..N candidates, or exactly the user-reviewed manual seed set.

    ROI is applied before Shirley; energy_offset_eV is an ADDITIONAL manual offset.
    No automated chemical shift/charge correction is inferred.
    """
    x=np.asarray(x,dtype=float).copy()+float(energy_offset_eV)
    y=np.asarray(y,dtype=float).copy()
    if len(x)!=len(y) or len(x)<35 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('At least 35 finite, paired BE/CPS points are required')
    order=np.argsort(x);x=x[order];y=y[order]
    if peak_window is not None:
        left,right=sorted(map(float,peak_window))
        keep=(x>=left)&(x<=right)
        x,y=x[keep],y[keep]
    if len(x)<35: raise ValueError('ROI must include at least 35 points')
    if np.any(np.diff(x)<=0): raise ValueError('BE must be strictly increasing')
    _limits(x,core,fwhm_limits,gl_limits)
    bg=shirley(x,y)
    net=y-bg
    if manual_centers is not None:
        seeds=[float(v) for v in manual_centers]
        if not 1<=len(seeds)<=(2 if core=='P2p' else 5):
            raise ValueError('Manual centers: 1-5 peaks (P2p: 1-2 doublet centers)')
        if any(not np.isfinite(v) or v<x[0] or v>x[-1] for v in seeds):
            raise ValueError('Manual center outside selected BE window')
        if len(set(round(v,4) for v in seeds))!=len(seeds):
            raise ValueError('Repeated initial peak centers')
        models=[_build_model(x,y,bg,seeds,core,fwhm_limits,gl_limits)]
    else:
        max_n=int(max_species or RECOMMENDED_MAX_PEAKS.get(core,3))
        max_n=int(np.clip(max_n,1,2 if core=='P2p' else 5))
        c=_next_peak_center(x,net,[])
        if c is None:raise ValueError('Cannot identify a positive net-signal peak')
        if core=='P2p':
            c=float(np.clip(c,x[0]+0.25,x[-1]-P2P_SPLIT_EV-0.25))
        centers=[float(c)];models=[]
        for _ in range(max_n):
            try:
                model=_build_model(x,y,bg,centers,core,fwhm_limits,gl_limits)
                models.append(model)
            except ValueError:
                break
            next_c=_next_peak_center(x,np.maximum(model['residual'],0),centers)
            if next_c is None:break
            if core=='P2p':
                next_c=float(np.clip(next_c,x[0]+0.25,x[-1]-P2P_SPLIT_EV-0.25))
                if any(abs(next_c-p)<1.3 for p in centers):break
            centers.append(float(next_c))
    if not models:raise ValueError('No model could be fitted')
    valid=[m for m in models if m['optimizer_success'] and np.isfinite(m['metrics']['BIC'])]
    best=min(valid or models,key=lambda m:m['metrics']['BIC'])
    warnings=_model_warnings(best,core,net,y)
    return {'core':core,'x':x,'y':y,'background':bg,'candidates':models,
            'best':best,'warnings':warnings,
            'additional_BE_shift_eV':float(energy_offset_eV),
            'fit_settings': {'ROI_eV':[float(x[0]),float(x[-1])],
                            'FWHM_bounds_eV':list(_limits(x,core,fwhm_limits,gl_limits)[0]),
                            'GL_fraction_bounds':list(_limits(x,core,fwhm_limits,gl_limits)[1]),
                            'mode':'manual_centers' if manual_centers is not None else 'auto_BIC',
                            'manual_centers_eV':None if manual_centers is None else list(map(float,manual_centers))}}


def model_warnings(model,core,net,y):
    """Public wrapper for UI-selected non-BIC models."""
    return _model_warnings(model,core,net,y)


def _serialize_model(m):
    return {'model_kind':m['model_kind'],'n_species':m['n_species'],
        'n_components':m['n_components'],'metrics':m['metrics'],
        'optimizer_success':m['optimizer_success'],'near_bound':m['near_bound'],
        'boundary_details':m['boundary_details'],'tiny_components':m['tiny_components'],
        'overlapping_components':m['overlapping_components'],
        'starts_tried':m['starts_tried'],
        'peaks':[{k:v for k,v in p.items() if k!='curve'} for p in m['peaks']]}


def serialize_fit(result):
    return {'core':result['core'],
        'energy_range_eV':[float(result['x'][0]),float(result['x'][-1])],
        'additional_BE_shift_eV':result['additional_BE_shift_eV'],
        'fit_settings':result['fit_settings'],
        'warnings':result['warnings'],
        'best_model':_serialize_model(result['best']),
        'model_candidates':[_serialize_model(m) for m in result['candidates']]}
