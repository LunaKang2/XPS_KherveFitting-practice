"""Reproducible XPS fitting proof of concept (independent SciPy baseline).
This is NOT a direct call into KherveFitting, and is not an AI chemical identifier.
Usage: python xps_automation_poc.py --zip LSJ-9_XPS_core_level_CSV.zip --out xps_poc_results
Dependencies: numpy, scipy, matplotlib (stdlib otherwise).
"""
import argparse, csv, io, json, zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# GL product profile, height normalized; this is a benchmark approximation.
# Compare against KherveFitting only after matching its exact GL definition.
def gl(x, center, fwhm, height, lor_frac):
    z=(x-center)/(fwhm/2)
    gaussian=np.exp(-np.log(2)*z*z)
    lorentzian=1/(1+z*z)
    product=(gaussian**(1-lor_frac))*(lorentzian**lor_frac)
    return height*product

def shirley(x, y, iterations=100):
    order=np.argsort(x)
    xx=x[order]; yy=y[order]
    # high-BE endpoint baseline at maximum energy; low-BE at min energy
    high=float(np.median(yy[-5:])); low=float(np.median(yy[:5]))
    b=np.linspace(low, high, len(xx))
    for _ in range(iterations):
        sub=np.maximum(yy-b,0)
        dx=np.diff(xx)
        cum=np.zeros_like(sub)
        cum[1:]=np.cumsum((sub[:-1]+sub[1:])*dx/2)
        if cum[-1] < 1e-12: break
        new=low+(high-low)*cum/cum[-1]
        if np.max(abs(new-b)) < 1e-6: b=new; break
        b=new
    restored=np.empty_like(b);restored[order]=b
    return restored

def get_data(archive, name):
    with zipfile.ZipFile(archive) as z:
        rows=list(csv.DictReader(io.StringIO(z.read(name+'.csv').decode('utf-8-sig'))))
    x=np.array([float(r['Binding Energy (eV)']) for r in rows]);y=np.array([float(r['Intensity (CPS)']) for r in rows]);return x,y

def fit_one(x,y,background,initial,bounds):
    def predict(par):
        z=background.copy()
        for i in range(0,len(par),4): z += gl(x,*par[i:i+4])
        return z
    par=least_squares(lambda p: predict(p)-y, initial, bounds=bounds, max_nfev=15000, xtol=1e-11,ftol=1e-11,gtol=1e-11)
    fitted=predict(par.x); residual=y-fitted; ss=float(np.sum(residual**2)); total=float(np.sum((y-y.mean())**2));
    n=len(y);k=len(par.x)
    return par, fitted, residual, dict(r2=1-ss/total,rmse=float(np.sqrt(ss/n)), rss=ss, aicc=(n*np.log(ss/n)+2*k+2*k*(k+1)/(n-k-1)) if n>k+1 else None, bic=n*np.log(ss/n)+k*np.log(n))

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--out',default='xps_poc_results');args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    x,y=get_data(args.zip,'F1s');order=np.argsort(x);x=x[order];y=y[order]
    bg=shirley(x,y)
    # Fit comparison: one F1s component versus two, same background & GL definition.
    guesses=[([685.1,1.9,175000,.35],([683.5,.6,0,0],[686.5,4,300000,1])),
             ([685.1,1.9,175000,.35,688.5,2.2,65000,.20],([683.5,.6,0,0,687.0,.6,0,0],[686.5,4,300000,1,690.0,4,300000,1]))]
    results=[]
    fig,axs=plt.subplots(2,1,figsize=(9,7),sharex=True,gridspec_kw={'height_ratios':[3,1]})
    axs[0].scatter(x,y,s=8,c='black',label='Measured F 1s')
    axs[0].plot(x,bg,ls=':',label='Iterative Shirley (PoC)')
    for idx,(guess,bounds) in enumerate(guesses,1):
        pars,fit,res,quality=fit_one(x,y,bg,guess,bounds)
        records=[]
        for i in range(idx):
            center,width,height,mix=pars.x[4*i:4*i+4]
            area=float(np.trapezoid(gl(x,center,width,height,mix),x))
            records.append(dict(center_eV=float(center),fwhm_eV=float(width),height_CPS=float(height),mix=float(mix),area_CPS_eV=area))
        results.append(dict(peak_count=idx,metrics=quality,peaks=records,optimizer_success=bool(pars.success)))
        axs[0].plot(x,fit,label=f'{idx}-peak fit, R²={quality["r2"]:.5f}')
        axs[1].plot(x,res,label=f'{idx}-peak residual')
    axs[0].set_ylabel('Intensity (CPS)');axs[0].legend(fontsize=8)
    axs[1].axhline(0,c='black',lw=.6);axs[1].set_xlabel('Binding Energy (eV; original imported values)');axs[1].set_ylabel('Residual (CPS)');axs[1].legend(fontsize=8)
    axs[0].invert_xaxis();fig.tight_layout();fig.savefig(out/'F1s_model_comparison.png',dpi=160);plt.close(fig)
    summary={'dataset':'LSJ-9 F1s','source':'CSV extracted from user-provided original instrument workbook','energy_correction_eV':0.0,'warning':'Independent benchmark; GL equation and Shirley algorithm may differ from KherveFitting/CasaXPS. Not a chemical assignment or verified reference.','model_results':results}
    (out/'F1s_fit_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'F1s_fit_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['peak_count','r2','rmse_CPS','AICc','BIC','peak_index','center_eV','fwhm_eV','area_CPS_eV'])
        for r in results:
            for i,peak in enumerate(r['peaks'],1):w.writerow([r['peak_count'],r['metrics']['r2'],r['metrics']['rmse'],r['metrics']['aicc'],r['metrics']['bic'],i,peak['center_eV'],peak['fwhm_eV'],peak['area_CPS_eV']])
    for r in results:print(r['peak_count'],'peaks:',r['metrics'],'centers',[round(p['center_eV'],3) for p in r['peaks']])
if __name__=='__main__':main()
