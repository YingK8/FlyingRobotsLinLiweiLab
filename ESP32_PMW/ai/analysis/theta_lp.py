"""主流程：老师的判据 J_Y = −d ln tan(θ/2)/dt，只算 J_Y 的几种变体。
θ = 轴 n(t) 与稳态轴 ref（台阶后 5–10 s 平均轴）的夹角；拟合窗 = 低通曲线的 t10…t90。
变体（同一条曲线、同一个窗）：
  J_Y      θ 标量 1.5 Hz 零相位低通，按残差 3·MAD 迭代剔点（缺省；ROBUST_K 环境变量可改）
  J_Y_raw  同上，不剔点
  J_Y_k15  同上，1.5·MAD 剔点
  J_Y_ts   同上，Theil–Sen 稳健斜率
  J_Y_vec  先对轴向量 (nx,ny,nz) 做 8 Hz 低通去噪、归一化，再算 θ，再 1.5 Hz 平滑；自己定 t10/t90；3·MAD 剔点。
           噪声在取角度之前已滤掉，小台阶时的"整流地板"偏差小得多。
另外保留指数拟合 J_exp：在 t10 … tc+8 s 拟 θ = A·e^(−(t−t10)/τ) + C（θ 为标量低通曲线），J_exp = 1/τ；画廊里的红线。
整条 take 的剔除：拟合 R²（不剔点的）低于 全部 take 的中位 − 3·MAD 时，该 take 不进统计。
旧版（含 1/T、指数拟合 J_exp）存为 theta_lp_old.py。
用法：uv run python ai/analysis/theta_lp.py   （INCLUDE_OLD=1 把 09-08 带进来，灰色）
输出：results/rim/theta_lp_rate.png, theta_lp_gallery.png, theta_lp_Yfit.png, theta_lp.json"""
import sys, json, csv, os; sys.path.insert(0,'.')
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import theilslopes
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['PingFang SC','Arial Unicode MS']; plt.rcParams['axes.unicode_minus']=False
from ai.analysis.takes import RIM, TAKES, AXIS_FILE, HAS_OLD
FS_U=200.0; FC=float(os.environ.get('FC','1.5')); VEC_FC=8.0   # FC：θ 低通截止。1.5 Hz 是为半环/整环定的；no ring 快模态 τ≈0.25 s，用 FC=4 核对过偏差
ROBUST_K=float(os.environ.get('ROBUST_K','3.0'))
T_HI=float(os.environ.get('T_HI','0.90'))
R2EXP_MIN=float(os.environ.get('R2EXP_MIN','0'))   # 段级质量闸（>0 启用）：R2exp（对全体指标）低于它整段剔除。
# 用途：吊挂摆动污染（顶部旋转关节卡滞→万向节摆叠在衰减上）让曲线不再是干净指数——R2exp 是最灵敏的症状，
# 而"相对本清单中位−3·MAD"的规则在坏段占比高时会失效（130918 六段把中位拖低后全体过关）。
# 2026-09-16 标定（half ring, WIN=multi, Δθ≥5°）：好段 ≥0.906，摆动污染段 ≤0.887，共振段 ≤0.77 → 用 0.90。
# 半环 campaign 跑 theta_lp 时传 R2EXP_MIN=0.90；no ring 低频（20–30 Hz）有合法的 0.87，别开或另标。   # J_Y 拟合窗的下端：降幅完成这个比例的时刻（t90/t95…）。只影响 J_Y 系，不影响 J_exp
STEP_WIN=(9.0,32.0)
# 时间窗方案。ours：我们的 schedule（断电后保持 10 s）。kevin：Kevin 的 tilt_schedule.py（断电后只等 DROP_MS=5 s 就降频）。
WIN=os.environ.get('WIN','ours')
if WIN=='multi':      # 多次重复录像切出来的段：段本身只有 台阶前 6 s + 台阶后 post
    PRE=(3.5,1.0); POST=(6.0,9.0); REF=(5.0,9.0); FIT_END=9.0; TU_END=9.5; STEP_WIN=(4.0,8.0)
elif WIN=='kevin':
    PRE=(2.5,0.5); POST=(3.0,4.5); REF=(3.0,4.5); FIT_END=4.8; TU_END=5.0; STEP_WIN=(5.0,None)   # None = 录像末尾前 6 s
else:
    PRE=(3.5,1.0); POST=(6.0,10.0); REF=(5.0,10.0); FIT_END=8.0; TU_END=10.0
OUT_TAG=os.environ.get('OUT_TAG','')   # 输出文件名前缀，如 halfring_
VARIANTS=[('J_Y','3·MAD 剔点（缺省）','C3','o'),('J_Y_raw','不剔点','gray','s'),('J_Y_k15','1.5·MAD 剔点','C0','D'),
          ('J_Y_ts','Theil–Sen','C2','^'),('J_Y_vec','向量低通版（3·MAD）','C4','v')]
R2OF={'J_Y':'R2Y_raw','J_Y_raw':'R2Y_raw','J_Y_k15':'R2Y_raw','J_Y_ts':'R2Y_raw','J_Y_vec':'R2Y_vec_raw','J_exp':'R2exp'}

def load(take):
    t,ns=[],[]
    for r in csv.DictReader(open(RIM/take/AXIS_FILE)):
        if float(r['agree_deg'])>10: continue
        t.append(float(r['t'])); ns.append([float(r['nx']),float(r['ny']),float(r['nz'])])
    t=np.array(t); t-=t[0]; ns=np.array(ns); ns/=np.linalg.norm(ns,axis=1,keepdims=True); return t,ns

def find_step_vec(t,ns,lo=None,hi=None):
    """台阶 = 相邻两个 1 s 中值轴向量夹角最大的时刻"""
    lo=STEP_WIN[0] if lo is None else lo; hi=(STEP_WIN[1] if STEP_WIN[1] is not None else t[-1]-6.0) if hi is None else hi
    best=(np.nan,-1.0)
    for i in range(len(t)):
        if not (lo<t[i]<hi): continue
        a=ns[(t>=t[i]-1)&(t<t[i])]; b=ns[(t>=t[i])&(t<t[i]+1)]
        if len(a)<10 or len(b)<10: continue
        ma=np.median(a,0); mb=np.median(b,0); ma/=np.linalg.norm(ma); mb/=np.linalg.norm(mb)
        j=np.degrees(np.arccos(np.clip(ma@mb,-1,1)))
        if j>best[1]: best=(t[i],j)
    return best[0]

def robust_line(x,y,k=None,min_keep=0.5,iters=6):
    """直线拟合 + 迭代剔点：|残差| > k·1.4826·MAD 的点剔掉，最多剔 50%。返回 (斜率, 截距, 内点掩码, 内点R²)"""
    if k is None: k=ROBUST_K
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x); m=np.ones(n,bool)
    for _ in range(iters):
        a,b=np.polyfit(x[m],y[m],1); r=y-(a*x+b); s=1.4826*np.median(np.abs(r[m]-np.median(r[m])))
        if s<1e-9: break
        m2=np.abs(r)<=k*s
        if m2.sum()<min_keep*n:
            idx=np.argsort(np.abs(r)); m2=np.zeros(n,bool); m2[idx[:int(np.ceil(min_keep*n))]]=True
        if (m2==m).all(): break
        m=m2
    a,b=np.polyfit(x[m],y[m],1); r=y[m]-(a*x[m]+b); return float(a),float(b),m,float(1-np.var(r)/np.var(y[m]))

def r2(x,y):
    a,b=np.polyfit(x,y,1); return float(1-np.var(y-(a*x+b))/np.var(y))

Yf=lambda th: np.log(np.tan(np.radians(np.clip(th,0.05,None))/2))

def cross(tu,y,i0,level):
    idx=np.where(y[i0:]<=level)[0]
    if not len(idx): return np.nan,None
    i=i0+idx[0]; y0,y1=y[i-1],y[i]
    return (tu[i-1]+(tu[i]-tu[i-1])*(y0-level)/(y0-y1) if y0!=y1 else tu[i]), i

def window(tu,lp,tc):
    """pre/post/Δθ 与 t10/t90；返回 None 表示台阶不可用"""
    pre=float(np.median(lp[(tu>=tc-PRE[0])&(tu<=tc-PRE[1])])); post=float(np.median(lp[(tu>=tc+POST[0])&(tu<=tc+POST[1])])); D=pre-post
    if not np.isfinite(D) or D<1.5: return None
    l10,l90=pre-0.1*D,pre-T_HI*D
    t10,i10=cross(tu,lp,int(np.searchsorted(tu,tc-0.8)),l10)
    if i10 is None: return None
    t90,i90=cross(tu,lp,i10,l90)
    if i90 is None: return None
    kk=(tu>=t10)&(tu<=t90)
    if kk.sum()<20: return None
    return dict(pre=pre,post=post,D=D,l10=l10,l90=l90,t10=t10,t90=t90,kk=kk)

def analyse(take):
    t,ns=load(take); tc=find_step_vec(t,ns)
    if not np.isfinite(tc): return None
    ref=ns[(t>=tc+REF[0])&(t<=tc+REF[1])].mean(0); ref/=np.linalg.norm(ref)
    tu=np.arange(tc-4,tc+TU_END,1/FS_U); b,a=butter(2,FC/(FS_U/2))
    # 标量版：先取角度，再低通
    th=np.degrees(np.arccos(np.clip(ns@ref,-1,1)))
    lp=filtfilt(b,a,np.interp(tu,t,th))
    w=window(tu,lp,tc)
    if w is None: return None
    kk=w['kk']; x=tu[kk]; Y=Yf(lp[kk])
    s3,i3,m3,R2_3=robust_line(x,Y)                 # 缺省 k（3）
    s15,_,m15,R2_15=robust_line(x,Y,k=1.5)
    s0=np.polyfit(x,Y,1)[0]; R2_0=r2(x,Y); sts=float(theilslopes(Y,x)[0])
    # 指数拟合（θ 空间）：t10 … tc+8 s，J_exp = 1/τ
    t10=w['t10']; ke=(tu>=t10)&(tu<=tc+FIT_END); fe=lambda xx,A,tau,C: A*np.exp(-(xx-t10)/tau)+C
    try:
        pe,_=curve_fit(fe,tu[ke],lp[ke],p0=[w['D'],0.5,w['post']],bounds=([0,0.02,-5],[30,20,15]),maxfev=20000); A_e,tau_e,C_e=map(float,pe)
        R2e=float(1-np.var(lp[ke]-fe(tu[ke],*pe))/np.var(lp[ke]))
    except Exception: A_e=tau_e=C_e=R2e=np.nan
    # 向量版：先对轴向量做 8 Hz 低通（高于进动频率 1.3–3.3 Hz，只去噪声和自转抖动，不抹掉绕圈），归一化，
    # 再取角度，再和标量版一样做 1.5 Hz 平滑。噪声在取角度之前已大部分滤掉 → 整流地板小得多。
    # 注意：不能直接用 1.5 Hz 做向量低通——那会把绕圈平均掉，只剩圈心（0.5 s 就到位），量的是另一件事。
    b8,a8=butter(2,VEC_FC/(FS_U/2))
    nv=np.stack([filtfilt(b8,a8,np.interp(tu,t,ns[:,k])) for k in range(3)],1); nv/=np.linalg.norm(nv,axis=1,keepdims=True)
    lpv=filtfilt(b,a,np.degrees(np.arccos(np.clip(nv@ref,-1,1)))); wv=window(tu,lpv,tc)
    if wv is not None:
        xv=tu[wv['kk']]; Yv=Yf(lpv[wv['kk']]); sv,_,_,R2v=robust_line(xv,Yv); R2v0=r2(xv,Yv); Jv=-sv
    else: Jv=R2v=R2v0=np.nan
    return dict(take=take,tc=tc,D=w['D'],pre=w['pre'],post=w['post'],l10=w['l10'],l90=w['l90'],t10=w['t10'],t90=w['t90'],
                J_Y=-s3,R2Y=R2_3,keepY=float(m3.mean()),J_Y_raw=-s0,R2Y_raw=R2_0,J_Y_k15=-s15,R2Y_k15=R2_15,keep15=float(m15.mean()),
                J_Y_ts=-sts,J_exp=(1/tau_e if tau_e==tau_e and tau_e>0 else np.nan),tau=tau_e,R2exp=R2e,fit=(A_e,tau_e,C_e),J_Y_vec=Jv,R2Y_vec=R2v,R2Y_vec_raw=R2v0,D_vec=(wv['D'] if wv else np.nan),
                tu=tu,lp=lp,lpv=lpv,th_raw=(t,th),Yfit=(s3,i3),Ypts=(x,Y,m3))

def r2_thr(vals):
    v=np.array([x for x in vals if x==x]); med=np.median(v); mad=1.4826*np.median(np.abs(v-med))
    return float(max(med-3*mad,0.5)),float(med),float(mad)

def linearity(hz,m,pts=None):
    hz=np.asarray(hz,float); m=np.asarray(m,float); ok=np.isfinite(m); hz,m=hz[ok],m[ok]
    a,b=np.polyfit(hz,m,1); sst=np.sum((m-m.mean())**2)
    out=dict(a=float(a),b=float(b),R2_lin=float(1-np.sum((m-(a*hz+b))**2)/sst))
    k=float(hz@m/(hz@hz)); out.update(k=k,R2_0=float(1-np.sum((m-k*hz)**2)/sst))
    if pts is not None:
        f,y=np.asarray(pts[0],float),np.asarray(pts[1],float); ok=np.isfinite(y); f,y=f[ok],y[ok]
        aa,bb=np.polyfit(f,y,1); out['R2_all']=float(1-np.sum((y-(aa*f+bb))**2)/np.sum((y-y.mean())**2))
    return out

res=[]; first={}
if __name__=='__main__':
    for take,hz,tag in TAKES:
        r=analyse(take)
        if r is None: print(f'{take[-6:]} {hz:3d}Hz: 台阶不可用，跳过'); continue
        r.update(hz=hz,tag=tag); res.append(r)
        if tag=='new' and hz not in first: first[hz]=r
        print(f'{take[-6:]} {hz:3d}Hz {tag:3s}: Δθ={r["D"]:4.1f}°  J_Y={r["J_Y"]:.2f} (R²={r["R2Y"]:.3f}, 留 {r["keepY"]*100:.0f}%)  不剔 {r["J_Y_raw"]:.2f}  1.5MAD {r["J_Y_k15"]:.2f}  TS {r["J_Y_ts"]:.2f}  向量 {r["J_Y_vec"]:.2f}  |  J_exp=1/τ {r["J_exp"]:.2f} (R²={r["R2exp"]:.3f})')
    # 整条 take 的剔除
    THR={}; print('\n==== 整条 take 剔除阈值（不剔点的 R² < 中位−3·MAD）====')
    for key in ('J_Y','J_Y_vec','J_exp'):
        thr,med,mad=r2_thr([r[R2OF[key]] for r in res]); THR[key]=thr
        bad=[r for r in res if r[R2OF[key]]==r[R2OF[key]] and r[R2OF[key]]<thr]
        print(f'  {key:8s}: 中位 {med:.3f} MAD {mad:.3f} 阈值 {thr:.3f}  剔除 {len(bad)} 条 '+', '.join(f'{r["take"][-6:]}({r["hz"]}Hz)' for r in bad))
    for k in ('J_Y_raw','J_Y_k15','J_Y_ts'): THR[k]=THR['J_Y']
    if R2EXP_MIN>0:
        gated=[r for r in res if not (r['R2exp']==r['R2exp'] and r['R2exp']>=R2EXP_MIN)]
        print(f'  段级质量闸 R2exp≥{R2EXP_MIN:g}: 整段剔除 {len(gated)} 条 '+', '.join(f'{r["take"][-6:]}({r["hz"]}Hz)' for r in gated))
    ok=lambda r,key: (np.isfinite(r[key]) and r[R2OF[key]]==r[R2OF[key]] and r[R2OF[key]]>=THR[key]
                      and (R2EXP_MIN<=0 or (r['R2exp']==r['R2exp'] and r['R2exp']>=R2EXP_MIN)))
    for r in res: r['ok_Y']=bool(ok(r,'J_Y')); r['ok_vec']=bool(ok(r,'J_Y_vec')); r['ok_exp']=bool(ok(r,'J_exp'))
    json.dump({'thr':THR,'robust_k':ROBUST_K,'takes':[{k:v for k,v in r.items() if k not in ('tu','lp','lpv','th_raw','Yfit','Ypts','fit')} for r in res]},
              open(RIM/(OUT_TAG+'theta_lp.json'),'w'),indent=1,default=float)
    hzs=sorted(first)
    def series(key,tag='new'):
        g=[r for r in res if r['tag']==tag and ok(r,key)]; hh=sorted({r['hz'] for r in g})
        m=[np.mean([r[key] for r in g if r['hz']==h]) for h in hh]
        s=[np.std([r[key] for r in g if r['hz']==h],ddof=1) if sum(r['hz']==h for r in g)>1 else 0 for h in hh]
        return g,np.array(hh,float),np.array(m),np.array(s)
    # ---- 主图：左 = 缺省 J_Y；右 = 各变体对比
    fig,ax=plt.subplots(1,3,figsize=(22,5.6)); LIN={}
    a=ax[0]
    for tag,col,mk,lab in [('new','C3','o',os.environ.get('SETUP','09-09 台位'))]+([('old','gray','s','09-08 台位')] if HAS_OLD else []):
        g,hh,m,s=series('J_Y',tag)
        for r in g: a.plot(r['hz']+np.random.uniform(-1.2,1.2),r['J_Y'],mk,color=col,ms=5,alpha=0.55,mfc='none' if tag=='old' else None)
        bad=[r for r in res if r['tag']==tag and not ok(r,'J_Y')]
        for j,r in enumerate(bad): a.plot(r['hz'],r['J_Y'],'x',color='k',ms=8,mew=2,label=(f'剔除：R² < {THR["J_Y"]:.3f}' if j==0 else None))
        a.errorbar(hh,m,yerr=s,fmt=mk+'-',color=col,ms=7,capsize=4,lw=1.6,label=f'{lab} 均值±SD')
        if tag=='new' and len(hh)>=2:
            L=linearity(hh,m,pts=([r['hz'] for r in g],[r['J_Y'] for r in g])); xs=np.linspace(0,160,3)
            a.plot(xs,L['a']*xs+L['b'],'k-',lw=1.2,alpha=0.7,label=f"直线 {L['a']:.4f}·f{L['b']:+.2f}，R²={L['R2_lin']:.2f}（均值）/ {L['R2_all']:.2f}（全部点）")
            a.plot(xs,L['k']*xs,'--',color=col,lw=1,alpha=0.7,label=f"过原点 {L['k']:.4f}·f，R²={L['R2_0']:.2f}")
    a.set_title(f'J_Y = −d ln tan(θ/2)/dt，t10–t90 段，{ROBUST_K:g}·MAD 剔点',fontsize=11)
    a=ax[1]
    for j,(key,lab,col,mk) in enumerate(VARIANTS):
        g,hh,m,s=series(key)
        if len(hh)<2: continue
        L=linearity(hh,m); LIN[key]=L
        a.errorbar(hh+(j-2)*1.0,m,yerr=s,fmt=mk+'-',color=col,ms=5,capsize=3,lw=1.2,alpha=0.9,label=f"{lab}：{L['a']:.4f}·f{L['b']:+.2f}，R²={L['R2_lin']:.2f}")
    a.set_title('J_Y 的五种算法（均值±SD，横向微错开）',fontsize=11)
    a=ax[2]
    for tag,col,mk,lab in [('new','C1','o',os.environ.get('SETUP','09-09 台位'))]+([('old','gray','s','09-08 台位')] if HAS_OLD else []):
        g,hh,m,s_=series('J_exp',tag)
        for r in g: a.plot(r['hz']+np.random.uniform(-1.2,1.2),r['J_exp'],mk,color=col,ms=5,alpha=0.55,mfc='none' if tag=='old' else None)
        bad=[r for r in res if r['tag']==tag and not ok(r,'J_exp')]
        for j,r in enumerate(bad): a.plot(r['hz'],r['J_exp'],'x',color='k',ms=8,mew=2,label=(f'剔除：R² < {THR["J_exp"]:.3f}' if j==0 else None))
        a.errorbar(hh,m,yerr=s_,fmt=mk+'-',color=col,ms=7,capsize=4,lw=1.6,label=f'{lab} 均值±SD')
        if tag=='new' and len(hh):
            L=linearity(hh,m,pts=([r['hz'] for r in g],[r['J_exp'] for r in g])); LIN['J_exp']=L; xs=np.linspace(0,160,3)
            a.plot(xs,L['a']*xs+L['b'],'k-',lw=1.2,alpha=0.7,label=f"直线 {L['a']:.4f}·f{L['b']:+.2f}，R²={L['R2_lin']:.2f}（均值）/ {L['R2_all']:.2f}（全部点）")
            a.plot(xs,L['k']*xs,'--',color=col,lw=1,alpha=0.7,label=f"过原点 {L['k']:.4f}·f，R²={L['R2_0']:.2f}")
    a.set_title('指数拟合 θ = A·e^(−t/τ) + C → J_exp = 1/τ',fontsize=11)
    for a in ax: a.set_xlim(0,160); a.set_ylim(0,None); a.set_xlabel('驱动频率 f (Hz)'); a.set_ylabel('J (1/s)'); a.grid(alpha=0.25); a.legend(fontsize=8,loc='upper left')
    fig.suptitle(os.environ.get('TITLE','对齐速率 J_Y（老师判据），'+os.environ.get('SETUP','2026-09-09 台位'))+('，灰=09-08 台位' if HAS_OLD else ''),fontsize=12)
    fig.tight_layout(); fig.savefig(RIM/(OUT_TAG+'theta_lp_rate.png'),dpi=120)
    # ---- 画廊：θ(t)，标出拟合窗
    n=len(hzs); cols=4; rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(5.2*cols,3.6*rows),squeeze=False)
    for a,hz in zip(axes.flat,hzs):
        r=first[hz]; tu=r['tu']-r['t10']; t,th=r['th_raw']
        a.plot(t-r['t10'],th,lw=0.4,color='C0',alpha=0.35,label='原始 θ'); a.plot(tu,r['lp'],color='C1',lw=2,label=f'θ 低通 {FC} Hz')
        a.plot(tu,r['lpv'],color='C4',lw=1,ls='--',alpha=0.8,label='向量低通版')
        a.axvspan(0,r['t90']-r['t10'],color='gold',alpha=0.15,label='J_Y 拟合窗 t10–t90')
        A_e,tau_e,C_e=r['fit']
        if tau_e==tau_e:
            ke=(tu>=0)&(tu<=r['tc']+FIT_END-r['t10']); a.plot(tu[ke],A_e*np.exp(-tu[ke]/tau_e)+C_e,'r-',lw=2,label=f'指数拟合 τ={tau_e*1000:.0f} ms')
        for lv in (r['l10'],r['l90']): a.axhline(lv,ls='--',color='g',lw=0.9)
        a.set_xlim(-4,TU_END); a.set_ylim(-0.3,max(14,r['pre']+2)); a.set_xlabel('t − t10 (s)'); a.set_ylabel('θ (°)')
        a.set_title(f'{hz} Hz {r["take"][-6:]}  Δθ={r["D"]:.1f}°  J_Y={r["J_Y"]:.2f} (R²={r["R2Y"]:.2f})  1/τ={r["J_exp"]:.2f} (R²={r["R2exp"]:.3f})',fontsize=8.5); a.legend(fontsize=7,loc='upper right'); a.grid(alpha=0.2)
    for a in list(axes.flat)[n:]: a.axis('off')
    fig.suptitle(f'每频率一条：θ(t)、J_Y 的拟合窗（黄）与指数拟合（红）',fontsize=12); fig.tight_layout(); fig.savefig(RIM/(OUT_TAG+'theta_lp_gallery.png'),dpi=105)
    # ---- Y(t) 拟合诊断
    fig,axes=plt.subplots(rows,cols,figsize=(5.2*cols,3.4*rows),squeeze=False)
    for a,hz in zip(axes.flat,hzs):
        r=first[hz]; x,Y,m=r['Ypts']; s3,i3=r['Yfit']; xx=x-r['t10']
        a.plot(xx[m],Y[m],'.',color='C0',ms=3,label=f'内点 {m.sum()}'); a.plot(xx[~m],Y[~m],'x',color='gray',ms=5,mew=1.2,label=f'剔除 {(~m).sum()}')
        a.plot(xx,s3*x+i3,'k-',lw=1.5,label=f'J_Y={r["J_Y"]:.2f} R²={r["R2Y"]:.3f}')
        a.set_title(f'{hz} Hz {r["take"][-6:]}  不剔 {r["J_Y_raw"]:.2f}  TS {r["J_Y_ts"]:.2f}',fontsize=9); a.set_xlabel('t − t10 (s)'); a.set_ylabel('Y = ln tan(θ/2)'); a.grid(alpha=0.2); a.legend(fontsize=7)
    for a in list(axes.flat)[n:]: a.axis('off')
    fig.suptitle(f'J_Y 的直线拟合：t10–t90 段的 Y(t)，灰×=按残差 {ROBUST_K:g}·MAD 剔除的点',fontsize=12); fig.tight_layout(); fig.savefig(RIM/(OUT_TAG+'theta_lp_Yfit.png'),dpi=105)
    # ---- 数字
    print(f"\n==== 按频率（{os.environ.get('SETUP','09-09 台位')}，均值±SD）====")
    print(f'{"f":>4} {"n":>2}  {"Δθ":>5}  '+'  '.join(f'{lab[:10]:>12}' for _,lab,_,_ in VARIANTS)+f'  {"J_exp=1/τ":>12}')
    for hz in hzs:
        g=[r for r in res if r['tag']=='new' and r['hz']==hz]
        cells=[]
        for key,_,_,_ in VARIANTS:
            v=[r[key] for r in g if ok(r,key)]; cells.append(f'{np.mean(v):.2f}±{(np.std(v,ddof=1) if len(v)>1 else 0):.2f}' if v else '—')
        ve=[r['J_exp'] for r in g if ok(r,'J_exp')]; ce=f'{np.mean(ve):.2f}±{(np.std(ve,ddof=1) if len(ve)>1 else 0):.2f}' if ve else '—'
        print(f'{hz:4d} {len(g):2d}  {np.mean([r["D"] for r in g]):5.1f}  '+'  '.join(f'{c:>12}' for c in cells)+f'  {ce:>12}')
    print('\n==== J(f) 线性度（各频率均值）====')
    for key,lab,_,_ in VARIANTS+[('J_exp','J_exp = 1/τ',None,None)]:
        if key in LIN: L=LIN[key]; print(f'  {lab:14s}: {L["a"]:.4f}·f{L["b"]:+.3f}  R²_lin={L["R2_lin"]:.3f}   过原点 R²={L["R2_0"]:.3f}')
    print(f'saved results/rim/{OUT_TAG}theta_lp_rate.png, {OUT_TAG}theta_lp_gallery.png, {OUT_TAG}theta_lp_Yfit.png, {OUT_TAG}theta_lp.json（WIN={WIN}）')
