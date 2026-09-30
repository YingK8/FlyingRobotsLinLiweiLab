"""判断一段录像里转子有没有按驱动频率转起来，也就是这一段能不能用。

为什么不能只看"有没有台阶"：转子不转时，断 A+C 一样会让碟子像单摆一样甩过去，θ 照样有台阶，
但那不是对齐动力学。可靠的信号是自转谱线。

谱线在哪：相机时间戳是 USB 到达时间（帧成串到达，串节律 f_b ≈ 39 Hz），一条真实的 f Hz 线会被
相位调制成 f、f ± f_b，并且都会在镜像轴 M 附近折回（M 用"真实时间戳采样一个合成 f Hz 正弦"标定，
这台相机 M ≈ 250 Hz）。f 越高调制越深，120 Hz 以上 f 本身那条线几乎消失，最强的是边带或镜像。
所以候选集合 = {f, 2f} ± {0, f_b}，再加各自的镜像。

主判据是**差分**：同一条录像里拿一段确定静止的时间当对照（多次重复的 schedule 里就是两次之间
全部线圈归零那 15 s；单条 take 就用按 EN 之前的头几秒），比较候选频率上的谱功率。
  比值 = 保持段峰背比 / 静止段峰背比（峰背比先在各自窗口内归一），取候选频率和三路信号里最大的那个。
实测（2026-09-11）：真正在转的 10 条给 111–64000×，确定没转的 7 段给 3.6–14×，中间没有重叠。
差分能自动抵消掉跟踪器噪声、万向节晃动这些两段都有的东西，比绝对信噪比稳得多。

没有静止对照段时退回到备用判据：把窗口切成约 1 s 的子窗，要求 ≥60% 的子窗最强峰彼此一致
（中位 ±2.5 Hz）、峰背比 ≥ 8、且该峰落在候选频率上。这个在低频可靠，高频（≥60 Hz、曝光把
碟子抹匀的情况）会偏保守。

用法：
  uv run python ai/analysis/spin_check.py <take> <hz> <hold_t0> <hold_t1> [rest_t0 rest_t1]
  from ai.analysis.spin_check import spin_ok
"""
import sys, csv; sys.path.insert(0,'.')
import numpy as np
from scipy.signal import lombscargle
from ai.analysis.takes import RIM
RATIO_MIN=30.0      # 差分判据阈值（实测正样本最低 111，负样本最高 14）
SNR_MIN=8.0; HIT_FRAC=0.6; TOL_HZ=2.5

def _load(take):
    """返回 t 和三路信号：倾角 θ、椭圆中心 cx、cy。
    不同构型自转信号出现在不同信号里：实心碟子在 θ 上最强；half ring（叶片模糊成对称亮环）θ 几乎没有
    调制，但质心偏摆（动不平衡）在 cx/cy 上很清楚。所以三路都算，取最强的一路。"""
    rows=list(csv.DictReader(open(RIM/take/'tilt_A.csv')))
    t=np.array([float(r['t']) for r in rows]); t-=t[0]
    sig={k:np.array([float(r[c]) for r in rows]) for k,c in (('θ','theta_deg'),('cx','cx'),('cy','cy'))}
    return t, sig

def candidates(t, tp, hz):
    """{f,2f} ± {0,f_b} 及其镜像。f_b 限制在 25–55 Hz（成串节律基频，别取到二次谐波）。"""
    pr=np.linspace(5,320,3000); pp=lombscargle(tp,np.sin(2*np.pi*hz*tp),2*np.pi*pr,normalize=True)
    pp[abs(pr-hz)<4]=0; M=hz+float(pr[np.argmax(pp)])
    dt=np.diff(t); fg=np.linspace(25,55,300)
    pb=lombscargle(t[1:],dt-dt.mean(),2*np.pi*fg,normalize=True); f_b=float(fg[np.argmax(pb)])
    c=set()
    for h in (hz,2*hz):
        for v in (h, h+f_b, abs(h-f_b)):
            for w in (v, abs(M-v)):
                if 8<w<M*0.98: c.add(round(w,1))
    return sorted(c), M, f_b

def _snr(t,x,a,b,freqs,M,nmin=150):
    """窗口内每个候选频率的峰背比 = 该频率的谱功率 / 本窗 8–0.98M Hz 的中位功率。
    先在各自窗口内归一，再跨窗口比，才不会被两段整体摆幅不同带偏（质心信号尤其明显）。"""
    k=(t>=a)&(t<b); ts=t[k]; xs=x[k]
    if len(ts)<nmin: return None
    xs=xs-xs.mean(); xs=xs-np.polyval(np.polyfit(ts,xs,1),ts)
    P=np.atleast_1d(lombscargle(ts,xs,2*np.pi*np.asarray(freqs,float),normalize=True))
    fb_=np.linspace(8,min(300,M*0.98),1200); bg=float(np.median(lombscargle(ts,xs,2*np.pi*fb_,normalize=True)))
    return P/max(bg,1e-12)

def spin_ok(take, hz, hold, rest=None, series=None):
    """hold=(t0,t1) 应该在转的窗口；rest=(t0,t1) 确定静止的对照窗口（没有就用备用判据）。"""
    t,sig = series if series is not None else _load(take)
    tp=t[(t>=hold[0])&(t<hold[1])]
    if len(tp)<150: return False, dict(why=f'保持段只有 {len(tp)} 个样本（窗口 {hold[0]:.1f}–{hold[1]:.1f} s 超出录像 0–{t[-1]:.1f} s？）', mode='—')
    C,M,f_b = candidates(t, tp, hz)
    if rest is not None:
        best=(0.0,None,None)
        for name,x in sig.items():
            Ph=_snr(t,x,hold[0],hold[1],C,M); Pr=_snr(t,x,rest[0],rest[1],C,M)
            if Ph is None or Pr is None: continue
            r=Ph/np.maximum(Pr,1e-9); i=int(np.argmax(r))
            if r[i]>best[0]: best=(float(r[i]),C[i],name)
        if best[1] is not None:
            ok=bool(best[0]>=RATIO_MIN)
            return ok, dict(mode='差分', ratio=best[0], at_hz=best[1], signal=best[2], expect=C, mirror=M, burst=f_b,
                            why='' if ok else f'自转线功率只比静止段高 {best[0]:.0f}×（需 ≥{RATIO_MIN:.0f}×）')
    x=sig['θ']
    Ph=_snr(t,x,hold[0],hold[1],C,M)
    if Ph is None: return False, dict(why='保持段样本太少', mode='—')
    # 备用：子窗一致性
    nsub=int(np.clip(round((hold[1]-hold[0])/1.0),3,6)); edges=np.linspace(hold[0],hold[1],nsub+1); pk=[]; sn=[]
    for i in range(nsub):
        kk=(t>=edges[i])&(t<edges[i+1])
        if kk.sum()<100: continue
        ts=t[kk]; xs=x[kk]-x[kk].mean(); xs=xs-np.polyval(np.polyfit(ts,xs,1),ts)
        f=np.linspace(8,min(300,M*0.98),4000); P=lombscargle(ts,xs,2*np.pi*f,normalize=True)
        i=int(np.argmax(P))
        if i>=0.97*len(f): continue   # 峰贴网格上边缘 = 归一化谱对噪声的边缘伪影，不是谱线（2026-09-16，170910 的"244 Hz"）
        pk.append(float(f[i])); sn.append(float(P.max()/np.median(P)))
    if not pk: return False, dict(why='样本太少', mode='备用')
    med=float(np.median(pk)); hits=sum(1 for p in pk if abs(p-med)<=TOL_HZ); snr=float(np.median(sn))
    on=min(abs(med-c) for c in C)<=TOL_HZ
    ok=(hits/len(pk)>=HIT_FRAC) and snr>=SNR_MIN and on
    why='' if ok else ('各子窗峰位不一致' if hits/len(pk)<HIT_FRAC else ('谱峰太弱' if snr<SNR_MIN else f'峰在 {med:.0f} Hz，不是自转线'))
    return ok, dict(mode='备用', hits=hits, nsub=len(pk), peak=med, snr=snr, expect=C, mirror=M, burst=f_b, why=why)

if __name__=='__main__':
    take=sys.argv[1]; hz=float(sys.argv[2]); hold=(float(sys.argv[3]),float(sys.argv[4]))
    rest=(float(sys.argv[5]),float(sys.argv[6])) if len(sys.argv)>6 else None
    ok,d=spin_ok(take,hz,hold,rest)
    print(f'{take} @ {hz:g} Hz  保持段 {hold[0]}–{hold[1]} s: {"转起来了" if ok else "没转起来（"+d.get("why","")+"）"}  [{d.get("mode")}判据]')
    if d.get('mode')=='差分': print(f'  自转线功率是静止段的 {d["ratio"]:.0f} 倍 @ {d["at_hz"]:.0f} Hz（信号 {d.get("signal")}）')
    else: print(f'  子窗命中 {d.get("hits")}/{d.get("nsub")}，峰 {d.get("peak",float("nan")):.0f} Hz，峰背比 {d.get("snr",float("nan")):.0f}')
