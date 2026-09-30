"""一条（或多条）新 take 的全流程：碟沿跟踪 → 立体轴 axis.csv → b=3° 校正 axis_b3.csv → 登记到 takes.csv → 自转同步检查。
在 ESP32_PMW 目录下运行：
  uv run python ai/analysis/process_take.py 2026-09-10_101010:120 2026-09-10_101230:120 ... [--tag new] [--refresh]
--refresh 最后自动重跑 batch_v4_rim.py + jband_rim.py，更新 results/rim/ 里的趋势图。
同步检查：单相机 tilt_A.csv 的 theta_deg 做 Lomb-Scargle（不等间隔采样，能看到 Nyquist 以上的线），
每 2 s 报最强峰；保持段的峰应该 = 驱动频率（自转同步）。峰跑到明显更低的频率 = 失步，这条要作废。"""
import sys, os, argparse, csv, subprocess
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT); sys.path.insert(0,'.')
import numpy as np
from ai.analysis.takes import RIM, add_take
def sync_check(take,hz):
    """自转是否同步于驱动频率。注意：frames.csv 的时间戳是 USB 到达时间，不是曝光时刻——帧成串到达，
    串的节律 ≈39 Hz、串内间隔 ≈4 ms（局部 ≈250 fps）、串间 ≈9 ms。按这种时间戳做谱，一条真实的 f 线会被
    相位调制成 f、f±39、以及它们的镜像 250−(…)；f 越高调制越深，到 ~150 Hz 时 f 本身那条线几乎消失，
    只剩 f±39 和镜像（2026-09-09 用合成正弦验证过：60/190 之类的峰就是 150 Hz 的镜像边带）。
    所以这里把这一组频率都算作"同步"。真正失步的判据是：峰落在这组之外，且随时间往下漂。"""
    from scipy.signal import lombscargle
    rows=list(csv.DictReader(open(RIM/take/'tilt_A.csv')))
    t=np.array([float(r['t']) for r in rows]); x=np.array([float(r['theta_deg']) for r in rows]); t-=t[0]
    dt=np.diff(t)
    fb_grid=np.linspace(15,90,1500); pb=lombscargle(t[1:],dt-dt.mean(),2*np.pi*fb_grid,normalize=True); f_b=float(fb_grid[np.argmax(pb)])   # 成串节律
    # 镜像轴 M：用真实时间戳对一个纯 hz Hz 正弦做谱，第二高峰在 M−hz 处（对这台相机 M≈250 Hz，不等于 1/中位帧间隔）
    kk=(t>t[-1]*0.4)&(t<t[-1]*0.4+4); tp=t[kk]; fpr=np.linspace(5,300,6000); pp=lombscargle(tp,np.sin(2*np.pi*hz*tp),2*np.pi*fpr,normalize=True)
    pp[abs(fpr-hz)<4]=0; M=hz+float(fpr[np.argmax(pp)]); f_loc=M
    base=[hz,hz+f_b,abs(hz-f_b),hz+2*f_b,abs(hz-2*f_b)]
    expect=sorted({round(v,1) for v in base+[abs(M-v) for v in base]+[abs(2*hz-M)] if 15<v<M})   # <15 Hz 是万向节摆动的地盘，不算
    near=lambda fp: min(abs(fp-e) for e in expect)<=3.0
    f=np.linspace(3,max(60,f_loc*0.98),4000); out=[]
    for t0 in np.arange(0,t[-1]-2,2.0):
        k=(t>=t0)&(t<t0+2)
        if k.sum()<100: continue
        xx=x[k]-x[k].mean(); xx-=np.polyval(np.polyfit(t[k],xx,1),t[k])
        p=lombscargle(t[k],xx,2*np.pi*f,normalize=True); out.append((t0,f[int(np.argmax(p))],float(p.max())))
    print(f'  自转同步检查（镜像轴 {M:.0f} Hz，成串节律 {f_b:.1f} Hz；同步于 {hz} Hz 时允许的谱峰: '+' '.join(f'{e:.0f}' for e in expect)+'）')
    print('   每 2 s 最强谱峰 Hz（* = 属于上面那组）: '+'  '.join(f'{t0:.0f}s:{fp:.0f}{"*" if near(fp) else ""}' for t0,fp,_ in out))
    first=next((i for i,(t0,fp,_) in enumerate(out) if near(fp) and fp>8),None)
    if first is None: print(f'  !! 没有任何窗口出现与 {hz} Hz 相容的自转线 — 可能没同步上或没转起来'); return
    after=out[first:]; bad=[(t0,fp) for t0,fp,_ in after if (not near(fp)) and fp>8]
    print(f'  从 {out[first][0]:.0f} s 起 {len(after)} 个窗口中 {len(after)-len(bad)} 个相容'+(f'；不相容的：'+' '.join(f'{t0:.0f}s:{fp:.0f}' for t0,fp in bad)+'  ← 若峰随时间往下漂就是失步' if bad else '  ✓'))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('items',nargs='+',help='take:hz，例如 2026-09-10_101010:120'); ap.add_argument('--tag',default='new'); ap.add_argument('--refresh',action='store_true'); ap.add_argument('--no-track',action='store_true',help='跳过碟沿跟踪（已有 tilt_A/B.csv 时）')
    a=ap.parse_args()
    from ai.rim_track import run_take
    from ai.analysis.axis_corr import correct_take
    for it in a.items:
        take,hz=it.split(':'); hz=int(round(float(hz)))
        print(f'=== {take} @ {hz} Hz')
        if not a.no_track: run_take(take)
        correct_take(take)
        if a.tag=='none': print('  --tag none：不登记到 takes.csv')
        else: print('  登记到 takes.csv' if add_take(take,hz,a.tag) else '  takes.csv 里已有')
        sync_check(take,hz)
    if a.refresh: subprocess.run([sys.executable,'ai/analysis/refresh.py'],check=True)
