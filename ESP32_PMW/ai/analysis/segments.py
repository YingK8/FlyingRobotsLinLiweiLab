"""把"一条录像多次重复"的 take 切成每次一段，逐段判断能不能用，再交给主流程算 J。

背景：ai/make_tilt.py 生成的 schedule 在一条录像里把单条 take 的流程重复 N 次，
每次 = 斜坡 → 保持 → 断 A+C → 等 post → 全部线圈归零 off（转子停下）→ 下一次。
所以一条录像里有 N 个台阶，中间有 N 个"确定静止"的窗口，后者正好当自转检测的对照。

做三件事：
 1. 找台阶。先用 schedule 的周期算出理论时刻，再在每个理论时刻附近 ±3 s 内找轴向量跳变最大的点。
    比"全局找跳变"稳，因为全关瞬间转子停转、轴也会动，容易被误认成台阶。
 2. 每段判可用（ai/analysis/spin_check）：保持段 vs 该次之前的全关段，自转线峰背比之比 ≥ RATIO_MIN。
    另外要求台阶足够大（Δθ ≥ MIN_STEP）。
 3. 把可用段各自的 axis.csv 切出来写成 results/rim/<take>_r<k>/axis.csv，并生成清单，
    这样主流程 theta_lp.py 不用改就能一段当一条 take 处理。

用法（在 ESP32_PMW 下）：
  uv run python ai/analysis/segments.py <take> --hz 80 --n 10 [--ramp-ms 5000 --hold-ms 5000
      --post-ms 10000 --off-ms 14000 --reset-ms 1000] [--out ai/analysis/takes_<tag>.csv]
然后：
  WIN=multi TAKES_MANIFEST=<清单> OUT_TAG=<前缀> uv run python ai/analysis/theta_lp.py

三叶螺旋桨（no ring，2026-09-14 起）：先跑 ai/analysis/spin_dir.py <take> <hz> 得 results/rim/<take>/spin_dir.json，
本脚本发现它就自动换成"旋向模式"：
  - 起转时刻用画面帧差找（静止的桨没有椭圆，碟沿跟踪在全关段是瞎的，锁不了全关跳变）；
    断 A+C = 起转 + ramp + hold，精确到帧。
  - 可用 = 停转方向逆时针 且 构型相关 ≥ POSE_MIN 且 台阶 ≥ MIN_STEP。
    停转方向而不是起转方向：斜坡中途失步会反向（09-14 100 Hz 第 2 次起转逆时针、停转顺时针、构型翻了）。
    构型与旋向互相独立，两者在 168 次里 161 次一致，剩下 7 次都核对过画面。
  - spin_check 照算照打印但不参与判定：全关段没有跟踪样本，差分判据用不了，而停转时自由减速 5–13 圈本身就证明它在转。
"""
import sys, csv, argparse, json; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np
from ai.analysis.takes import RIM
from ai.analysis.spin_check import spin_ok, RATIO_MIN
MIN_STEP=1.5      # 台阶小于这个度数就不算数据
POSE_MIN=0.90     # 旋向模式：构型相关下限。09-14 分布是双峰：正常 0.90–1.00，异常 0.16–0.76，中间只有 1 次
JUMP_WIN=3.0      # 理论台阶时刻附近的搜索半径（秒）

def load_axis(take, gate=10.0, fname='axis.csv'):
    t,ns,rows=[],[],[]
    for r in csv.DictReader(open(RIM/take/fname)):
        rows.append(r)
        if float(r['agree_deg'])>gate: continue
        t.append(float(r['t'])); ns.append([float(r['nx']),float(r['ny']),float(r['nz'])])
    t=np.array(t); t0=t[0]; t-=t0; ns=np.array(ns); ns/=np.linalg.norm(ns,axis=1,keepdims=True)
    return t, ns, rows, t0

def jump_at(t, ns, t_guess, half=JUMP_WIN):
    """在 t_guess±half 内找相邻两个 1 s 中值轴夹角最大的时刻"""
    best=(np.nan,-1.0)
    for i in range(len(t)):
        if not (t_guess-half < t[i] < t_guess+half): continue
        a=ns[(t>=t[i]-1)&(t<t[i])]; b=ns[(t>=t[i])&(t<t[i]+1)]
        if len(a)<10 or len(b)<10: continue
        ma=np.median(a,0); mb=np.median(b,0); ma/=np.linalg.norm(ma); mb/=np.linalg.norm(mb)
        j=np.degrees(np.arccos(np.clip(ma@mb,-1,1)))
        if j>best[1]: best=(t[i],j)
    return best

def find_lead(t, ns, period, ramp_s, hold_s, post_s, n):
    """提前量（录像开头到按 EN）：扫 0–20 s，取让 N 个"全关"时刻处跳变之和最大的那个。
    全关时刻 = lead + ramp + hold + post + k·period。全关的跳变比断 A+C 更大（转子停转、碟子垂下）。"""
    best=(np.nan,-1.0)
    for lead in np.arange(0.0,20.0,0.25):
        tot=0.0
        for k in range(n):
            ta=lead+ramp_s+hold_s+post_s+k*period
            if ta>t[-1]-1: break
            _,j=jump_at(t,ns,ta,half=1.5); tot+= min(j if np.isfinite(j) else 0, 30.0)   # 封顶：跟踪崩溃的巨跳（195914 有 66°）会劫持锁相
        if tot>best[1]: best=(lead,tot)
    return best[0]

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('take'); ap.add_argument('--hz',type=float,required=True); ap.add_argument('--n',type=int,required=True)
    ap.add_argument('--ramp-ms',type=int,default=None,help='缺省跟 make_tilt：≤100→5000，>100→10000（09-16 起；09-16 之前的 >100 Hz 录像要传 --ramp-ms 15000 --hold-ms 5000）')
    ap.add_argument('--hold-ms',type=int,default=None,help='缺省 ≤100→5000，>100→3000（09-16 起）')
    ap.add_argument('--post-ms',type=int,default=10000); ap.add_argument('--off-ms',type=int,default=14000)
    ap.add_argument('--reset-ms',type=int,default=1000); ap.add_argument('--out',default=None)
    ap.add_argument('--design',default='half',choices=['whole','half','no'],
                    help='跟 make_tilt 的 design 一致：whole（09-17 起的录像）停机时间 ×1.5=22.5 s，>100 Hz 斜坡缺省 15000；half/no 不变')
    ap.add_argument('--tag',default='new')
    ap.add_argument('--lead',type=float,default=None,help='手动指定按 EN 提前量（秒），跳过自动锁相')
    ap.add_argument('--dirjson',default=None,help='缺省 results/rim/<take>/spin_dir.json（存在就用旋向模式）；写 none 强制旧模式')
    ap.add_argument('--axis',default='axis.csv',help='轴来源：axis.csv（立体）或 axis_monoB.csv（单相机 B，见 axis_mono.py）')
    ap.add_argument('--suffix',default='',help='段目录名后缀，例 mB → <take>_mB_r<k>，免得覆盖立体版的段')
    a=ap.parse_args()
    ramp=(a.ramp_ms if a.ramp_ms is not None else (5000 if a.hz<=100 else (15000 if a.design=='whole' else 10000)))/1000
    off_ms=round((a.reset_ms+a.off_ms)*1.5)-a.reset_ms if a.design=='whole' else a.off_ms
    hold=(a.hold_ms if a.hold_ms is not None else (5000 if a.hz<=100 else 3000))/1000; post=a.post_ms/1000; off=(off_ms+a.reset_ms)/1000; period=ramp+hold+post+off
    t,ns,rows,t0=load_axis(a.take,fname=a.axis)
    dj=Path(a.dirjson) if a.dirjson and a.dirjson!='none' else RIM/a.take/'spin_dir.json'
    DIR=json.load(open(dj)) if (a.dirjson!='none' and dj.exists()) else None
    if DIR:
        # spin_dir 的时间从 frames.csv 第一帧算起；axis 的 t 从第一个跟踪到的帧算起（静止的桨跟踪不到，两者可差好几秒）
        f0=float(next(csv.DictReader(open(f'results/flights/{a.take}/frames.csv')))['t_capture'])
        off=f0-t0; byk={r['k']:r for r in DIR['rows']}
        print(f'{a.take}: 旋向模式（{dj}），录像 {t[-1]+(-off):.0f} s，每次 {period:.0f} s')
        print(f'{"段":>3} {"断A+C(s)":>9} {"全关(s)":>8} {"起转":>4} {"停转":>4} {"构型":>5} {"Δθ(°)":>7} {"自转比":>9}  可用')
    else:
        lead=a.lead if a.lead is not None else find_lead(t,ns,period,ramp,hold,post,a.n)
        print(f'{a.take}: 录像 {t[-1]:.0f} s，每次 {period:.0f} s，推定按 EN 的提前量 {lead:.1f} s')
        print(f'{"段":>3} {"断A+C(s)":>9} {"全关(s)":>8} {"跳变(°)":>7} {"Δθ(°)":>7} {"自转比":>9} {"信号":>4}  可用')
    man=[]; info=[]
    try:
        from ai.analysis.spin_check import _load as _sc_load
        REST_TS,_=_sc_load(a.take)
    except Exception: REST_TS=None
    REST_SEEN=[]
    if REST_TS is not None:                     # 先按理论网格扫一遍所有全关窗，让第一段也能借到共享对照
        base=(0.0 if DIR else lead)
        for kk in range(a.n):
            w=(base+ramp+hold+post+kk*period+2.0, base+ramp+hold+post+kk*period+off-0.5)
            if w[1]<=REST_TS[-1]:
                REST_SEEN.append((int(((REST_TS>=w[0])&(REST_TS<w[1])).sum()),w))
    for k in range(a.n):
        if DIR:
            r=byk.get(k+1,{})
            if r.get('onset') is None: print(f'{k+1:3d}  {r.get("why","没有起转时刻")}，跳过'); continue
            tc=r['onset']+off+ramp+hold; ta=tc+post; jump=float('nan')
            if ta>t[-1]+1: print(f'{k+1:3d}  录像不够长，跳过'); continue
        else:
            ta_guess=lead+ramp+hold+post+k*period
            if ta_guess>t[-1]-1: print(f'{k+1:3d}  录像不够长（全关 {ta_guess:.0f} s 超出 {t[-1]:.0f} s），跳过'); continue
            ta,jump=jump_at(t,ns,ta_guess)        # 全关事件：跳变最大的那个
            if not np.isfinite(ta): print(f'{k+1:3d}  没找到全关事件'); continue
            tc=ta-post                            # 真正的断 A+C 时刻，在全关之前 post 秒
        if tc-3.5<0: print(f'{k+1:3d}  断电时刻 {tc:.1f} s 之前基线不够，跳过'); continue
        # 台阶大小：断电前 3.5–1 s 与断电后 6–10 s 的平均轴夹角
        pre=ns[(t>=tc-3.5)&(t<tc-1)]; pos=ns[(t>=tc+6)&(t<min(tc+post-0.5, ta-0.5))]
        if len(pre)<20 or len(pos)<20: print(f'{k+1:3d}  窗口样本不足'); continue
        mp=pre.mean(0); mq=pos.mean(0); mp/=np.linalg.norm(mp); mq/=np.linalg.norm(mq)
        D=float(np.degrees(np.arccos(np.clip(mp@mq,-1,1))))
        # 自转检测：保持段 vs 本次之前的全关段（第 1 次用录像开头）
        hold_win=(tc-min(4.0,hold-0.5), tc-0.3)
        # 对照窗要取全关段的"后半"：断电后转子还在自由减速，全关段前半仍有残余自转线，
        # 拿它当静止参考会把比值压下去（150902 第 2 段就是这么被误判的）。
        # 静止对照：本次全关之后 2 s（实测 1–2 s 内转子就停）到全关结束前 0.5 s
        rest_win=(ta+2.0, ta+off-0.5)
        # 全关期间碟子偶尔晃到跟踪丢失，本段静止窗可能几乎没样本（170910 r1 只有 36 个）→
        # 改用同 take 已见过的、样本最多的全关窗做共享对照（相机与背景不变，差分判据仍成立）
        if REST_TS is not None:
            n_own=int(((REST_TS>=rest_win[0])&(REST_TS<rest_win[1])).sum())
            REST_SEEN.append((n_own,rest_win))
            if n_own<150 and REST_SEEN:
                nb,wb=max(REST_SEEN)
                if nb>=150: rest_win=wb
        ok_spin,d=spin_ok(a.take,a.hz,hold_win,rest_win)
        if DIR:
            st=r.get('dir_stop'); pm=r.get('pose_min') or 0.0
            # 停转方向 None = 检测器无数据（新 robot 停得快，慢速段只有 0.5-3 圈数不出旋向，2026-09-22 起出现），
            # 不是构型证据。此时回退到 起转 CCW + 构型相关——停转规则要防的"斜坡失步反转"本来就会把
            # pose_min 打到 ~0.25（09-14 反例），构型闸单独就能抓住。停转明确 CW 的照旧不可用。
            dir_ok = st=='CCW' or (st is None and r.get('dir')=='CCW')
            why=('' if dir_ok else f'停转方向 {st} ')+('' if pm>=POSE_MIN else f'构型相关 {pm:.2f}<{POSE_MIN} ')+('' if D>=MIN_STEP else '台阶太小 ')
            usable=not why
            print(f'{k+1:3d} {tc:9.1f} {ta:8.1f} {str(r.get("dir")):>4} {str(st):>4} {pm:5.2f} {D:7.1f} {d.get("ratio",float("nan")):9.1f}  {"可用" if usable else "不可用："+why}')
            info.append(dict(seg=k+1,tc=float(tc),t_alloff=float(ta),D=D,dir_start=r.get('dir'),dir_stop=st,pose=pm,ratio=d.get('ratio'),spin_ok=bool(ok_spin),usable=usable,why=why))
        else:
            usable=bool(ok_spin and D>=MIN_STEP)
            print(f'{k+1:3d} {tc:9.1f} {ta:8.1f} {jump:7.1f} {D:7.1f} {d.get("ratio",float("nan")):9.1f} {str(d.get("signal")):>4}  {"可用" if usable else "不可用："+("台阶太小 " if D<MIN_STEP else "")+d.get("why","")}')
            info.append(dict(seg=k+1,tc=float(tc),t_alloff=float(ta),jump=float(jump),D=D,ratio=d.get('ratio'),signal=d.get('signal'),usable=usable,why=d.get('why','')))
        if not usable: continue
        # 切出这一段的 axis.csv（台阶前 6 s 到台阶后 post-0.5 s），时间戳保持原样
        name=f'{a.take}_{a.suffix}_r{k+1}' if a.suffix else f'{a.take}_r{k+1}'; (RIM/name).mkdir(parents=True,exist_ok=True)
        lo,hi=t0+tc-6.0, t0+min(tc+post-0.5, ta-0.5)
        sub=[r for r in rows if lo<=float(r['t'])<=hi]
        with open(RIM/name/'axis.csv','w',newline='') as f:
            fields=['frame','t','nx','ny','nz','agree_deg']; w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(sub)
        man.append((name,int(a.hz),a.tag))
    out=a.out or f'ai/analysis/takes_{a.take[-6:]}.csv'
    with open(out,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['take','hz','tag']); w.writerows(man)
    json.dump(info,open(RIM/f'{a.take}_segments.json','w'),indent=1)
    print(f'\n可用 {len(man)}/{a.n} 段，清单 {out}')
    print(f'接着跑：WIN=multi TAKES_MANIFEST={out} OUT_TAG={a.take[-6:]}_ uv run python ai/analysis/theta_lp.py')
