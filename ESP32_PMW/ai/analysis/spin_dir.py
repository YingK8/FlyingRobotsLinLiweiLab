"""三叶螺旋桨（no ring 构型）每次起转的旋向：逆时针 = 正常，顺时针 = 反推力，整机被推到奇怪构型。

为什么需要：spin_check 只看"有没有自转线"，实值信号的谱分不出正反转 —— 反转的桨一样有 f 线。
2026-09-14 no ring 批次里有些重复一开始顺时针转、之后整机飞到别的构型或乱飞（用户观察），
这些段的台阶物理上不是同一个实验，必须剔掉。

做法（纯图像，不依赖碟沿跟踪；静止的桨没有椭圆，跟踪器在起转瞬间是瞎的）：
 1. 起转窗口内逐帧灰度的 max − min 投影 = 叶片扫过的区域（静止的轴、轴承相减抵消），
    取最大连通域拟椭圆 → 旋转中心 + 把斜视的圆归一化回单位圆的仿射。
 2. 桨有三片叶子，所以图像绕中心的 3 次角谐波 m3 = Σ (I − min)·e^{−3iφ} 的相位 = 3 × 叶片角。
    相邻帧相位差 /3 = 每帧转角，只在 (−60°, 60°] 内无歧义（三重对称 + 帧率）。
 3. 斜坡从 1 Hz 起步，开头几秒每帧转角远小于 60°：从开始转动累加到平滑后的每帧转角首次超过 40°，
    累计转角的符号就是旋向。仿射保持手性；图像 y 轴朝下，所以 φ 增大 = 屏幕上顺时针。

用法：uv run python ai/analysis/spin_dir.py            # 合成数据自检
      from ai.analysis.spin_dir import sense_window     # 对真实录像的一个起转窗口
"""
import math
import numpy as np
import cv2

STOP_DEG = 40.0      # 每帧视在转角超过它就停止累加（离 60° 的混叠边界留余量）
MOVE_DEG = 1.5       # 小于它算没在转
MIN_TURNS = 1.0      # 累计不到一圈不下结论


def _swept_ellipse(G):
    """max−min 投影的最大连通域 → 椭圆 ((cx,cy),(d1,d2),ang)。G: (N,H,W) uint8。"""
    M = G.max(0).astype(np.int16); m = G.min(0).astype(np.int16); D = (M - m).astype(np.uint8)
    thr = max(40, int(0.35 * np.percentile(D, 99.9)))
    bw = (D > thr).astype(np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    if n < 2: return None, m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)
    if len(c) < 20: return None, m
    return cv2.fitEllipse(c), m


def sense_window(G, max_shift=50):
    """一个起转窗口（从静止开始的连续帧）→ dict(sense='CCW'|'CW'|None, turns, same_frac, curve, ...)。
    sense 是屏幕上看到的旋向（图像 y 朝下已处理）。
    每帧转角 = 相邻两帧"椭圆归一化后的角向亮度剖面"的循环互相关峰位，限制在 ±max_shift°
    （三片叶子 → 真实转角超过 60° 就会认错；起步阶段每帧只转几度到二十几度）。
    斜视 + 桨距让三片叶子在图上长得不一样，所以不假设 3 次谐波，直接用整条剖面做互相关；
    停顿的帧峰位在 0 附近，不引入偏差。"""
    G = np.asarray(G)
    e, m = _swept_ellipse(G)
    if e is None: return dict(sense=None, why='找不到叶片扫过的区域')
    (cx, cy), (d1, d2), ang = e; a, b = d1 / 2, d2 / 2
    if min(a, b) < 8: return dict(sense=None, why=f'扫过区域太小 ({a:.0f}×{b:.0f}px)', ell=e)
    th = math.radians(ang); c, s = math.cos(th), math.sin(th)
    phis = np.radians(np.arange(360)); rs = np.linspace(0.35, 1.0, 12)
    U = rs[:, None] * np.cos(phis)[None, :]; V = rs[:, None] * np.sin(phis)[None, :]
    X = (cx + a * U * c - b * V * s).astype(np.float32); Y = (cy + a * U * s + b * V * c).astype(np.float32)   # 保持手性
    mp = cv2.remap(m.astype(np.float32), X, Y, cv2.INTER_LINEAR)
    prof = np.array([(cv2.remap(g.astype(np.float32), X, Y, cv2.INTER_LINEAR) - mp).mean(0) for g in G])   # (N,360)
    prof -= prof.mean(1, keepdims=True)
    F = np.fft.fft(prof, axis=1)
    C = np.real(np.fft.ifft(F[1:] * np.conj(F[:-1]), axis=1))            # C[k,s] = Σ p_{k}(φ+s) p_{k+1}... 见下
    # ifft(F_k·conj(F_{k−1}))(s) = Σ_φ p_{k−1}(φ)·p_k(φ+s)：峰在 s 处 ⇔ 图案转了 +s（φ 增大方向）
    lags = np.r_[np.arange(0, max_shift + 1), np.arange(-max_shift, 0)]
    Cw = C[:, lags]; j = np.argmax(Cw, 1); dphi = lags[j].astype(float)
    # 抛物线亚度
    jm = (j - 1) % len(lags); jp = (j + 1) % len(lags); rr = np.arange(len(j))
    y0, ym, yp = Cw[rr, j], Cw[rr, jm], Cw[rr, jp]; den = ym - 2 * y0 + yp
    ok = (np.abs(lags[jm] - lags[j]) == 1) & (np.abs(lags[jp] - lags[j]) == 1) & (np.abs(den) > 1e-9)
    dphi = dphi + np.where(ok, 0.5 * (ym - yp) / np.where(ok, den, 1), 0)
    energy = np.sqrt((prof[1:] ** 2).mean(1) * (prof[:-1] ** 2).mean(1)) + 1e-9
    qual = C.max(1) / (energy * 360)                                      # 归一化互相关峰值，1 = 完全一样
    k = 9; pad = np.pad(dphi, (k // 2, k // 2), mode='edge')
    sm = np.array([np.median(pad[i:i + k]) for i in range(len(dphi))])
    moving = np.where(np.abs(sm) > MOVE_DEG)[0]
    if len(moving) == 0: return dict(sense=None, why='窗口内没转', ell=e)
    i0 = moving[0]
    fast = np.where((np.abs(sm[i0:]) > STOP_DEG) | (qual[i0:] < 0.3))[0]   # 快到接近混叠，或叶片糊了
    i1 = i0 + (fast[0] if len(fast) else len(sm) - i0)
    sel = np.arange(i0, i1)
    if len(sel) < 10: return dict(sense=None, why=f'慢速段太短 ({len(sel)} 帧)', ell=e)
    net = float(dphi[sel].sum()); turns = net / 360.0
    mv = sel[np.abs(dphi[sel]) > MOVE_DEG]
    same = float(np.mean(np.sign(dphi[mv]) == np.sign(net))) if len(mv) else 0.0
    out = dict(turns=round(turns, 2), same_frac=round(same, 2), n=int(len(sel)), i0=int(i0), i1=int(i1), ell=e,
               peak_rate=float(np.abs(sm[i0:i1]).max()), curve=np.cumsum(dphi).tolist())
    if abs(turns) < MIN_TURNS:
        out.update(sense=None, why=f'慢速段累计只转了 {turns:+.2f} 圈'); return out
    out['sense'] = 'CW' if net > 0 else 'CCW'                           # y 朝下：φ 增大 = 屏幕顺时针
    return out


def _synth(sign, F=80.0, T=5.0, fps=190.0, dur=2.5, H=400, W=640):
    """斜视的三叶桨，余弦斜坡 1→F Hz，外加一根静止的亮轴。sign=+1 屏幕逆时针。"""
    N = int(dur * fps); t = np.arange(N) / fps
    f = 1 + (F - 1) * (1 - np.cos(np.pi * np.clip(t - 0.3, 0, None) / T)) / 2
    f[t < 0.3] = 0
    f = f * (0.55 + 0.45 * np.sign(np.sin(2 * np.pi * 3 * t)))           # 起步时一顿一顿的
    ang = 2 * np.pi * np.cumsum(f) / fps
    G = np.full((N, H, W), 40, np.uint8)
    cx, cy, a, b, tilt = 330.0, 210.0, 90.0, 45.0, math.radians(25)
    ct, st = math.cos(tilt), math.sin(tilt)
    for i in range(N):
        img = G[i]
        cv2.line(img, (330, 210), (360, 40), 200, 6)                     # 静止的轴
        for blade in range(3):
            base = -sign * ang[i] + blade * 2 * np.pi / 3                   # 屏幕逆时针 = 数学角在 y 朝下坐标里减小
            pts = []
            wid = [0.18, 0.06, 0.30][blade]                                     # 三片叶子看起来不一样
            for rr, dd in [(0.15, -wid), (1.0, -0.05), (1.0, wid), (0.15, wid)]:
                al = base + dd; u, v = rr * math.cos(al), rr * math.sin(al)
                x = cx + a * u * ct - b * v * st; y = cy + a * u * st + b * v * ct
                pts.append((int(round(x)), int(round(y))))
            cv2.fillPoly(img, [np.array(pts, np.int32)], 250)
    return G


if __name__ == '__main__' and len(__import__('sys').argv) == 1:
    for sign, want in [(+1, 'CCW'), (-1, 'CW')]:
        for F, T in [(20, 5), (80, 5), (150, 15)]:
            res = sense_window(_synth(sign, F=F, T=T, dur=3.0 if T == 5 else 5.0))
            print(f'  synth {want:>3} ramp→{F:3d} Hz: got {res["sense"]}  turns {res.get("turns")}  '
                  f'same {res.get("same_frac")}  peak {res.get("peak_rate", 0):.0f}°/frame')
            assert res['sense'] == want, res
    print('spin_dir self-check OK')


# ---------------------------------------------------------------- 整条多次重复录像
def schedule(hz, ramp_ms=None, hold_ms=None, post_ms=10000, off_ms=14000, reset_ms=1000):
    """与 ai/make_tilt.py 的缺省一致（2026-09-16 起）：≤100 Hz 斜坡 5 s/保持 5 s，>100 Hz 斜坡 10 s/保持 3 s。
    09-16 之前的 >100 Hz 录像用的是 15 s/5 s —— 分析那些老 take 时要显式传 ramp_ms=15000, hold_ms=5000。
    返回 (ramp, hold, post, period) 秒。"""
    ramp = (ramp_ms if ramp_ms is not None else (5000 if hz <= 100 else 10000)) / 1000
    hold = (hold_ms if hold_ms is not None else (5000 if hz <= 100 else 3000)) / 1000
    post = post_ms / 1000
    return ramp, hold, post, ramp + hold + post + (off_ms + reset_ms) / 1000


def motion_energy(take, stride=4, cam='B'):
    """整条录像的帧差能量（1/4 分辨率）。返回 t, E。"""
    import csv
    ts = np.array([float(r['t_capture']) for r in csv.DictReader(open(f'results/flights/{take}/frames.csv'))]); ts -= ts[0]
    cap = cv2.VideoCapture(f'results/flights/{take}/{cam}/{cam}.mp4'); prev = None; T, E = [], []; i = 0
    while cap.grab():
        if i % stride == 0:
            ok, f = cap.retrieve()
            if not ok: break
            g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (160, 100), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev is not None: E.append(float(np.abs(g - prev).mean())); T.append(ts[min(i, len(ts) - 1)])
            prev = g
        i += 1
    return np.array(T), np.array(E)


def find_lead(t, E, period, active):
    """按 schedule 周期锁定第一次起转的时刻（可以是负的：按 EN 早于开始录像，第一次只录到一半）。
    扫一个完整周期的相位，活动段（斜坡+保持+post）与静止段的能量中位数差最大者胜。
    不用"静止之后第一次动"，因为乱飞的那几次在全关期间也在动，会产生假起点。"""
    best = (0.0, -np.inf)
    for ph in np.arange(0.0, period, 0.25):
        q = (t - ph) % period
        on = q < active; off = (q > active + 3.0) & (q < period - 0.5)
        if on.sum() < 50 or off.sum() < 50: continue
        sc = np.median(E[on]) - np.median(E[off])
        if sc > best[1]: best = (float(ph), sc)
    ph = best[0]
    return ph - period if ph + active - period > 1.0 and np.median(E[t < min(2.0, ph)]) > np.median(E) else ph


def refine_onset(take, s, cam='B', pad=4.0):
    """schedule 给的起转时刻附近，用画面帧差找真正开始动的那一帧（要求之后 5 帧里至少 3 帧也在动）。"""
    ts, G = read_window(take, cam, max(0.0, s - pad), s + pad, stamps=True)
    if len(G) < 100: return s, 'window'
    d = np.abs(np.diff(G[:, ::4, ::4].astype(np.int16), axis=0)).mean((1, 2))
    base = np.percentile(d, 15); thr = max(3 * base, base + 1.5)
    for k in np.where(d > thr)[0]:
        if (d[k:k + 5] > thr).sum() >= 3: return float(ts[k]), 'ok'
    return s, 'no-motion'


def read_window(take, cam, t0, t1, stamps=False):
    import csv
    ts = np.array([float(r['t_capture']) for r in csv.DictReader(open(f'results/flights/{take}/frames.csv'))]); ts -= ts[0]
    i0, i1 = int(np.searchsorted(ts, t0)), int(np.searchsorted(ts, t1))
    cap = cv2.VideoCapture(f'results/flights/{take}/{cam}/{cam}.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    G = []
    for _ in range(i1 - i0):
        ok, f = cap.read()
        if not ok: break
        G.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    return (ts[i0:i0 + len(G)], np.array(G)) if stamps else np.array(G)


def verdict(r):
    """两台相机合议：至少一台判逆时针、且没有一台判顺时针 → 'CCW'；有顺时针没逆时针 → 'CW'；
    两台打架或都没结论 → None（交给构型检查）。2026-09-14 80 Hz 十次与画面逐一核对：判出来的全对。"""
    v = {r.get('A'), r.get('B')} - {None}
    return v.pop() if len(v) == 1 else None


def regrid(d, tol=0.5):
    """起转时刻必须落在 schedule 网格上（固件计时是精确的）。用各次画面起转拟合网格 t1+(k−1)·P，
    偏离网格 > tol 的那几次改用网格值；若整体残差中位就 > 0.1 s（例：15 s 斜坡 1 Hz 起步太缓，阈值穿越抖 ±1 s），全部用网格值。
    残差中位本身写进 d['grid_resid']：若远大于 0.1 s，说明锁错了相位（09-11 半环 80/110 Hz 就是），整条的判定不能信。"""
    P = d['period']; on = [(r['k'], r['onset']) for r in d['rows'] if r.get('onset') is not None]
    if len(on) < 3: d['grid_resid'] = None; return d
    k = np.array([a for a, _ in on]); t = np.array([b for _, b in on])
    t1 = float(np.median(t - (k - 1) * P)); res = t - (t1 + (k - 1) * P); med = float(np.median(np.abs(res)))
    d['grid_t1'] = t1; d['grid_resid'] = med
    for r in d['rows']:
        if r.get('onset') is None: continue
        g = t1 + (r['k'] - 1) * P
        if med > 0.1 or abs(r['onset'] - g) > tol:
            r['onset_raw'] = r['onset']; r['onset'] = round(g, 3)
    return d


def take_directions(take, hz, n=10, energy=None, verbose=True, first=None):
    """每次重复起转窗口的旋向（A、B 两台相机各判一次）。energy=(t,E) 可传缓存。"""
    import csv
    ramp, hold, post, period = schedule(hz)
    t, E = energy if energy is not None else motion_energy(take)
    if first is None: first = find_lead(t, E, period, ramp + hold + post)
    dur = t[-1]
    W = 3.0 if hz <= 100 else 5.0
    rows = []
    for k in range(n):
        s = first + k * period
        r = dict(k=k + 1, sched=round(s, 2))
        if s < 0.3:
            r.update(A=None, B=None, why='起转早于开始录像'); rows.append(r)
            if verbose: print(f"  {k+1:2d}  起转 {s:6.1f}s   起转没录到"); continue
        if s + ramp + hold + post > dur:
            r.update(A=None, B=None, why='录像不够长'); rows.append(r)
            if verbose: print(f"  {k+1:2d}  起转 {s:6.1f}s   录像不够长"); continue
        on, how = refine_onset(take, s)
        r['onset'] = round(on, 2)
        for cam in 'AB':
            res = sense_window(read_window(take, cam, max(0, on - 0.3), on + W))
            r[cam] = res.get('sense'); r[cam + '_turns'] = res.get('turns'); r[cam + '_same'] = res.get('same_frac')
            r[cam + '_curve'] = res.get('curve'); r[cam + '_span'] = (res.get('i0'), res.get('i1'))
            if res.get('sense') is None: r[cam + '_why'] = res.get('why')
        r['dir'] = verdict(r)
        rows.append(r)
        if verbose:
            print(f"  {k+1:2d}  → {str(r['dir']):>4}  起转 {on:6.1f}s (schedule {s:6.1f})   A {str(r['A']):>4} ({r.get('A_turns')} 圈, 同号 {r.get('A_same')})"
                  f"   B {str(r['B']):>4} ({r.get('B_turns')} 圈, 同号 {r.get('B_same')})"
                  + (f"   A: {r['A_why']}" if 'A_why' in r else '') + (f"   B: {r['B_why']}" if 'B_why' in r else ''), flush=True)
    return regrid(dict(take=take, hz=hz, first_start=first, period=period, rows=rows))


def stop_direction(take, d, pad=3.0):
    """全关后自由减速到停的那段，时间倒放 → 就是"从静止起转"，交给 sense_window，结果再反号。
    起转方向不一定是测量窗里的方向（斜坡中途失步重来会反向），停转方向才是 post 窗口里真实的旋向。"""
    ramp, hold, post, _ = schedule(d['hz'])
    for r in d['rows']:
        on = r.get('onset')
        if on is None: continue
        ta = on + ramp + hold + post
        for cam in 'AB':
            G = read_window(take, cam, ta - 0.3, ta + pad)
            if len(G) < 50: r[cam + '_stop'] = None; continue
            res = sense_window(G[::-1])
            v = res.get('sense'); r[cam + '_stop'] = {'CW': 'CCW', 'CCW': 'CW'}.get(v)
            r[cam + '_stop_turns'] = -res['turns'] if res.get('turns') is not None else None
        v = {r.get('A_stop'), r.get('B_stop')} - {None}
        r['dir_stop'] = v.pop() if len(v) == 1 else None
    return d


def mean_image(take, cam, t0, t1, stride=3):
    """窗口内每 stride 帧取一帧、裁到转子附近、1/4 分辨率的平均灰度图。"""
    G = read_window(take, cam, t0, t1)[::stride, 20:380, 100:560]
    if len(G) < 5: return None
    return np.mean([cv2.resize(g, (115, 90), interpolation=cv2.INTER_AREA) for g in G], 0).astype(np.float32)


def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def pose_scores(take, d):
    """构型检查，与旋向、碟沿跟踪都独立：每次重复在"保持段末"和"断电后末段"各取 1.5 s 平均图，
    与本条录像的参考图（旋向判为逆时针的那些次的逐像素中位数）做归一化互相关。
    正常构型 → 接近 1；静止挂着的 T 形、翻过去的转盘、整机乱晃的糊影 → 明显偏低。
    低频（20–30 Hz）叶片没糊成盘、跟踪器看不到椭圆，这个检查照样能用。"""
    ramp, hold, post, _ = schedule(d['hz'])
    imgs = {}
    for r in d['rows']:
        on = r.get('onset')
        if on is None: continue
        tc = on + ramp + hold
        for cam in 'AB':
            imgs[(r['k'], cam, 'hold')] = mean_image(take, cam, tc - 2.0, tc - 0.5)
            imgs[(r['k'], cam, 'post')] = mean_image(take, cam, tc + post - 3.0, tc + post - 0.5)
    good = [r['k'] for r in d['rows'] if r.get('dir') == 'CCW' and r.get('onset') is not None]
    ref_ks = good if len(good) >= 3 else [r['k'] for r in d['rows'] if r.get('onset') is not None]
    out = {}
    for cam in 'AB':
        for ph in ('hold', 'post'):
            stack = [imgs[(k, cam, ph)] for k in ref_ks if imgs.get((k, cam, ph)) is not None]
            if not stack: continue
            ref = np.median(stack, 0)
            for r in d['rows']:
                im = imgs.get((r['k'], cam, ph))
                if im is not None: out.setdefault(r['k'], {})[f'{cam}_{ph}'] = round(_ncc(im, ref), 3)
    for r in d['rows']:
        sc = out.get(r['k'], {})
        r['pose'] = sc
        r['pose_min'] = min(sc.values()) if sc else None
    return d


if __name__ == '__main__' and len(__import__('sys').argv) > 1:
    import sys, json, os
    take, hz = sys.argv[1], float(sys.argv[2])
    cache = sys.argv[3] if len(sys.argv) > 3 else None
    en = None
    if cache and os.path.exists(cache):
        d = np.load(cache); en = (d['t'], d['E'])
    print(f'=== {take} @ {hz:.0f} Hz')
    extra = sys.argv[4:]
    fo = float(extra[extra.index('--first') + 1]) if '--first' in extra else None
    if '--regrid' in extra:
        out = stop_direction(take, pose_scores(take, regrid(json.load(open(f'results/rim/{take}/spin_dir.json')))))
        print(f"  网格 t1={out['grid_t1']:.2f}s 残差中位 {out['grid_resid']:.2f}s")
        for r in out['rows']:
            print(f"  r{r['k']:<2d} 起转 {str(r.get('dir')):>4}  停转 {str(r.get('dir_stop')):>4}  构型 {r.get('pose_min')}  onset {r.get('onset')} (原 {r.get('onset_raw', '同')})")
        json.dump(out, open(f'results/rim/{take}/spin_dir.json', 'w'), indent=1, default=str); sys.exit()
    if fo is not None:
        out = stop_direction(take, pose_scores(take, take_directions(take, hz, energy=en, first=fo)))
        for r in out['rows']:
            print(f"  r{r['k']:<2d} 起转 {str(r.get('dir')):>4}  停转 {str(r.get('dir_stop')):>4}  构型 {r.get('pose_min')}")
        print(f"  网格残差中位 {out.get('grid_resid')}")
        json.dump(out, open(f'results/rim/{take}/spin_dir.json', 'w'), indent=1, default=str); sys.exit()
    if len(sys.argv) > 4 and sys.argv[4] == '--stop-only':
        out = stop_direction(take, json.load(open(f'results/rim/{take}/spin_dir.json')))
        for r in out['rows']:
            print(f"  r{r['k']:<2d} 起转 {str(r.get('dir')):>4}  停转 {str(r.get('dir_stop')):>4} "
                  f"(A {r.get('A_stop')} {r.get('A_stop_turns')}, B {r.get('B_stop')} {r.get('B_stop_turns')})  构型 {r.get('pose_min')}")
        json.dump(out, open(f'results/rim/{take}/spin_dir.json', 'w'), indent=1, default=str); sys.exit()
    if len(sys.argv) > 4 and sys.argv[4] == '--pose-only':
        out = pose_scores(take, json.load(open(f'results/rim/{take}/spin_dir.json')))
    else:
        out = pose_scores(take, take_directions(take, hz, energy=en))
    for r in out['rows']:
        print(f"  r{r['k']:<2d} {str(r.get('dir')):>4}  构型相关 最低 {r.get('pose_min')}  {r.get('pose')}")
    os.makedirs(f'results/rim/{take}', exist_ok=True)
    json.dump(out, open(f'results/rim/{take}/spin_dir.json', 'w'), indent=1, default=str)
    print(f"  第一次起转 {out['first_start']:.2f} s → results/rim/{take}/spin_dir.json")
