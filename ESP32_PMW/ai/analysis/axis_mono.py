"""单相机轴向量：rim/<take>/tilt_<cam>.csv 的椭圆 → 反投影 → 两个候选法向里挑对的那个 → axis_mono<cam>.csv（格式同 axis.csv）。

为什么需要：三叶螺旋桨（no ring，2026-09-14）正常构型下扫出的盘对 A 相机几乎是侧视，
断电后更侧，A 的跟踪覆盖率断电前 23–80%、断电后 ~0%；B 是 ~100%。立体轴在测量窗里根本没有数据。

二义性：一个椭圆对应两个圆盘朝向，关于视线镜像。B 斜着看（椭圆长短轴比 ~0.5），两候选相差很大，
所以拿一个参考方向（立体轴在"A 还看得见"的那些帧里的中位数）挑夹角小的那个就行；
每帧都报两候选的夹角 sep_deg，太小的帧（接近正视，二义性分不开）直接丢。
厚度/桨距带来的朝相机偏置近似常数，θ 相对沉降后的轴计算时基本抵消 —— 用 09-11 半环（立体好用）逐段对照验证过才用。

agree_deg 列写 0（theta_lp / segments 用它做 ≤10° 的门限，单相机没有两视图一致性可言）。
用法：uv run python ai/analysis/axis_mono.py <take> [--cam B] [--windows spin]
"""
import sys, csv, math, json, argparse
sys.path.insert(0, '.')
import numpy as np
from ai.alignment_rate import _load_rig, VIRTUAL_F
from controller.pose.conic import backproject_ellipse
from ai.analysis.takes import RIM

SEP_MIN = 25.0   # 两候选夹角小于它就不要这一帧


def stereo_ref(take, windows=None, gate=10.0):
    rows = list(csv.DictReader(open(RIM / take / 'axis.csv')))
    t = np.array([float(r['t']) for r in rows]); ag = np.array([float(r['agree_deg']) for r in rows])
    n = np.array([[float(r['nx']), float(r['ny']), float(r['nz'])] for r in rows])
    m = ag <= gate
    if windows:
        w = np.zeros(len(t), bool)
        for a, b in windows: w |= (t >= a) & (t < b)
        if (m & w).sum() >= 50: m &= w
    ref = np.median(n[m], 0); return ref / np.linalg.norm(ref), int(m.sum())


def mono_take(take, cam='B', windows=None, out=None):
    rig = _load_rig(); Kv = np.diag([VIRTUAL_F, VIRTUAL_F, 1.0])
    R = rig[cam][2][:3, :3]                      # 相机 → A 系（A 为单位阵）
    ref, nref = stereo_ref(take, windows)
    rows = []; seps = []
    for r in csv.DictReader(open(RIM / take / f'tilt_{cam}.csv')):
        e = ((float(r['cx']), float(r['cy'])), (float(r['d1']), float(r['d2'])), float(r['ang_deg']))
        try: cands = [R @ np.asarray(p.normal, float) for p in backproject_ellipse(e, Kv, 10.0)]
        except Exception: continue
        if len(cands) < 2: continue
        c0, c1 = cands[0] / np.linalg.norm(cands[0]), cands[1] / np.linalg.norm(cands[1])
        sep = math.degrees(math.acos(min(1.0, abs(float(c0 @ c1))))); seps.append(sep)
        if sep < SEP_MIN: continue
        n = c0 if abs(c0 @ ref) >= abs(c1 @ ref) else c1
        if n @ ref < 0: n = -n
        rows.append((r['frame'], r['t'], *np.round(n, 5), 0.0, round(sep, 1)))
    out = out or RIM / take / f'axis_mono{cam}.csv'
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['frame', 't', 'nx', 'ny', 'nz', 'agree_deg', 'sep_deg']); w.writerows(rows)
    seps = np.array(seps)
    print(f'{take}: 单相机 {cam} {len(rows)} 帧（参考方向来自 {nref} 个立体帧）  两候选夹角 中位 {np.median(seps):.0f}°  <{SEP_MIN:.0f}° 丢 {(seps < SEP_MIN).mean():.1%}')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('take'); ap.add_argument('--cam', default='B')
    ap.add_argument('--windows', default=None, help="'spin'：参考方向只用 spin_dir.json 里正常次的断电前窗口")
    a = ap.parse_args(); win = None
    if a.windows == 'spin':
        d = json.load(open(RIM / a.take / 'spin_dir.json'))
        from ai.analysis.spin_dir import schedule
        ramp, hold, post, _ = schedule(d['hz'])
        f0 = float(next(csv.DictReader(open(f'results/flights/{a.take}/frames.csv')))['t_capture'])
        win = [(f0 + r['onset'] + ramp + hold - 4, f0 + r['onset'] + ramp + hold) for r in d['rows']
               if r.get('dir_stop') == 'CCW' and (r.get('pose_min') or 0) >= 0.9]
    mono_take(a.take, a.cam, win)
