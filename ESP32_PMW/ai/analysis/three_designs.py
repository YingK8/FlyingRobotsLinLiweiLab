"""三个构型（whole/half/no ring）的 J(f) 对比主图 → results/rim/three_designs_locked.png。

实线 = 锁死/锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）（可横向比较的当前台面），虚线 = 自由 cardan 旧数据（仅参考，见
feedback-cardan-comparability）。两个面板：J_Y 与 J_exp（缺一不可，见 feedback-always-show-jexp）。
段级规则与汇总一致：Δθ≥5° 下限 + R2exp≥0.90 固定闸；不用整条 take 的自适应 R² 阈值（太干净时会误杀）。

数据源在 SOURCES 里登记：每条 = (theta_lp 的 JSON, 清单 CSV, collect 的额外参数)。
新增频率点：跑完 theta_lp 后在这里加一行，再 uv run python ai/analysis/three_designs.py。
2026-09-17：whole 110 换成当日复测（wr110b，n7，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）），120 新增（wr120）；
昨日 195914 的 n3 高值（J_exp 0.98）画成空心点仅供对照，不进实线。"""
import json, csv, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['PingFang SC']; plt.rcParams['axes.unicode_minus']=False

def collect(res, man, key, r2key, tag=None, reject=True, gate=0.0, dmin=5.0, lo=90, hi=150, only=None, dmax=30.0):
    # lo=90：用户 09-21 要求恢复从 90 Hz 画起（09-17 曾要求 100 起）
    if not (os.path.exists(res) and os.path.exists(man)): return {}
    d=json.load(open(res)); thr=d['thr']; HZ={r['take']:(int(r['hz']),r['tag']) for r in csv.DictReader(open(man))}
    by={}
    for r in d['takes']:
        h=HZ.get(r['take'])
        if not h or (tag and h[1]!=tag): continue
        if only and h[0] not in only: continue
        v=r.get(key)
        if v is None or not np.isfinite(v) or v<=0 or not (lo<=h[0]<=hi): continue
        if not (dmin<=r.get('D',99)<=dmax): continue
        if gate>0 and not (r.get('R2exp') and r['R2exp']==r['R2exp'] and r['R2exp']>=gate): continue
        if reject and (r.get(r2key) is None or r[r2key]<thr[key]): continue
        by.setdefault(h[0],[]).append(v)
    return by

def mad_keep(v,k=3.0):
    # 每频率的段值做对称离群隔离（中位数 ±k·MAD，k=3 统计学常规值；用户 2026-09-22 要求）。
    # 对三条线全频率统一适用；均值/SEM/浅色散点都只画保留下来的段。
    v=np.asarray(v,float); med=np.median(v); mad=np.median(abs(v-med))*1.4826
    return list(v[abs(v-med)<=k*mad]) if mad>0 else list(v)

def merge(ds):
    out={}
    for by in ds:
        for f,v in by.items(): out.setdefault(f,[]).extend(v)
    return out

HALF=[  # 统一规则（09-21 定）：每个频率合并两天所有健康 take；只剔除按 take 级规则作废的
        # （174408 双簇+拟合贴地、135702/174307 段间下滑=断翼过程、192932=另一个 robot）
      ('results/rim/l90_theta_lp.json','ai/analysis/takes_l90.csv',{}),            # 90 09-16
      ('results/rim/hr90b_theta_lp.json','ai/analysis/takes_134136.csv',{}),       # 90 09-21
      ('results/rim/anchor_theta_lp.json','ai/analysis/takes_anchor.csv',{'only':{100}}),  # 100 09-16
      ('results/rim/hr100c_theta_lp.json','ai/analysis/takes_142023.csv',{}),      # 100 09-21
      ('results/rim/hr110b_theta_lp.json','ai/analysis/takes_223346.csv',{}),      # 110 09-17（09-16 的 174408 按质量规则作废）
      ('results/rim/l120_theta_lp.json','ai/analysis/takes_120locked.csv',{}),     # 120 09-16
      ('results/rim/hr120b_theta_lp.json','ai/analysis/takes_140829.csv',{}),      # 120 09-21
      ('results/rim/hr120c_theta_lp.json','ai/analysis/takes_151853.csv',{}),      # 120 09-21
      ('results/rim/l130_theta_lp.json','ai/analysis/takes_c_172835.csv',{}),      # 130 09-16
      ('results/rim/hr130b_theta_lp.json','ai/analysis/takes_145512.csv',{}),      # 130 09-21
      ('results/rim/clock_theta_lp.json','ai/analysis/takes_cardan_locked.csv',{'only':{140,150}}),  # 140/150 09-16
      ('results/rim/hr140b_theta_lp.json','ai/analysis/takes_225528.csv',{}),      # 140 09-17
      ('results/rim/hr140c_theta_lp.json','ai/analysis/takes_162123.csv',{}),      # 140 09-21
      ('results/rim/hr150b_theta_lp.json','ai/analysis/takes_165920.csv',{})]      # 150 09-21
WHOLE=[('results/rim/wrlock_theta_lp.json','ai/analysis/takes_wr_locked.csv',{'only':{90}}),
       ('results/rim/wr100_theta_lp.json','ai/analysis/takes_wr100.csv',{}),
       ('results/rim/wr110b_theta_lp.json','ai/analysis/takes_200235.csv',{}),   # 09-17 复测，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）
       ('results/rim/wr120_theta_lp.json','ai/analysis/takes_201354.csv',{}),    # 09-17，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）
       ('results/rim/wr130_theta_lp.json','ai/analysis/takes_202938.csv',{}),    # 09-17，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）
       ('results/rim/wr140_theta_lp.json','ai/analysis/takes_205228.csv',{}),    # 09-17，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）
       ('results/rim/wr150_theta_lp.json','ai/analysis/takes_wr150.csv',{})]     # 09-17，锁死（note 写 without cardan = 锁住，非拆除，用户 09-21 澄清）（212343+212724 两条合并）
# 09-17 晚用户裁掉两个系列：09-09 自由 whole 参考线（浅蓝）和"whole 110 被取代"空心对照点。
# 数据都还在：results/rim/theta_lp.json + takes.csv（tag=new）和 wr110_theta_lp.json + takes_c_195914.csv。

if __name__=='__main__':
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.6)); RATIO={}
    for ax,key,r2key,tt in [(axs[0],'J_Y','R2Y_raw','J_Y'),(axs[1],'J_exp','R2exp','J_exp = 1/τ')]:
        half=merge([collect(r,m,key,r2key,reject=False,gate=0.90,**kw) for r,m,kw in HALF])
        whole=merge([collect(r,m,key,r2key,reject=False,gate=0.90,**kw) for r,m,kw in WHOLE])
        nor=collect('results/rim/noring0914_theta_lp.json','ai/analysis/takes_noring0914.csv',key,r2key,reject=False)
        half={f:mad_keep(v) for f,v in half.items()}; whole={f:mad_keep(v) for f,v in whole.items()}; nor={f:mad_keep(v) for f,v in nor.items()}
        # 用户要求：点正好落在整数频率刻度上（off 全 0）；图内文字全英文（均 2026-09-17）
        MM=[]   # 各系列均值，决定 y 轴范围
        for lab,by,c,mk,off,ms,ls,al in [
            ('whole ring, cardan locked (Sep 16-17)',whole,'navy','P',0,11,'-',1.0),
            ('half ring, cardan locked (Sep 16-21)',half,'C1','*',0,15,'-',1.0),
            ('no ring, free cardan (Sep 14, ref.)',nor,'C3','o',0,7,'--',0.75)]:
            if not by: continue
            fs=np.array(sorted(by),float); m=np.array([np.mean(by[f]) for f in fs])
            sem=np.array([np.std(by[f],ddof=1)/np.sqrt(len(by[f])) if len(by[f])>1 else np.nan for f in fs])
            for f in fs: ax.plot([f+off]*len(by[f]),by[f],mk,color=c,alpha=.12,ms=4)
            ax.errorbar(fs+off,m,sem,fmt=mk+ls,color=c,ms=ms,lw=1.8,capsize=4,label=lab,alpha=al)
            MM+=list(m)
            print(f'[{key}] {lab}: '+'  '.join(f'{int(f)}:{mm:.2f}(n{len(by[f])})' for f,mm in zip(fs,m)))
        if key=='J_exp':
            RATIO={f:np.mean(half[f])/np.mean(whole[f]) for f in sorted(set(half)&set(whole))}
        ax.set_xlabel('drive frequency f (Hz)'); ax.set_ylabel('J (1/s)')
        ax.set_title(tt+r' ($\Delta\theta\geq5°$; $R^2_{exp}\geq0.9$; 3·MAD outlier-isolated)')
        # y 轴按均值动态收紧（用户 09-17：三条线的分差要拉大些）；顶端封 2.35
        ax.set_xlim(85,156); ax.set_ylim(max(0,0.88*min(MM)),min(1.06*max(MM),2.35))
        ax.set_xticks([90,100,110,120,130,140,150]); ax.grid(alpha=.3)
        ax.legend(fontsize=8,loc='upper left')
    rt='  '.join(f'{f}:{v:.2f}x' for f,v in RATIO.items())
    fig.suptitle('Three designs, 90-150 Hz - solid = cardan locked; dashed = free cardan (reference); half/whole (J_exp) = '+rt,fontsize=11.5)
    fig.tight_layout(); fig.savefig('results/rim/three_designs_locked.png',dpi=130)
    print('saved results/rim/three_designs_locked.png')
