"""按天上色的段级散点诊断图 → results/rim/day_scatter.png。

回答 2026-09-21 的问题：把 cardan 刚性（全部锁死）的数据全混在一起，散布有多大？
每天的数据自己是不是集中的？段级闸与主图一致（Δθ≥5°、R2exp≥0.90）。
天的归属直接取段名前 10 位日期；09-16/17/21 全部为锁死万向节（note 的 without cardan = 锁住，用户澄清）。"""
import json, csv, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['PingFang SC']; plt.rcParams['axes.unicode_minus']=False

def segs(res, man, key, only=None, gate=0.90, dmin=5.0, dmax=30.0):
    if not (os.path.exists(res) and os.path.exists(man)): return []
    d=json.load(open(res)); HZ={r['take']:int(r['hz']) for r in csv.DictReader(open(man))}
    out=[]
    for r in d['takes']:
        hz=HZ.get(r['take'])
        if hz is None or (only and hz not in only): continue
        v=r.get(key); R=r.get('R2exp')
        if v is None or not np.isfinite(v) or v<=0: continue
        if not (dmin<=r.get('D',99)<=dmax): continue
        if not (R and R==R and R>=gate): continue
        out.append((hz, r['take'][:10], v))
    return out

HALF=[('l90_theta_lp.json','takes_l90.csv',None),('anchor_theta_lp.json','takes_anchor.csv',None),
      ('l120_theta_lp.json','takes_120locked.csv',None),('l130_theta_lp.json','takes_c_172835.csv',None),
      ('clock_theta_lp.json','takes_cardan_locked.csv',{140,150}),
      ('hr90b_theta_lp.json','takes_134136.csv',None),('hr100c_theta_lp.json','takes_142023.csv',None),
      ('hr110b_theta_lp.json','takes_223346.csv',None),('hr120b_theta_lp.json','takes_140829.csv',None),
      ('hr140b_theta_lp.json','takes_225528.csv',None),('hr120c_theta_lp.json','takes_151853.csv',None),
      ('hr130b_theta_lp.json','takes_145512.csv',None),('hr140c_theta_lp.json','takes_162123.csv',None),('hr150b_theta_lp.json','takes_165920.csv',None),('hr110d_theta_lp.json','takes_192932.csv',None)]
WHOLE=[('wrlock_theta_lp.json','takes_wr_locked.csv',{90}),('wr100_theta_lp.json','takes_wr100.csv',None),
       ('wr110_theta_lp.json','takes_c_195914.csv',None),('wr110b_theta_lp.json','takes_200235.csv',None),
       ('wr120_theta_lp.json','takes_201354.csv',None),('wr130_theta_lp.json','takes_202938.csv',None),
       ('wr140_theta_lp.json','takes_205228.csv',None),('wr150_theta_lp.json','takes_wr150.csv',None)]
DAY={'2026-09-16':('Sep 16 (cardan locked)','C0',-1.5),
     '2026-09-17':('Sep 17 (cardan locked)','C1',0.0),
     '2026-09-21':('Sep 21 (cardan locked)','C2',1.5)}

if __name__=='__main__':
    fig,axs=plt.subplots(2,2,figsize=(14,9))
    for row,(design,SRC) in enumerate([('half ring',HALF),('whole ring',WHOLE)]):
        for col,key in enumerate(['J_Y','J_exp']):
            ax=axs[row][col]; allpts=[]
            for res,man,only in SRC:
                allpts+=segs('results/rim/'+res,'ai/analysis/'+man,key,only)
            for day,(lab,c,off) in DAY.items():
                pts=[(h,v) for h,d,v in allpts if d==day]
                if not pts: continue
                hz=np.array([p[0] for p in pts],float); v=np.array([p[1] for p in pts])
                ax.plot(hz+off,v,'o',color=c,ms=4,alpha=.45)
                fs=sorted(set(hz))
                m=[v[hz==f].mean() for f in fs]; s=[v[hz==f].std(ddof=1) if (hz==f).sum()>1 else 0 for f in fs]
                ax.errorbar(np.array(fs)+off,m,s,fmt='_',color=c,ms=14,mew=2.5,capsize=4,lw=1.4,
                            label=lab if (row,col)==(0,0) or True else None)
            ax.set_title(f'{design} — {key}（点=单段，横杠=当天均值±SD，按天错位 ±1.5 Hz）',fontsize=10)
            ax.set_xlabel('f (Hz)'); ax.set_ylabel('J (1/s)'); ax.grid(alpha=.3)
            ax.set_xticks([90,100,110,120,130,140,150])
            h,l=ax.get_legend_handles_labels(); uniq=dict(zip(l,h))
            ax.legend(uniq.values(),uniq.keys(),fontsize=8,loc='upper left')
    fig.suptitle('刚性悬挂（cardan 一直锁死）全部段按天上色 —— 天内散布 vs 天间分家',fontsize=12)
    fig.tight_layout(); fig.savefig('results/rim/day_scatter.png',dpi=125)
    print('saved results/rim/day_scatter.png')
