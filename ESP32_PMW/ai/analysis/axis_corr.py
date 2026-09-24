"""b=3.0° 的碟沿厚度校正：rim/<take>/{tilt_A,tilt_B}.csv → axis_b3.csv（格式同 axis.csv）。
单条：correct_take(take)；直接运行则对 takes.csv 里全部 take 重建。"""
import sys, csv, math; sys.path.insert(0,'.')
import numpy as np
from ai.alignment_rate import _load_rig, VIRTUAL_F
from controller.pose.conic import backproject_ellipse
from ai.analysis.takes import RIM, TAKES
B_DEG=3.0
def rot_away(n,d,b):
    if n@d<0: n=-n
    ax=np.cross(n,d); na=np.linalg.norm(ax)
    if na<1e-9: return n
    ax/=na; k=-b; c,s=math.cos(k),math.sin(k); return n*c+np.cross(ax,n)*s+ax*(ax@n)*(1-c)
def correct_take(take,b_deg=B_DEG,verbose=True):
    b=math.radians(b_deg); rig=_load_rig(); Kv=np.diag([VIRTUAL_F,VIRTUAL_F,1.0]); RB,tB=rig['B'][2][:3,:3],rig['B'][2][:3,3]
    d=RIM/take; A={int(r['frame']):r for r in csv.DictReader(open(d/'tilt_A.csv'))}; B={int(r['frame']):r for r in csv.DictReader(open(d/'tilt_B.csv'))}
    raw=[]
    for i in sorted(set(A)&set(B)):
        try:
            ea=((float(A[i]['cx']),float(A[i]['cy'])),(float(A[i]['d1']),float(A[i]['d2'])),float(A[i]['ang_deg']))
            eb=((float(B[i]['cx']),float(B[i]['cy'])),(float(B[i]['d1']),float(B[i]['d2'])),float(B[i]['ang_deg']))
            ca=[np.asarray(p.normal) for p in backproject_ellipse(ea,Kv,10.0)]; cb=[RB@np.asarray(p.normal) for p in backproject_ellipse(eb,Kv,10.0)]
        except Exception: continue
        pa=np.array([float(A[i]['cx'])/VIRTUAL_F,float(A[i]['cy'])/VIRTUAL_F,1.0]); pa/=np.linalg.norm(pa); P=pa*150.0
        dA=-P/np.linalg.norm(P); dB=tB-P; dB/=np.linalg.norm(dB)
        ca=[rot_away(n,dA,b) for n in ca]; cb=[rot_away(n,dB,b) for n in cb]
        best=None
        for na in ca:
            for nb in cb:
                sc=abs(na@nb)
                if best is None or sc>best[0]: best=(sc,na,nb if na@nb>=0 else -nb)
        sc,na,nb=best; n=na+nb; n/=np.linalg.norm(n); raw.append((i,float(A[i]['t']),n,math.degrees(math.acos(min(1.0,sc)))))
    anchor=min(raw,key=lambda r:r[3])[2]
    with open(d/'axis_b3.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['frame','t','nx','ny','nz','agree_deg'])
        w.writerows([(i,t,*(n if n@anchor>=0 else -n).round(5),round(ag,2)) for i,t,n,ag in raw])
    ag=np.array([r[3] for r in raw])
    if verbose: print(f'{take}: {len(raw)} 帧  b={b_deg}° 后两相机一致性中位 {np.median(ag):.2f}°  p90 {np.percentile(ag,90):.2f}°')
    return float(np.median(ag))
if __name__=='__main__':
    for take,hz,tag in TAKES: correct_take(take)
