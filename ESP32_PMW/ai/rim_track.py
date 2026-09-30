"""碟沿跟踪：径向边缘搜索 + RANSAC 椭圆。
每帧：用上一帧椭圆做预测 → 沿 90 个方向的外法线采样亮度剖面 → 取最外侧的“亮→暗”梯度极小（碟沿外缘，亚像素）
→ RANSAC 拟椭圆（轮毂/吊线/阴影边界当离群点）→ 内点去畸变后再拟一次得到 VIRTUAL_F 系的椭圆写 CSV。
输出到 scratchpad/rim/<take>/tilt_{A,B}.csv（列与原 tilt csv 兼容 + 原始像素系椭圆 + 内点数），然后 stereo_axis 得 axis.csv。"""
import sys, csv, math, time; sys.path.insert(0,'.')
import numpy as np, cv2
from pathlib import Path
from ai.alignment_rate import _min_plate, _load_rig, VIRTUAL_F, _trim_fit
S=Path(__file__).resolve().parent.parent / 'results'   # run_take 写到 S/'rim'/<take>/
N_ANG=90; STEP=0.5; G_MIN=10.0; THR_IN=1.0; RANSAC_IT=60
PHIS=np.linspace(0,2*np.pi,N_ANG,endpoint=False)

def ellipse_pts(e,phis):
    (cx,cy),(d1,d2),ang=e; a,b=d1/2,d2/2; th=math.radians(ang); c,s=math.cos(th),math.sin(th)
    x=cx+a*np.cos(phis)*c-b*np.sin(phis)*s; y=cy+a*np.cos(phis)*s+b*np.sin(phis)*c
    dx=-a*np.sin(phis)*c-b*np.cos(phis)*s; dy=-a*np.sin(phis)*s+b*np.cos(phis)*c
    nx,ny=dy,-dx; nrm=np.hypot(nx,ny); nx=nx/nrm; ny=ny/nrm
    flip=((x-cx)*nx+(y-cy)*ny)<0; nx[flip]*=-1; ny[flip]*=-1
    return x,y,nx,ny
def ell_dist(e,px,py):
    (cx,cy),(d1,d2),ang=e; a,b=d1/2,d2/2; th=math.radians(ang); c,s=math.cos(th),math.sin(th)
    u=(px-cx)*c+(py-cy)*s; v=-(px-cx)*s+(py-cy)*c; t=np.arctan2(v/b,u/a)
    return np.hypot(u-a*np.cos(t),v-b*np.sin(t))
def find_edges(gray,e,R):
    x,y,nx,ny=ellipse_pts(e,PHIS); s=np.arange(-R,R+STEP/2,STEP)
    X=(x[:,None]+nx[:,None]*s[None,:]).astype(np.float32); Y=(y[:,None]+ny[:,None]*s[None,:]).astype(np.float32)
    prof=cv2.remap(gray,X,Y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    prof=cv2.GaussianBlur(prof,(5,1),1.0)
    d=np.gradient(prof,STEP,axis=1)
    m=np.zeros(d.shape,bool); m[:,1:-1]=(d[:,1:-1]<d[:,:-2])&(d[:,1:-1]<=d[:,2:])&(d[:,1:-1]<-G_MIN)
    has=m.any(1); j=d.shape[1]-1-np.argmax(m[:,::-1],axis=1)
    rows=np.where(has)[0]; j=j[rows]
    # 抛物线亚像素
    dm=d[rows,j-1]; d0=d[rows,j]; dp=d[rows,j+1]; den=(dm-2*d0+dp); delta=np.where(np.abs(den)>1e-6,0.5*(dm-dp)/np.where(np.abs(den)>1e-6,den,1),0.0); delta=np.clip(delta,-1,1)
    sp=s[j]+delta*STEP
    return np.c_[x[rows]+nx[rows]*sp, y[rows]+ny[rows]*sp], has.sum()
def ransac_ellipse(P,rng):
    n=len(P)
    if n<10: return None,None
    best=None; bestin=None
    for _ in range(RANSAC_IT):
        idx=(rng.integers(0,n)+np.arange(5)*(n//5)+rng.integers(-max(1,n//12),max(1,n//12)+1,5))%n
        try: e=cv2.fitEllipse(P[idx].astype(np.float32))
        except cv2.error: continue
        if not np.all(np.isfinite(e[1])) or min(e[1])<8: continue
        inl=ell_dist(e,P[:,0],P[:,1])<THR_IN
        if bestin is None or inl.sum()>bestin.sum(): best,bestin=e,inl
    if bestin is None or bestin.sum()<10: return None,None
    for _ in range(2):
        try: e=cv2.fitEllipseDirect(P[bestin].astype(np.float32))
        except cv2.error: return None,None
        if not np.all(np.isfinite(e[1])): return None,None
        bestin=ell_dist(e,P[:,0],P[:,1])<THR_IN
        if bestin.sum()<10: return None,None
    return e,bestin
def init_from_threshold(gray_u8,plate):
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    mask=(cv2.subtract(gray_u8,plate)>40).astype(np.uint8); mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cnts: return None
    c=max(cnts,key=cv2.contourArea)
    if cv2.contourArea(c)<300 or len(c)<10: return None
    return _trim_fit(c.reshape(-1,1,2).astype(np.float32))

def track_video(mp4,K,dist,stamps,dump_times=None,t0=None,dump_prefix=None):
    plate=_min_plate(mp4); cap=cv2.VideoCapture(str(mp4)); rng=np.random.default_rng(0)
    rows=[]; e_prev=None; locked=False; i=0; n_reinit=0; dumps={}
    while True:
        ok,f=cap.read()
        if not ok: break
        g8=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if f.ndim==3 else f; g=g8.astype(np.float32)
        if e_prev is None:
            e_prev=init_from_threshold(g8,plate); n_reinit+=1
            if e_prev is None: i+=1; continue
            for _ in range(3):   # 从阈值初值向碟沿收敛
                P,_n=find_edges(g,e_prev,12.0); e,inl=ransac_ellipse(P,rng)
                if e is None: break
                e_prev=e
        R=6.0 if locked else 12.0
        P,nfound=find_edges(g,e_prev,R); e,inl=ransac_ellipse(P,rng)
        if e is None or inl.sum()<0.35*N_ANG:
            locked=False; e_prev=None
            if dump_times is not None: pass
            i+=1; continue
        locked=inl.sum()>=0.6*N_ANG; e_prev=e
        pts=P[inl].reshape(-1,1,2).astype(np.float32)
        pu=cv2.undistortPoints(pts,K,dist)*VIRTUAL_F
        try: eu=cv2.fitEllipseDirect(pu.reshape(-1,2).astype(np.float32))
        except cv2.error: i+=1; continue
        (cx,cy),(d1,d2),ang=eu; major,minor=max(d1,d2),min(d1,d2)
        if not (np.isfinite(major) and major>0): i+=1; continue
        theta=math.degrees(math.acos(min(1.0,minor/major)))
        rows.append((i,stamps.get(i,i),round(theta,3),int(inl.sum()),round(cx,3),round(cy,3),round(d1,3),round(d2,3),round(ang,3),
                     round(e[0][0],2),round(e[0][1],2),round(e[1][0],2),round(e[1][1],2),round(e[2],2)))
        if dump_times is not None and t0 is not None:
            tt=stamps.get(i,None)
            if tt is not None:
                for dt in dump_times:
                    if abs(tt-t0-dt)<0.0025 and dt not in dumps:
                        vis=f.copy()
                        for p in P: cv2.circle(vis,(int(round(p[0])),int(round(p[1]))),1,(0,0,255),-1)
                        for p in P[inl]: cv2.circle(vis,(int(round(p[0])),int(round(p[1]))),1,(0,255,0),-1)
                        cv2.ellipse(vis,e,(255,200,0),1)
                        r=min(e[1])/max(e[1]); cv2.putText(vis,f't={dt:.1f}s inl={inl.sum()} ratio={r:.2f}',(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                        dumps[dt]=vis
        i+=1
    cap.release()
    if dump_prefix and dumps:
        for dt,vis in dumps.items(): cv2.imwrite(f'{dump_prefix}_{dt:.0f}s.png',vis)
    return rows,i,n_reinit

def run_take(take,dump_times=None):
    rig=_load_rig(); src=Path(f'results/flights/{take}'); out=S/'rim'/take; out.mkdir(parents=True,exist_ok=True)
    stamps={}
    for row in csv.DictReader(open(src/'frames.csv')):
        stamps[int(row['index'])]={c:float(row.get(f't_{c.lower()}') or row['t_capture']) for c in 'AB'}
    t0=min(v['A'] for v in stamps.values())
    log=[]
    for cam in 'AB':
        K,dist,_=rig[cam]; tic=time.time()
        rows,n,nre=track_video(src/cam/f'{cam}.mp4',K,dist,{k:v[cam] for k,v in stamps.items()},dump_times,t0,str(out/f'overlay_{cam}') if dump_times else None)
        with open(out/f'tilt_{cam}.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['frame','t','theta_deg','area_px','cx','cy','d1','d2','ang_deg','rcx','rcy','rd1','rd2','rang']); w.writerows(rows)
        log.append(f'{cam}: {len(rows)}/{n} 帧 ({time.time()-tic:.0f}s, 重初始化 {nre} 次, 内点中位 {np.median([r[3] for r in rows]):.0f}/{N_ANG})')
    from ai.alignment_rate import stereo_axis
    if (out/'axis.csv').exists(): (out/'axis.csv').unlink()
    p=stereo_axis(out)
    ag=np.array([float(r['agree_deg']) for r in csv.DictReader(open(p))]); t=np.array([float(r['t']) for r in csv.DictReader(open(p))]); t-=t[0]
    log.append(f'axis: 一致性中位 全程 {np.median(ag):.1f}°  10-16s {np.median(ag[(t>10)&(t<16)]):.1f}°  20-28s {np.median(ag[(t>20)&(t<28)]):.1f}°  过15°比例 后段 {(ag[(t>20)&(t<28)]<=15).mean():.0%}')
    return take,log
if __name__=='__main__':
    take=sys.argv[1]; dt=[float(x) for x in sys.argv[2:]] or None
    t,log=run_take(take,dt); print(t); print('\n'.join(log))
