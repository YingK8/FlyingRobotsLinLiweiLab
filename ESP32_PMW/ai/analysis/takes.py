"""take 清单与共用路径。新 take 用 process_take.py 登记，不要再在脚本里写死列表。"""
from pathlib import Path
import csv, os
INCLUDE_OLD=os.environ.get('INCLUDE_OLD','0')=='1'   # 缺省只用当前台位（tag=new）；INCLUDE_OLD=1 才把 09-08 的 old 带上
ROOT=Path(__file__).resolve().parents[2]          # ESP32_PMW
RIM=ROOT/'results'/'rim'                           # 每条 take 一个子目录：tilt_A/B.csv, axis.csv, axis_b3.csv
MANIFEST=Path(os.environ.get('TAKES_MANIFEST') or Path(__file__).with_name('takes.csv'))   # take,hz,tag；TAKES_MANIFEST 可换成别的清单（如 takes_halfring.csv）
AXIS_FILE='axis.csv'                               # 改成 'axis_b3.csv' 即用 3° 厚度校正后的轴
def load_takes():
    rows=[]
    if not MANIFEST.exists(): return rows
    for r in csv.DictReader(open(MANIFEST)):
        tk=(r.get('take') or '').strip()
        if not tk or tk.startswith('#'): continue
        rows.append((tk,int(round(float(r['hz']))),(r.get('tag') or 'new').strip() or 'new'))
    return rows
def add_take(take,hz,tag='new'):
    rows=load_takes()
    if any(t==take for t,_,_ in rows): return False
    new=not MANIFEST.exists()
    with open(MANIFEST,'a',newline='') as f:
        w=csv.writer(f)
        if new: w.writerow(['take','hz','tag'])
        w.writerow([take,hz,tag])
    return True
TAKES=[r for r in load_takes() if r[2]!='skip' and (INCLUDE_OLD or r[2]!='old')]   # tag=skip 的永远不进分析
HAS_OLD=any(tag=='old' for _,_,tag in TAKES)
HZS=sorted({hz for _,hz,tag in TAKES if tag=='new'})
