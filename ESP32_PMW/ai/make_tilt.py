"""生成多次重复的 tilt.json：一条录像里把单条 take 的流程完整重复 N 次，每次之间所有线圈归零让转子停下。

每一次重复：
  全部线圈开（carrier 100）→ EASE 斜坡 1→f → 保持 → 断 A+C（channels [0,2] carrier 0）→ 等 post
  → 全部线圈归零（activateChannels mask 15 value 0）→ 频率在断电状态下静默降回 1 Hz → 等满 off
静默降频不通电（carrier 为 0），只是让下一次起转和第一次一样从 1 Hz 开始，而不是在 f 上突然通电。

用法（在 ESP32_PMW 下）：
  uv run python ai/make_tilt.py --hz 40 --n 4            # 写 spiffs_data/tilt.json
  uv run python ai/make_tilt.py --hz 120 --n 10 whole    # whole ring：停机时间 ×1.5（22.5 s），>100 Hz 斜坡缺省 15 s
  uv run python ai/make_tilt.py --hz 40 --n 4 --stdout   # 只打印，不写文件
然后照常 pio run -e tilt -t uploadfs（先关线圈或按 GPIO14，uploadfs 会复位板子并立刻开跑）。"""
import argparse, json, sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('design',nargs='?',default='half',choices=['whole','half','no'],
                help='whole：每次拍完后的停机时间（降频+全关）×1.5，且 >100 Hz 斜坡缺省 15000（10 s 斜坡约四成起转失败）；half/no 不变（缺省 half）')
ap.add_argument('--hz',type=float,required=True); ap.add_argument('--n',type=int,default=4)
ap.add_argument('--ramp-ms',type=int,default=None,help='缺省：≤100 Hz 用 5000，更高用 10000（2026-09-16 起；whole 时更高用 15000）')
ap.add_argument('--hold-ms',type=int,default=None,help='缺省：≤100 Hz 用 5000，更高用 3000（2026-09-16 起；之前一律 5000）'); ap.add_argument('--post-ms',type=int,default=10000)
ap.add_argument('--off-ms',type=int,default=14000,help='全部线圈归零后的等待时长；与 reset-ms 合计为两次之间的停机时间（缺省 1+14=15 s；whole 时合计 ×1.5=22.5 s）')
ap.add_argument('--reset-ms',type=int,default=1000,help='断电状态下频率 f→1 Hz 的时长，排在 off 之前')
ap.add_argument('--out',default='spiffs_data/tilt.json'); ap.add_argument('--stdout',action='store_true')
a=ap.parse_args()
f=a.hz; ramp=a.ramp_ms if a.ramp_ms is not None else (5000 if f<=100 else (15000 if a.design=='whole' else 10000))
hold=a.hold_ms if a.hold_ms is not None else (5000 if f<=100 else 3000)
off=a.off_ms
if a.design=='whole': off=round((a.reset_ms+a.off_ms)*1.5)-a.reset_ms   # 总停机 ×1.5；降频段仍 1 s，加长的全在全关等待里
sch=[]
for k in range(1,a.n+1):
    sch+= [{"method":"label","value":f"RUN_{k}_SPINUP_{f:g}HZ"},
           {"method":"activateChannels","mask":15,"value":100.0},
           {"method":"addEaseRampTask","from":1.0,"to":f,"duration_ms":ramp},
           {"method":"label","value":f"RUN_{k}_HOLD"},
           {"method":"addWaitTask","duration_ms":hold},
           {"method":"label","value":f"RUN_{k}_CUT_A_C"},
           {"method":"addCarrierDutyCycleTask","channels":[0,2],"value":0.0},
           {"method":"addWaitTask","duration_ms":a.post_ms},
           {"method":"label","value":f"RUN_{k}_ALL_OFF"},
           {"method":"activateChannels","mask":15,"value":0.0},
           {"method":"addEaseRampTask","from":f,"to":1.0,"duration_ms":a.reset_ms},
           {"method":"addWaitTask","duration_ms":off}]
doc={"resolution_ms":25,"initial_freq":0.0,"initial_duty":[50,50,50,50],"direction":"CW","schedule":sch}
txt=json.dumps(doc,indent=2,ensure_ascii=False)
period=(ramp+hold+a.post_ms+a.reset_ms+off)/1000; drive=(ramp+hold+a.post_ms)/1000
print(f'# [{a.design}] {f:g} Hz × {a.n} 次，每次 {period:.1f} s（斜坡 {ramp/1000:g} + 保持 {hold/1000:g} + 断 A+C {a.post_ms/1000:g} + 降频 {a.reset_ms/1000:g} + 全关 {off/1000:g}）',file=sys.stderr)
print('# 断 A+C 的时刻（EN 之后）：'+', '.join(f'{(ramp+hold)/1000+(k-1)*period:.0f} s' for k in range(1,a.n+1))+f'；全部结束 {a.n*period:.0f} s',file=sys.stderr)
print(f'# 每次之间线圈全部为 0 共 {(a.reset_ms+off)/1000:g} s（其中降频 {a.reset_ms/1000:g} s 也不通电）',file=sys.stderr)
print(f'# 通电合计 {drive*a.n:.0f} s',file=sys.stderr)
if a.stdout: print(txt)
else: Path(a.out).write_text(txt+'\n'); print(f'# 已写 {a.out}',file=sys.stderr)
