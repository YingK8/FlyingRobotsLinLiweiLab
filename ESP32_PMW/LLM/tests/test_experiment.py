"""Offline workflow regressions: geometry, stale evidence and hardware-free UI mocks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from controller.calib.rig import StereoRig, Camera
from LLM.record.io import write_json, sha256
from LLM.analyse.rig import fit_target, camera_key, localize_fixture, require_verified
from LLM.record.experiment import new_session, audit_take, check_trial, finish_take
from LLM.tests.test_capture_pose import _capture


CAM = dict(indices=(1,0), width=1280,height=800,fps=120,rotate180=True,backend='auto',codec='auto')


def synthetic(root):
    K = np.array([[900.,0,640],[0,900,400],[0,0,1]])
    B=np.eye(4); B[0,3]=60
    rig=StereoRig(cameras=(Camera(K,np.zeros(5),np.eye(4),'A'),Camera(K,np.zeros(5),B,'B')),
                  meta={'image_size':[1280,800],'rotate180':True})
    path=root/'rig.json'; rig.save(path)
    return rig,path


def pixels(rig, xyz):
    return [cv2.projectPoints((xyz-c.position)@c.R, np.zeros(3),np.zeros(3),c.K,c.dist)[0].reshape(-1,2) for c in rig.cameras]


class WorkflowTests(unittest.TestCase):
    def test_sync_check_checks_given_trial_and_reference_without_geometry(self):
        from LLM.record.experiment import prepare_sync_run, prepare_run, postprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            session=new_session(root/'sessions',dict(CAM,width=320,height=240,rotate180=False))
            capture=root/'capture';_capture(capture)
            meta=json.loads((capture/'meta.json').read_text())
            meta.update(session_id=session['session_id'],camera_config=session['camera'])
            write_json(capture/'meta.json',meta)
            trial=audit_take(capture,session)
            ref=root/'ref';ref.mkdir()
            for tag in 'AB': cv2.imwrite(str(ref/f'{tag}.png'),np.zeros((240,320,3),np.uint8))
            reference_meta={'session_id':session['session_id'],'camera':session['camera']}
            write_json(ref/'meta.json',reference_meta)
            kw=dict(note='synthetic sync test',warning_review='synthetic capture warnings reviewed')
            with self.assertRaises(ValueError):prepare_sync_run(root,session,{},ref,**kw)
            bare=prepare_sync_run(root,session)  # trial, reference and note are optional
            self.assertFalse((bare/'raw/static_reference').exists() or (bare/'raw/trial_audit.json').exists())
            run=prepare_sync_run(root,session,trial,ref,**kw)
            next_run=prepare_sync_run(root,session,trial,ref,**kw)
            self.assertNotEqual(run,next_run,'Repeating Block 8 must preserve the previous recording')
            manifest=json.loads((run/'manifest.json').read_text())
            self.assertEqual(manifest['geometry_status'],'not_assessed')
            self.assertFalse((run/'calibration/world_frame.json').exists())
            self.assertEqual(sha256(ref/'A.png'),sha256(run/'raw/static_reference/A.png'))
            with self.assertRaisesRegex(ValueError,'geometry'):
                postprocess(run,[0,0,2,2],[0,0,2,2],annotations="x.csv")
            with self.assertRaises(ValueError):
                prepare_run(root,session,root/'missing_rig.json',{}, {},trial,**kw)
            write_json(ref/'meta.json',{**reference_meta,'session_id':'stale'})
            with self.assertRaisesRegex(ValueError,'different'):
                prepare_sync_run(root,session,trial,ref,**kw)

    def test_fixed_rig_does_not_refit_changed_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            rig,path=synthetic(Path(tmp))
            obj=np.array([[x,y,0] for y in range(0,100,20) for x in range(0,100,20)],float)
            R=cv2.Rodrigues(np.array([.2,.1,0]))[0];xyz=obj@R.T+[0,0,500]
            uv=pixels(rig,xyz)
            T,result=fit_target(rig,obj,*uv)
            self.assertLess(result['rms_px'],1e-7)
            np.testing.assert_allclose(T[:3,3],[0,0,500],atol=1e-5)
            # Unchanged target images + shifted camera: fitting target cannot hide this.
            rig.b.T_world_cam[1,3]+=8
            _,bad=fit_target(rig,obj,*uv)
            self.assertGreater(bad['rms_px'],3)

    def test_new_charuco_bag_checks_without_recalibration(self):
        from LLM.analyse.rig import verify_bag
        from controller.calib.calibrate import CharucoSpec
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rig,path=synthetic(root)
            spec=CharucoSpec(9,6,16.52,12.39,'DICT_4X4_100','9x6_letter')
            data=json.loads(path.read_text());data.update(spec.summary());write_json(path,data)
            bag=root/'new_check';(bag/'A').mkdir(parents=True);(bag/'B').mkdir()
            write_json(bag/'meta.json',dict(camera_indices=[1,0],rotate180=True,board=spec.summary()))
            board=spec.board.generateImage((900,600))
            outline=np.array([[0,0,0],[148.68,0,0],[148.68,99.12,0],[0,99.12,0]])
            for i,angle in enumerate([-.3,-.15,0,.15,.3,.4]):
                R=cv2.Rodrigues(np.array([angle,.1,0]))[0]
                uv=pixels(rig,outline@R.T+[-60+i*8,-50,450+i*5])
                for tag,corners in zip('AB',uv):
                    H=cv2.getPerspectiveTransform(np.array([[0,0],[899,0],[899,599],[0,599]],np.float32),corners.astype(np.float32))
                    image=cv2.warpPerspective(board,H,(1280,800),borderValue=255)
                    cv2.imwrite(str(bag/tag/f'pair_{i:03d}.png'),image)
            before=sha256(path)
            report=verify_bag(bag,path,CAM,session_id='test')
            self.assertTrue(report['passed'],report)
            self.assertEqual(sha256(path),before)
            # Missing one paired image cannot be silently discarded to make a PASS.
            (bag/'B/pair_005.png').unlink()
            report=verify_bag(bag,path,CAM,session_id='test')
            self.assertFalse(report['passed'])

    def test_fixture_detects_rendered_ids_and_builds_gravity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); rig,path=synthetic(root); pair=root/'pair';pair.mkdir()
            c=camera_key(CAM)
            write_json(pair/'meta.json', {'session_id':'test','camera':c,'times':[1.,1.001]})
            layout={'dictionary':'DICT_APRILTAG_36h11','units':'mm','survey_confirmed':True,
                    'tag_black_edge_mm':25,'bearing_center_fixture_mm':[35,20,0],
                    'x_hint_fixture':[1,0,0],'markers':[]}
            images=[np.full((800,1280),255,np.uint8) for _ in range(2)]
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            for i,x in [(1,0),(2,60)]:
                obj=np.array([[x,0,0],[x+25,0,0],[x+25,25,0],[x,25,0]],float)
                layout['markers'].append({'id':i,'corners_fixture_mm':obj.tolist()})
                marker=cv2.aruco.generateImageMarker(dictionary,i,200)
                for im,uv in zip(images,pixels(rig,obj+[0,0,450])):
                    H=cv2.getPerspectiveTransform(np.array([[0,0],[199,0],[199,199],[0,199]],np.float32),uv.astype(np.float32))
                    warped=cv2.warpPerspective(marker,H,(1280,800),borderValue=255)
                    np.minimum(im,warped,out=im)
            for tag,im in zip('AB',images): cv2.imwrite(str(pair/f'{tag}.png'),im)
            layout_path=root/'layout.json';write_json(layout_path,layout)
            plumb={}
            for key,xyz in [('top',[100,-50,450]),('bottom',[100,50,450])]:
                uv=pixels(rig,np.array([xyz],float));plumb[key]={'a':uv[0][0].tolist(),'b':uv[1][0].tolist()}
            world=localize_fixture(pair,path,layout_path,plumb,c,session_id='test')
            np.testing.assert_allclose(world['origin_camera_mm'],[35,20,450],atol=3)
            np.testing.assert_allclose(np.array(world['R_camera_from_world'])[:,2],[0,-1,0],atol=1e-7)
            self.assertEqual(world['evidence']['tag_ids'],[1,2])
            layout['survey_confirmed']=False;write_json(layout_path,layout)
            with self.assertRaisesRegex(ValueError,'surveyed'):
                localize_fixture(pair,path,layout_path,plumb,c,session_id='test')

    def test_audit_binds_recorded_config_and_import_recomputes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session=new_session(root/'sessions',dict(CAM,width=320,height=240,rotate180=False))
            capture=root/'capture';_capture(capture)
            # Synthetic existing test captures are 320x240; use their actual metadata.
            meta=json.loads((capture/'meta.json').read_text())
            meta.update(session_id=session['session_id'],camera_config=session['camera'])
            write_json(capture/'meta.json',meta)
            report=audit_take(capture,session)
            check_trial(report,session,'synthetic timing only')
            different=copy.deepcopy(session);different['camera']['indices']=[0,1]
            with self.assertRaisesRegex(ValueError,'different'):
                audit_take(capture,different)
            with self.assertRaisesRegex(ValueError,'trial'):
                check_trial(report,different,'reviewed')
            meta['note']='changed after audit';write_json(capture/'meta.json',meta)
            with self.assertRaisesRegex(ValueError,'changed'):
                check_trial(report,session,'reviewed')
            # Fresh finish uses the new metadata and never receives the old report.
            fresh=finish_take(capture,session)
            self.assertNotEqual(fresh['input_sha256'],report['input_sha256'])

    def test_verification_binds_rig_session_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,path=synthetic(root)
            evidence=root/'frame.png';evidence.write_bytes(b'fixture')
            report=dict(kind='fixed_rig_verification',passed=True,session_id='s',camera=camera_key(CAM),
                        rig_sha256=sha256(path),input_sha256={str(evidence):sha256(evidence)})
            require_verified(report,path,CAM,'s')
            with self.assertRaises(ValueError): require_verified(report,path,CAM,'next_session')
            evidence.write_bytes(b'changed')
            with self.assertRaises(ValueError): require_verified(report,path,CAM,'s')

    def test_live_frequency_uses_reported_values_and_marks_stale_data(self):
        from LLM.record.panel import _FrequencyDisplay, _planned_phase
        display = _FrequencyDisplay()
        self.assertIn('等待 ESP', display.text(0))
        with patch('LLM.record.panel.time.monotonic', return_value=10):
            display.receive('t=1000 freq=0.0')
        self.assertIn('0.0 Hz', display.text(10))
        with patch('LLM.record.panel.time.monotonic', return_value=11):
            display.receive('t=2000 freq=37.5')
        display.receive('freq=150', sent=True)
        display.receive('!freq=163.00 rejected')
        display.receive('state=2')
        self.assertIn('37.5 Hz', display.text(12))
        self.assertIn('3.0s 未更新', display.text(14))
        planned = [(1,20,5,0,1),(20,150,15,0,1),(150,150,5,0,1)]
        self.assertEqual(_planned_phase(planned,5), '计划阶段 2：RAMP')
        self.assertEqual(_planned_phase(planned,20), '计划阶段 3：HOLD')
        self.assertEqual(_planned_phase(planned,25), '计划结束')

    def test_panel_stop_priority_and_exception_cleanup(self):
        from LLM.record.panel import run_panel
        for mode in ('manual', 'manual_failure', 'automatic', 'cancel_countdown',
                     'interrupt_countdown', 'interrupt_drive', 'stop_unconfirmed',
                     'stop_error', 'bright_led', 'no_ready'):
            events=[];handles={};clock=[0.]
            automatic = not mode.startswith('manual')
            start_times = []
            writers = []
            class Handle:
                def __init__(self): self.value=True;self.disabled=False
                def on_click(self,callback):self.callback=callback
            class Gui:
                def add_button(self,label,**kw):handles[label]=Handle();return handles[label]
                def add_image(self,*a,**kw):return Handle()
                def add_markdown(self,*a,**kw):return Handle()
                def add_checkbox(self,*a,**kw):return Handle()
            class Server:
                gui=Gui()
                def __init__(self,**kw):pass
                def get_port(self):return 8088
                def stop(self):events.append('server_closed')
            class Source:
                sources=[SimpleNamespace(actual={},n_grabbed=10,n_dropped=0)]*2
                last_skew=0.;last_stamps=(0.,0.)
                n=0
                def read(self):
                    self.n+=1;clock[0]+=.1;self.last_stamps=(clock[0],clock[0])
                    if not automatic and self.n==1:handles['Record'].callback(None)
                    if not automatic and self.n==5:handles['START ramp + LED'].callback(None)
                    if automatic and self.n==3:handles['START ramp + LED'].callback(None)
                    if mode=='cancel_countdown' and self.n==50:handles['Stop / finish'].callback(None)
                    if mode=='interrupt_countdown' and self.n==30:raise KeyboardInterrupt
                    if mode in ('interrupt_drive', 'stop_unconfirmed', 'stop_error') and self.n==55:raise KeyboardInterrupt
                    if not automatic and self.n==6:
                        if mode=='manual_failure':raise OSError('fake camera failure')
                        # STOP and START together must not permit another START.
                        handles['Stop / finish'].callback(None);handles['START ramp + LED'].callback(None)
                    return clock[0],[np.full((64,80),255 if mode=='bright_led' else 0,np.uint8)]*2
                def close(self):events.append('camera_closed')
            class Writer:
                errors=dropped=0;n=0
                def __init__(self,out,**kw):
                    self.dir=Path(out)/'take';self.meta=kw['meta'];events.append('record');writers.append(self)
                def add(self,*a):self.n+=1
                def close(self):events.append('writer_closed');return self.dir
            class Link:
                capture_first=None;capture_last=None;sync_sent=False
                state=None
                def __init__(self,*a,**kw):
                    events.append('link_open')
                    if mode=='no_ready':raise ValueError('no READY')
                def note_capture(self,t):
                    if self.capture_first is None:self.capture_first=t
                    self.capture_last=t
                def sync_can_start(self):return not self.sync_sent and self.capture_first is not None and self.capture_last-self.capture_first>=.25
                def drain(self):
                    self.on_line('t=1000 freq=37.5')
                def send_takeoff(self):
                    self.sync_sent=True;events.append('START');start_times.append(clock[0]-self.capture_first)
                def stop(self,timeout_s):
                    if timeout_s == 0:
                        events.append('timed_stop')
                        return False
                    events.append('stop')
                    if mode=='stop_error':raise OSError('fake serial disconnected')
                    return mode!='stop_unconfirmed'
                def send(self,cmd):events.append(cmd)
                def close(self):events.append('link_closed')
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);s=new_session(root/'sessions',dict(CAM,width=80,height=64,
                                                               exposure_s=None,auto_exposure=True,gain=100))
                run=root/'run';(run/'raw').mkdir(parents=True);write_json(run/'manifest.json',{'camera_session':s})
                pattern=root/'p.json';write_json(pattern,{'segments':[{'from_hz':2,'to_hz':50,'seconds':1,'shape':'linear'}]})
                with patch('builtins.print') as output, \
                     patch('LLM.record.panel._check_loopback'),patch('viser.ViserServer',Server),patch('LLM.record.panel.sources.open_stereo',return_value=Source()), \
                     patch('LLM.record.panel.FlightWriter',Writer),patch('LLM.record.panel.time.monotonic',side_effect=lambda:clock[0]), \
                     patch('controller.control.hover_controller_runner.CommandLink',Link), \
                     patch('controller.camera.uvc.configure',
                           side_effect=lambda exposure, gain, **kw: events.append(('exposure',exposure,kw))) as configure, \
                     patch('controller.control.hover_controller_runner._thermal_gate',
                           side_effect=AssertionError('LLM must not wait on thermal model')):
                    args=dict(output_root=root/'takes',max_seconds=10,run=run,ramp_path=pattern,
                              serial_port='FAKE',roi_a=[0,0,5,5],roi_b=[0,0,5,5])
                    args['auto_start']=automatic
                    if mode=='manual_failure':
                        with self.assertRaisesRegex(OSError,'fake'):run_panel(s,**args)
                    elif mode=='bright_led':
                        with self.assertRaisesRegex(RuntimeError,'START cancelled'):run_panel(s,**args)
                    elif mode=='no_ready':
                        with self.assertRaisesRegex(ValueError,'no READY'):run_panel(s,**args)
                    else:
                        result=run_panel(s,**args)
                        self.assertTrue(result['drive_stop_attempted'])
                        self.assertEqual(result['drive_stop_acknowledged'],mode not in ('stop_unconfirmed','stop_error'))
                        self.assertIsNotNone(result['take'])
                        self.assertEqual(writers[-1].meta['drive_stop_acknowledged'],result['drive_stop_acknowledged'])
                        if mode=='stop_error':self.assertIn('fake serial',result['drive_stop_error'])
                    configure.assert_called_once_with(None,100,auto_exposure=True)
                    if mode=='automatic':
                        frequency_lines = [str(c.args[0]) for c in output.call_args_list
                                           if c.args and 'ESP 回报频率' in str(c.args[0])]
                        self.assertTrue(any('RAMP' in line and '37.5 Hz' in line for line in frequency_lines))
                        self.assertTrue(any('STOP 已发送' in line for line in frequency_lines))
                        self.assertLess(len(frequency_lines),15)  # throttle output, not one line per frame
                self.assertEqual(events.count('START'),int(mode in ('manual','manual_failure','automatic',
                                                                   'interrupt_drive','stop_unconfirmed','stop_error')))
                if mode=='automatic':
                    self.assertGreaterEqual(start_times[0],5)
                    self.assertLess(start_times[0],5.11)
                    self.assertEqual(events.count('timed_stop'),1)
                if mode=='no_ready':
                    self.assertNotIn('record',events)
                    self.assertIn('camera_closed',events)
                    continue
                self.assertLess(events.index(('exposure',None,{'auto_exposure':True})),events.index('link_open'))
                self.assertLess(events.index('stop'),events.index('link_closed'))
                self.assertLess(events.index('stop'),events.index('writer_closed'))
                for e in ('link_closed','camera_closed','server_closed'):self.assertIn(e,events)


def run():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WorkflowTests))
    if not result.wasSuccessful():raise AssertionError('Experiment workflow regression failed')

if __name__=='__main__':run()
