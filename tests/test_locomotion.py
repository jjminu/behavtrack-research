from copy import deepcopy
from pathlib import Path
import threading
import cv2
import numpy as np
import pytest
from behavtrack import cage_locomotion as l
from behavtrack.locomotion_config import settings,calibration,requested
from test_cage_activity import make_video,fake

def config():
    return settings(dict(calibration=dict(points=[[0,0],[1,0],[1,1],[0,1]],width_cm=40,height_cm=30,planar_confirmed=True)))

def det(points):
    return dict(box_format='xyxy',boxes=[[x-3,y-3,x+3,y+3] for x,y in points],scores=[.99]*len(points))

def candidate(points,c=None):
    c=c or config();polygon,transform=l.geometry(c,(160,120),[0,0,1,1]);return l.candidates(det(points),len(points),c,polygon,transform)[0]

def test_metric_scale_and_order_independent_link():
    c=config();a=candidate([[20,30],[100,80]],c);b=candidate([[104,80],[24,30]],c)
    result,reason=l.link(a,b,.2,c,(160,120))
    assert reason=='valid'
    assert result['distance']==pytest.approx(2.) # 4px * .25cm/px * 2 animals
    assert result['mean_speed']==pytest.approx(5.)
    assert result['links'][0]['end']==[24,30]

def test_projective_floor_calibration_recovers_known_path():
    c=config();c['calibration']['points']=[[.1,.2],[.9,.1],[.8,.9],[.2,.8]]
    _,H=l.geometry(c,(160,120),[0,0,1,1]);ground=np.array([[[5.,8.],[8.,12.]]],np.float32)
    image=cv2.perspectiveTransform(ground,np.linalg.inv(H));restored=cv2.perspectiveTransform(image,H)
    assert np.linalg.norm(restored[0,1]-restored[0,0])==pytest.approx(5.,abs=1e-5)

@pytest.mark.parametrize('change',[{'width_cm':0},{'height_cm':float('nan')},{'planar_confirmed':False},{'points':[[0,0],[1,1],[1,0],[0,1]]}])
def test_bad_calibration_is_rejected(change):
    c=config()['calibration'];c.update(change)
    with pytest.raises(ValueError):calibration(c)

def test_overlap_missing_ambiguous_and_jump_are_unknown():
    c=config();polygon,H=l.geometry(c,(160,120),[0,0,1,1])
    assert l.candidates(det([[20,20]]),2,c,polygon,H)[1]=='count_mismatch'
    assert l.candidates(det([[20,20],[21,20]]),2,c,polygon,H)[1]=='overlap'
    a=candidate([[30,20],[50,20]],c);b=candidate([[40,10],[40,30]],c)
    assert l.link(a,b,.2,c,(160,120))[1]=='ambiguous_match'
    c['max_speed_cm_s']=1
    assert l.link(candidate([[20,30],[100,80]]),candidate([[24,30],[104,80]]),.2,c,(160,120))[1]=='implausible_step'

def test_unknown_time_never_becomes_zero_distance_or_full_time_speed():
    rows=[dict(start_s=59.9,end_s=60.1,duration_s=.2,status='valid',reason='valid',distance=4,mean_speed=10),
          dict(start_s=60.1,end_s=60.3,duration_s=.2,status='unknown',reason='overlap',distance=None,mean_speed=None)]
    s=l.summarize(rows,2)
    assert s['mean_speed']==pytest.approx(10)
    assert s['distance_per_animal']==2
    assert s['coverage_pct']==pytest.approx(50)
    bins=l.time_bins(rows,60,2)
    assert sum(b['distance_total'] or 0 for b in bins)==pytest.approx(4)
    assert sum(b['unknown_s'] for b in bins)==pytest.approx(.2)

class KnownPath:
    runtime={'name':'known synthetic coordinates'}
    def __init__(self,*args):pass
    def predict(self,frame,fid):
        if fid==10:return det([[30,30]]) # deliberately miss one animal
        p=[[20+fid,30],[100+fid,80]]
        return det(p[::-1] if fid%4 else p)
    def close(self):pass

def test_video_pipeline_known_distance_gaps_hashes_and_overlay(tmp_path):
    rec=make_video(tmp_path);hashes=l.read_json(rec.folder/'recording.json')['sha256'];c=config()
    result=l.analyze(rec.folder,dict(c,roi=[0,0,1,1],setup_id='fixed',comparison_block='unit'),Path('unused'),Path('unused'),predictor_factory=KnownPath)
    assert result['state']=='complete',result['error']
    report=l.read_report(result['path']);s=report['summary']
    # Ten adjacent 0.2s intervals. One missing sample invalidates its two edges.
    assert s['valid_pairs']==8 and s['valid_s']==pytest.approx(1.6)
    assert s['distance_total']==pytest.approx(8.)
    assert s['mean_speed']==pytest.approx(2.5)
    assert s['unknown_pairs']==2
    assert (Path(result['path'])/'tracking.mp4').stat().st_size>100
    assert all(l.digest(rec.folder/k)==v for k,v in hashes.items())
    (Path(result['path'])/'pairs.csv').write_text('changed')
    with pytest.raises(ValueError,match='변경'):l.read_report(result['path'])

def report(cage,water,value,source,unit='cm',coverage=100):
    r=fake(cage,water,0,source)
    r.update(unit=unit,method=l.VERSION)
    r['summary'].update(mean_speed=value,distance_per_animal=value*10,coverage_pct=coverage)
    return r

def test_group_water_comparison_cage_unit_and_exclusions(tmp_path):
    records=[report('C1','control',2,'one'),report('C1','control',4,'two'),report('C2','control',5,'three'),
             report('H1','hydrogen',7,'four'),report('P','oxygen',12,'pixels',unit='px'),report('G','oxygen',13,'gaps',coverage=79)]
    for mode in ('water','group'):
        r=l.comparison(records,mode);groups=r['strata'][0]['groups'];control=next(g for g in groups if g['label']=='control')
        assert control['n_cages']==2 and control['mean']==4
        assert control['sd']==pytest.approx(np.sqrt(2))
        assert len(r['excluded'])==2 and r['unit']=='cm/s' and r['p_value'] is None
    out=tmp_path/'compare';l.write_comparison(out,records)
    assert (out/'cages.csv').is_file() and (out/'report.html').is_file()

def test_cancel_does_not_load_model_or_mark_stationary(tmp_path):
    rec=make_video(tmp_path);cancel=threading.Event();cancel.set()
    r=l.analyze(rec.folder,dict(config(),roi=[0,0,1,1],setup_id='fixed',comparison_block='unit'),Path('none'),Path('none'),cancel,predictor_factory=KnownPath)
    assert r['state']=='cancelled' and r['report']['summary']['distance_total'] is None

def test_capture_automatic_settings_are_lightweight_and_bounded():
    assert requested({}) is None
    assert requested({'locomotion':config()})['calibration']['width_cm']==40
    with pytest.raises(ValueError):requested({'locomotion':True})


def test_white_body_centroid_video_distance_in_cm(tmp_path):
    from behavtrack import research as r
    c=config();spec=r.protocol(dict(group='합성 거리 검증',cage='virtual',animals=2,food_rect=[0,0,1,1],water_type='control',administration='bottle_only'),(160,120))
    rec=r.ResearchRecording(tmp_path,spec,0,10)
    for i in range(21):
        frame=np.full((120,160,3),30,np.uint8)
        for x,y in ((20+i,30),(100+i,80)):cv2.circle(frame,(x,y),7,(230,230,230),-1)
        rec.record(frame,rec.start_monotonic+i*.1)
    rec.finish()
    answer=l.analyze(rec.folder,dict(c,roi=[0,0,1,1],setup_id='synthetic',comparison_block='not_real_water_effect',detector='bright_foreground',white_threshold=140),Path('unused'),Path('unused'))
    assert answer['state']=='complete'
    s=answer['report']['summary'];assert s['coverage_pct']==100
    assert s['distance_total']==pytest.approx(10,abs=.03)
    assert s['mean_speed']==pytest.approx(2.5,abs=.01)


def test_api_requires_cm_calibration_or_explicit_pixel_preview(tmp_path):
    from behavtrack.research_web import Application
    app=Application(tmp_path);rec=make_video(app.results)
    try:
        with pytest.raises(ValueError,match='cm 단위'):
            app.locomotion_workflow('locomotion-analyze',dict(session=rec.folder.name,roi=[0,0,1,1]))
        assert app.locomotion_get('locomotion-sources',{})['sessions'][0]['started_utc']
        with pytest.raises(ValueError):app.locomotion_file({'session':rec.folder.name,'run':'../../outside','file':'locomotion.json'})
    finally:app.close()
