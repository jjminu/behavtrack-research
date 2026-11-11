"""Bound locomotion jobs, artifacts and cage comparisons."""
from pathlib import Path
import re
import uuid
from .integrity import read_json

class LocomotionWorkflow:
    def locomotion_run(self,session,name):
        if not isinstance(name,str) or not re.fullmatch(r'loco_[a-f0-9]{12}',name):raise ValueError('이동거리 분석 결과를 선택하세요.')
        source=self.activity_source(session);parent=(source/'locomotion').resolve();run=(parent/name).resolve()
        if parent.parent!=source or run.parent!=parent or not run.is_dir():raise ValueError('추적 결과 폴더를 확인하세요.')
        return run

    def locomotion_file(self,q):
        name=q.get('file','')
        if q.get('comparison'):
            ident=q['comparison']
            if not re.fullmatch(r'compare_[a-f0-9]{12}',ident):raise ValueError('이동성 비교 결과를 선택하세요.')
            parent=(self.results/'locomotion_comparisons').resolve();folder=(parent/ident).resolve()
            if folder.parent!=parent or name not in ('comparison.json','cages.csv','report.html'):raise ValueError('지원하지 않는 이동성 비교 파일입니다.')
        else:
            folder=self.locomotion_run(q['session'],q['run'])
            if name not in ('locomotion.json','pairs.csv','detections.jsonl','tracking.mp4','chart.png','report.html') and not re.fullmatch(r'example_\d{6}\.jpg',name):raise ValueError('지원하지 않는 추적 파일입니다.')
        path=(folder/name).resolve()
        if path.parent!=folder.resolve() or not path.is_file():raise ValueError('추적 파일을 찾을 수 없습니다.')
        return path

    def locomotion_get(self,action,q):
        from .cage_locomotion import read_report
        if action=='locomotion-sources':
            sources=self.activity_sources()
            for s in sources['sessions']:
                s['locomotion']=[]
                for p in sorted((self.activity_source(s['id'])/'locomotion').glob('loco_*/locomotion.json'),key=lambda x:x.stat().st_mtime,reverse=True):
                    try:
                        r=read_json(p);s['locomotion'].append(dict(id=p.parent.name,state=r['state'],unit=r['unit'],metadata=r['metadata'],summary=r.get('summary')))
                    except (ValueError,OSError,KeyError):pass
            return sources
        if action=='locomotion-report':return read_report(self.locomotion_run(q['session'],q['run']))
        if action=='locomotion-comparison':return read_json(self.locomotion_file(dict(q,file='comparison.json')))
        raise ValueError('지원하지 않는 이동성 조회입니다.')

    def locomotion_workflow(self,action,v):
        from .cage_locomotion import analyze,read_report,write_comparison
        if action=='locomotion-analyze':
            folder=self.activity_source(v.get('session'))
            if not v.get('calibration') and v.get('pixel_preview') is not True:
                raise ValueError('cm 단위 분석에는 바닥 네 모서리와 실측 길이 보정이 필요합니다. 화면을 새로고침해 기존 보정을 불러오거나 픽셀 미리보기를 명시적으로 선택하세요.')
            def execute(cancel,progress):
                answer=analyze(folder,v,self.root/'.ai_venv/Scripts/python.exe',self.root/'models/topviewmouse/detector.pt',cancel,progress,self.predictor_factory)
                if answer['state']=='complete':
                    with self.remote.lock:
                        for path in self.remote.uploads.glob('upload_*/receipt.json'):
                            receipt=read_json(path)
                            if receipt.get('server_session')==folder.name:
                                if answer['report']['unit']=='px' and receipt['results'].get('locomotion',{}).get('unit')=='cm':continue
                                receipt['results']['locomotion']=dict(run=answer['run'],summary=answer['report']['summary'],unit=answer['report']['unit'])
                                self.remote.save(receipt)
                return answer
            return self.start_job('이동거리·평균 속도',execute)
        if action=='locomotion-compare':
            entries=v.get('entries',[])
            if not isinstance(entries,list) or not 1<=len(entries)<=300:raise ValueError('비교할 이동거리 결과를 선택하세요.')
            def execute(cancel,progress):
                reports=[]
                for i,item in enumerate(entries):
                    if cancel.is_set():return dict(state='cancelled')
                    reports.append(read_report(self.locomotion_run(item['session'],item['run'])));progress(i+1,len(entries),'추적 원본·유효 비율 확인 중')
                folder=self.results/'locomotion_comparisons'/('compare_'+uuid.uuid4().hex[:12])
                result=write_comparison(folder,reports,v.get('mode','water'),v.get('metric','mean_speed'))
                return dict(kind='cage_locomotion_comparison',state='complete',comparison=folder.name,report=result)
            return self.start_job('그룹·물 종류별 운동성 비교',execute)
        raise ValueError('지원하지 않는 이동성 작업입니다.')
