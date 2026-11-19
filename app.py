import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import uuid
from database import init_db, rows, connect, event
from modules.automation_engine import process_report, verify_report
from modules.work_order_service import (approve_closure, request_reinspection,
                                       submit_repair_evidence, update_work_order)
from modules.priority_engine import calculate
from config import (MAX_UPLOAD_MB, MODEL_PATH, MODEL_SHA256, MODEL_SIZE_BYTES, MODEL_URL,
                    DETECTION_CONFIDENCE, DETECTION_IOU, CAMERA_FRAME_INTERVAL_SECONDS, CAMERA_MAX_SAMPLES)
from modules.demo_reset import reset_demo_environment

st.set_page_config(page_title='RoadGuard AI | Operations',page_icon='🛣️',layout='wide')
init_db()
theme_css = (Path(__file__).parent / 'assets' / 'theme.css').read_text(encoding='utf-8')
st.markdown(f'<style>{theme_css}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 🛣️ ROADGUARD **AI**')
    st.caption('SMART CITY · MAINTENANCE OPS')
    page=st.radio('WORKSPACE',['Command Center','Live Camera','Citizen Report','Officer Report','TrustShield AI','Damage Map','Work Orders','Contractor Portal','Repair Verification','Analytics & Reports','Demo & Settings'],label_visibility='collapsed')
    st.divider(); st.caption('DEMO ENVIRONMENT')
    st.caption('Local SQLite · Simulated notifications')

def metric(label,value,help=None): st.metric(label,value,help=help)
def report_form(source):
    conf, iou = st.columns(2)
    confidence = conf.slider('Pothole confidence threshold', 0.05, 0.95, float(DETECTION_CONFIDENCE), 0.05,
                             help='Higher values reduce weak candidate boxes and can miss more potholes.')
    nms_iou = iou.slider('Detection NMS IoU threshold', 0.10, 0.90, float(DETECTION_IOU), 0.05,
                         help='Controls overlap suppression between predicted boxes.')
    with st.form('report_'+source,clear_on_submit=True):
        c1,c2=st.columns(2)
        road=c1.text_input('Road name *',placeholder='e.g. MG Road')
        ward=c2.selectbox('Ward *',['Ward 12','Ward 4','Ward 7','Ward 9'])
        a,b=c1.columns(2); lat=a.number_input('Latitude',value=None,format='%.6f',placeholder='Optional'); lon=b.number_input('Longitude',value=None,format='%.6f',placeholder='Optional')
        who=st.text_input('Reporter / officer name',value='Demo reporter')
        desc=st.text_area('Description',placeholder='Describe the road condition and nearby landmark')
        image=st.file_uploader('Evidence photograph *',type=['jpg','jpeg','png','webp'],help=f'Maximum {MAX_UPLOAD_MB} MB')
        go=st.form_submit_button('Submit & run automated workflow',type='primary',width='stretch')
    if go:
        if not road.strip() or not image: st.error('Road name and evidence photograph are required.')
        else:
            try:
                rid=process_report(image,source,road.strip(),ward,lat,lon,desc,who,
                                   detection_options={'confidence': confidence, 'iou': nms_iou})
                st.session_state['latest_detection_report_id'] = rid
                stored=rows('SELECT verification_status,duplicate_of FROM reports WHERE id=?',(rid,))[0]
                if stored['duplicate_of']:
                    st.success(f'Report RG-{rid:04d} linked to canonical RG-{stored["duplicate_of"]:04d}; repeat count and priority updated. No second dispatch was created.')
                else:
                    st.success(f'Report RG-{rid:04d} recorded and awaiting human verification. No contractor was dispatched.')
                st.rerun()
            except Exception as e: st.error(f'Could not process report: {e}')

def event_log(limit=16):
    st.caption('Live log from SQLite. Notification rows are simulated; no external message is sent.')
    if st.button('Refresh activity', key=f'refresh_events_{limit}'):
        st.rerun()
    ev=rows('''SELECT e.created_at,e.report_id,e.stage,e.message,e.level,r.source,r.road
        FROM events e LEFT JOIN reports r ON r.id=e.report_id ORDER BY e.id DESC LIMIT ?''',(limit,))
    if not ev:
        st.info('No events yet. Submit a report to create persisted workflow events.')
    else:
        df=pd.DataFrame([dict(time=e['created_at'],report=f"RG-{e['report_id']:04d}" if e['report_id'] else '—',
                              source=e['source'] or 'System',road=e['road'] or '',stage=e['stage'],
                              level=e['level'],event=e['message']) for e in ev])
        st.dataframe(df,width='stretch',hide_index=True)


def latest_workflow_status():
    latest=rows('SELECT id,source,road FROM reports ORDER BY id DESC LIMIT 1')
    if not latest:
        st.info('No report has entered the workflow yet.')
        return
    report=latest[0]
    logged={r['stage'] for r in rows('SELECT stage FROM events WHERE report_id=?',(report['id'],))}
    order=rows('SELECT id,status FROM work_orders WHERE report_id=?',(report['id'],))
    stages=[('Received','Received'),('AI detection','Detection'),('Duplicate check','Duplicate check'),
            ('TrustShield','TrustShield'),('Priority','Priority'),('Human verification','Human verification'),
            ('Contractor assignment','Assignment'),('Notification record','Notification'),
            ('Repair evidence','Repair evidence'),('Admin approval','Admin approval')]
    st.caption(f"Latest report RG-{report['id']:04d} · {report['source']} · {report['road']}")
    cols=st.columns(2)
    for index,(label,stage) in enumerate(stages):
        complete=stage in logged
        if label=='Contractor assignment': complete=bool(order)
        if label=='Notification record': complete=bool(order) and 'Notification' in logged
        cols[index%2].markdown(f"{'🟢' if complete else '⚪'} **{label}** · {'recorded' if complete else 'not recorded'}")
    if order:
        st.caption(f"Persisted work order {order[0]['id']} · current state: {order[0]['status']}")


if page=='Command Center':
    st.markdown('<div class="hero"><div class="pill">AUTONOMOUS ROAD MAINTENANCE · LIVE OPERATIONS</div><h1>Good day, Operations.</h1><div class="muted">One intake pipeline for camera, citizen and municipal evidence. Every downstream action is logged.</div></div>',unsafe_allow_html=True)
    reps=rows('SELECT * FROM reports'); orders=rows('SELECT * FROM work_orders');
    cols=st.columns(4)
    for c,(lab,val) in zip(cols,[('Total reports',len(reps)),('Camera detections',sum(r['source']=='Vehicle camera' for r in reps)),('Citizen reports',sum(r['source']=='Citizen' for r in reps)),('Officer reports',sum(r['source']=='Municipal officer' for r in reps))]): c.metric(lab,val)
    cols=st.columns(4)
    for c,(lab,val) in zip(cols,[('Human-verified reports',sum(r['verification_status']=='Verified' for r in reps)),('Pending human review',sum(r['verification_status']=='Pending' for r in reps)),('Open work orders',sum(o['status'] not in ('Closed','Rejected') for o in orders)),('Completed repairs',sum(o['status']=='Closed' for o in orders))]): c.metric(lab,val)
    left,right=st.columns([1.05,1])
    with left:
        st.subheader('Priority distribution')
        if reps:
            df=pd.DataFrame([dict(priority=r['priority']) for r in reps]); st.bar_chart(df['priority'].value_counts(),color='#31c9bd')
        else: st.info('New report priorities will appear here.')
        st.subheader('Damage distribution by ward')
        if reps: st.bar_chart(pd.DataFrame([dict(ward=r['ward']) for r in reps])['ward'].value_counts(),color='#55a8ff')
    with right:
        st.subheader('Automation pipeline')
        latest_workflow_status()
    st.subheader('Live automation event log'); event_log()
elif page in ('Citizen Report','Officer Report'):
    st.title('Public road report' if page=='Citizen Report' else 'Municipal field inspection')
    st.caption('Citizen and municipal reports enter independently. AI results are advisory; every report awaits human verification before contractor dispatch.')
    report_form('Citizen' if page=='Citizen Report' else 'Municipal officer')
elif page=='TrustShield AI':
    st.title('TrustShield AI · Human evidence review')
    st.caption('TrustShield is advisory. Every new report waits for a reviewer decision; AI confidence alone never dispatches a contractor.')
    rs=rows('SELECT * FROM reports ORDER BY id DESC')
    if rs:
        df=pd.DataFrame([dict(id=f"RG-{r['id']:04d}",source=r['source'],road=r['road'],ward=r['ward'],
                              score=r['trust_score'],trust=r['trust_status'],detection=r['detection'],
                              model_confidence=r['detection_conf'],verification=r['verification_status'],
                              reasons=r['trust_reasons'],duplicate=r['duplicate_of']) for r in rs])
        st.dataframe(df,width='stretch',hide_index=True)
        rid=st.selectbox('Review report',[r['id'] for r in rs],
                         format_func=lambda x:f'RG-{x:04d} · '+next(r['road'] for r in rs if r['id']==x))
        report=next(row for row in rs if row['id']==rid)
        a,b,c=st.columns([1,1,1.4])
        with a:
            if report['image_path']: st.image(report['image_path'],caption='Submitted evidence',width='stretch')
        with b:
            if report['annotated_path'] and Path(report['annotated_path']).is_file():
                st.image(report['annotated_path'],caption='Model boxes · decision support only',width='stretch')
            else: st.info('No annotated detections were produced for this report.')
        with c:
            st.markdown(f"**TrustShield {report['trust_status']} | {report['trust_score']}/100**  \nDetection: {report['detection']}  \nBest confidence: {report['detection_conf'] if report['detection_conf'] is not None else 'Not available'}  \nPriority: {report['priority']} | {report['priority_score']}/100  \nReasons: {report['trust_reasons']}")
            st.caption('Priority weights: severity 40%, traffic 25%, road importance 20%, repeat reports 15%.')
            import json
            boxes=json.loads(report['detection_boxes'] or '[]')
            if boxes: st.dataframe(pd.DataFrame(boxes),width='stretch',hide_index=True)
        if report['verification_status']=='Duplicate':
            st.warning(f"Merged into canonical report RG-{report['duplicate_of']:04d}. Review the canonical report; duplicates cannot dispatch separately.")
        elif report['verification_status']=='Pending':
            st.warning('Pending human verification. Low-confidence, no-detection, and not-evaluated results are not automatically dispatched.')
            with st.form('human_verification_form'):
                reviewer=st.text_input('Reviewer name *')
                decision=st.radio('Decision',['Confirm road defect','Reject / not a pothole'],horizontal=True)
                reason=st.text_area('Verification reason *',placeholder='Describe the visual evidence and why this decision is appropriate.')
                submit=st.form_submit_button('Record decision',type='primary')
            if submit:
                try:
                    work_id=verify_report(rid,reviewer,reason,confirmed=(decision=='Confirm road defect'))
                    if decision=='Confirm road defect':
                        st.success(f"Human verification recorded. {'Work order '+work_id+' assigned.' if work_id else 'No ward contractor is configured; assignment exception was logged.'}")
                    else: st.success('Rejection recorded. No contractor was assigned.')
                    st.rerun()
                except ValueError as exc: st.error(str(exc))
        else:
            st.success(f"{report['verification_status']} by {report['verified_by'] or 'reviewer'} · {report['verified_at'] or ''}")
            st.write(f"**Recorded verification reason:** {report['verification_reason'] or 'No reason recorded'}")
            order=rows('SELECT id,status FROM work_orders WHERE report_id=?',(rid,))
            if order: st.caption(f"Work order {order[0]['id']} · {order[0]['status']}")
    else: st.info('No evidence reports yet.')
elif page=='Live Camera':
    st.title('Vehicle camera monitoring')
    st.warning('Simulation mode. This prototype accepts a local video clip. A configured pothole YOLO model is required for genuine frame detections; no detections are fabricated.')
    st.markdown('**Camera ID:** RG-CAM-01 &nbsp; · &nbsp; **Source:** Prerecorded demo video')
    cc, ci, cs, cm = st.columns(4)
    confidence = cc.slider('Confidence', 0.05, 0.95, float(DETECTION_CONFIDENCE), 0.05, key='camera_confidence')
    nms_iou = ci.slider('NMS IoU', 0.10, 0.90, float(DETECTION_IOU), 0.05, key='camera_iou')
    frame_skip = cs.slider('Seconds between frames', 0.5, 10.0, float(CAMERA_FRAME_INTERVAL_SECONDS), 0.5)
    max_samples = cm.slider('Maximum sampled frames', 1, 60, int(CAMERA_MAX_SAMPLES), 1)
    video=st.file_uploader('Load prerecorded road video',type=['mp4','mov','avi'])
    if video:
        st.video(video)
        st.info('Start Monitoring samples the prerecorded clip and runs the configured detector. Candidate frames enter the shared review queue; camera detections do not dispatch contractors before human verification.')
    if st.button('Start Monitoring',type='primary'):
        if not video:
            st.warning('Choose a prerecorded clip before starting monitoring.')
        else:
            from modules.camera_monitor import monitor_video
            vp=Path('data/uploads')/(uuid.uuid4().hex+'_'+Path(video.name).name)
            vp.write_bytes(video.getvalue())
            try:
                with st.spinner('Sampling video and running YOLO pothole inference…'):
                    frame_results=monitor_video(vp, max_samples=max_samples,
                                                seconds_between_samples=frame_skip,
                                                confidence=confidence, iou=nms_iou)
            except Exception as exc:
                frame_results=[]
                st.error(f'Camera processing failed: {type(exc).__name__}: {exc}')
            examined=len(frame_results)
            candidates=sum(frame['result']['result']=='Pothole candidate' for frame in frame_results)
            submitted=sum(frame.get('report_id') is not None for frame in frame_results)
            suppressed=sum(frame.get('suppressed', False) for frame in frame_results)
            if not examined: st.error('No decodable frames found in this clip.')
            elif not candidates: st.info(f'Processed {examined} frames; no pothole candidates returned. If no trained model is configured, each frame is marked not evaluated.')
            else: st.success(f'Processed {examined} frames; {candidates} candidate frame(s), {submitted} report(s) entered the human review queue. No contractor was dispatched.')
            if suppressed: st.caption(f'{suppressed} repeated frame detection(s) were associated with an earlier pothole and suppressed to avoid duplicate reports and work orders.')
            st.subheader('Sampled frame inference')
            for frame in frame_results:
                frame_result=frame['result']
                with st.expander(f"Frame {frame['frame_no']} · {frame_result['result']} · {len(frame_result.get('boxes', []))} box(es)",expanded=frame_result['result']=='Pothole candidate'):
                    if frame_result.get('annotated_path') and Path(frame_result['annotated_path']).is_file():
                        st.image(frame_result['annotated_path'],caption=f"Pothole boxes · best confidence {frame_result['confidence']:.3f}",width='stretch')
                        st.dataframe(pd.DataFrame(frame_result['boxes']),width='stretch',hide_index=True)
                    else: st.caption(frame_result['note'])
    if st.button('Stop Monitoring'): event('Camera','Camera RG-CAM-01 monitoring stopped.'); st.info('Monitoring stopped.')
    st.subheader('Recent camera events'); event_log()
elif page in ('Work Orders','Contractor Portal','Repair Verification'):
    st.title({'Work Orders':'Work order management','Contractor Portal':'Contractor portal','Repair Verification':'Repair evidence verification'}[page])
    if page=='Repair Verification':
        st.info('Closure requires stored after-repair evidence and an explicit admin reviewer decision. This demo app records approver identity but does not provide login-based authentication.')
    os=rows("SELECT w.*,r.road,r.image_path,r.priority,r.priority_score,r.lat,r.lon,r.trust_status,r.verification_status FROM work_orders w JOIN reports r ON r.id=w.report_id ORDER BY w.created_at DESC")
    if not os:
        pending=rows("SELECT COUNT(*) n FROM reports WHERE verification_status='Pending'")[0]['n']
        st.info(f'No work orders yet. {pending} report(s) await human verification; only verified canonical reports are dispatched.')
    for order in os:
        with st.expander(f"{order['id']} · {order['road']} · {order['priority']} · {order['status']}",expanded=(page=='Contractor Portal')):
            a,b=st.columns([1,2])
            with a:
                if order['image_path']: st.image(order['image_path'],caption='Before evidence',width='stretch')
            with b:
                st.write(f"**Contractor:** {order['contractor']} · **Ward:** {order['ward']} · **Due:** {order['due_at'][:10]}")
                st.write(f"**Priority:** {order['priority']} · {order['priority_score']}/100 · **TrustShield:** {order['trust_status']}")
                if page=='Contractor Portal' and order['status'] in {'Assigned','Accepted','In progress','On site'}:
                    from modules.work_order_service import STATUS_TRANSITIONS
                    allowed=sorted(STATUS_TRANSITIONS.get(order['status'],set()))
                    if allowed:
                        actor=st.text_input('Contractor representative (demo)',value=order['contractor'],key='actor_'+order['id'])
                        status=st.selectbox('Next work status',allowed,key='next_'+order['id'])
                        if st.button('Save contractor update',key='save_'+order['id']):
                            try:
                                update_work_order(order['id'],status,actor)
                                st.success('Contractor update persisted and logged.')
                                st.rerun()
                            except ValueError as exc: st.error(str(exc))
                if page in {'Contractor Portal','Work Orders'} and order['status'] in {'In progress','On site'}:
                    actor=st.text_input('Evidence submitter',value=order['contractor'],key='evidence_actor_'+order['id'])
                    after=st.file_uploader('After-repair photograph',type=['jpg','jpeg','png','webp'],key='after_'+order['id'])
                    if after and st.button('Submit completion request',key='complete_'+order['id']):
                        try:
                            submit_repair_evidence(order['id'],after,actor)
                            st.success('Evidence stored. The work order awaits admin review; no repair is marked complete yet.')
                            st.rerun()
                        except ValueError as exc: st.error(str(exc))
                if order['after_path']:
                    if Path(order['after_path']).is_file():
                        q1,q2=st.columns(2)
                        if order['image_path']: q1.image(order['image_path'],caption='Before',width='stretch')
                        q2.image(order['after_path'],caption='Submitted after evidence · human review required',width='stretch')
                    st.warning('AI repair assessment is not available. A reviewer must inspect the submitted evidence.')
                if page=='Repair Verification' and order['status']=='Completion requested':
                    admin=st.text_input('Admin approver name *',key='admin_'+order['id'])
                    reason=st.text_area('Admin approval reason *',key='close_reason_'+order['id'])
                    if st.button('Approve closure',key='approve_'+order['id'],type='primary'):
                        try:
                            approve_closure(order['id'],admin,reason)
                            st.success('Admin approval recorded and work order closed.')
                            st.rerun()
                        except ValueError as exc: st.error(str(exc))
                    reviewer=st.text_input('Reviewer for reinspection',key='reinspect_by_'+order['id'])
                    reinspection=st.text_input('Reinspection reason',key='reinspect_reason_'+order['id'])
                    if st.button('Request reinspection',key='reinspect_'+order['id']):
                        try:
                            request_reinspection(order['id'],reviewer,reinspection)
                            st.success('Reinspection requested and logged.')
                            st.rerun()
                        except ValueError as exc: st.error(str(exc))
                if order['status']=='Closed':
                    st.success(f"Admin approved by {order['closed_by'] or 'reviewer'} on {order['closed_at'] or 'unknown date'}.")
                    st.caption(f"Closure reason: {order['closure_reason'] or 'No reason recorded'}")
elif page=='Damage Map':
    st.title('Damage map')
    rs=rows('SELECT road,ward,lat,lon,priority,trust_status FROM reports WHERE lat IS NOT NULL AND lon IS NOT NULL')
    if rs:
        import folium
        from streamlit_folium import st_folium
        m=folium.Map(location=[rs[0]['lat'],rs[0]['lon']],zoom_start=12,tiles='CartoDB dark_matter')
        for r in rs: folium.Marker([r['lat'],r['lon']],tooltip=f"{r['road']} · {r['priority']}",popup=f"{r['ward']} · TrustShield {r['trust_status']}").add_to(m)
        st_folium(m,use_container_width=True,height=550)
    else: st.info('Reports with coordinates will appear on the map.')
elif page=='Analytics & Reports':
    st.title('Analytics & reports'); rs=rows('SELECT * FROM reports ORDER BY id DESC')
    if rs:
        df=pd.DataFrame([dict(report=f"RG-{r['id']:04d}",date=r['created_at'][:10],source=r['source'],ward=r['ward'],road=r['road'],trust=r['trust_status'],priority=r['priority'],score=r['priority_score']) for r in rs])
        st.bar_chart(df.groupby('source').size(),color='#39c9bb'); st.dataframe(df,width='stretch',hide_index=True)
        st.download_button('Download CSV',df.to_csv(index=False),'roadguard_reports.csv','text/csv')
    else: st.info('No report data yet.')
    st.subheader('Automation event log'); event_log(100)
elif page=='Demo & Settings':
    st.title('Demo mode & system settings')
    st.info('Seeded contractors and notification dispatches are demo data. Notifications are simulated; no external email or contractor message is sent.')
    import os
    model_path=Path(os.environ.get('ROADGUARD_YOLO_MODEL',str(MODEL_PATH))).expanduser()
    if model_path.is_file():
        if model_path.resolve()==MODEL_PATH.resolve(): model_integrity=f'pinned SHA-256 `{MODEL_SHA256}`'
        else: model_integrity='custom model path · pothole class names are validated at inference'
        st.markdown(f'**Pothole detector:** Ready · `{model_path}` · {model_path.stat().st_size:,} bytes · {model_integrity}')
    else:
        st.warning(f'Pothole weights are missing at `{model_path}`. Run the project download script. Inference will report Not evaluated until installed.')
    st.caption(f'Model source: {MODEL_URL} · MIT checkpoint; RDD2022 training data has separate CC BY-SA 4.0 terms.')
    st.markdown('**Dispatch gate:** all new reports require a recorded human decision. Priority weights: severity 40%, traffic 25%, road importance 20%, repeat reports 15%.')
    st.markdown('**Duplicate rules:** 250 m GPS proximity or perceptual hash Hamming distance ≤ 8. **Demo wards:** Ward 12, Ward 4 and Ward 7 have seeded contractors.')
    st.subheader('Complete demo reset')
    st.warning('This removes report/work-order/event records and files in the app upload folder. Seeded contractor records and model/evaluation files are retained.')
    reset_confirm=st.checkbox('I understand this deletes all local demo reports, work orders, event history, and uploaded evidence.',key='confirm_demo_reset')
    if st.button('Reset demo environment',type='secondary',disabled=not reset_confirm):
        try:
            result=reset_demo_environment()
            st.success(f"Reset complete · {result['deleted_files']} uploaded file(s) removed. Contractor seeds retained.")
            st.rerun()
        except Exception as exc: st.error(f'Demo reset failed safely: {type(exc).__name__}: {exc}')

latest_report_id=st.session_state.pop('latest_detection_report_id',None)
if latest_report_id:
    latest=rows('SELECT detection,detection_conf,detection_model,detection_boxes,annotated_path,verification_status,verified_by,verification_reason FROM reports WHERE id=?',(latest_report_id,))
    if latest:
        latest=latest[0]
        st.divider()
        st.subheader(f'Latest model output · RG-{latest_report_id:04d}')
        left,right=st.columns([1.2,1])
        with left:
            if latest['annotated_path'] and Path(latest['annotated_path']).is_file(): st.image(latest['annotated_path'],caption='Pothole boxes labelled with confidence scores',width='stretch')
            else: st.info('No annotation was produced. Detection may be unavailable or no pothole boxes passed the confidence threshold.')
        with right:
            st.metric('Detection result',latest['detection'])
            st.write(f"**Model:** {latest['detection_model'] or 'Not available'}")
            st.write(f"**Best pothole confidence:** {latest['detection_conf'] if latest['detection_conf'] is not None else 'Not available'}")
            st.write(f"**Human review:** {latest['verification_status']} · {latest['verified_by'] or 'review pending'}")
            st.caption('AI results are advisory. Contractor dispatch remains blocked until verification.')
            import json
            boxes=json.loads(latest['detection_boxes'] or '[]')
            if boxes: st.dataframe(pd.DataFrame(boxes),width='stretch',hide_index=True)
