"""Recorded-data research UI. Launch: python -m streamlit run dashboard/app.py."""
import sys
from pathlib import Path
from html import escape
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError as exc:
    raise SystemExit('Dashboard dependencies missing. Run: python -m pip install -r requirements-dashboard.txt') from exc
import numpy as np
import pandas as pd
from dashboard.backend import (ROOT, FINAL, MODELS, Replay, load_record, recordings,
                               policy_rows, result_rows, demo_factory, verified_events, demo_alarm_starts)
from dashboard.presentation import (COLORS, TARGETS, TARGET_HELP, SCORE_HELP, MODES,
                                    MODE_LABELS, summary, display_number)

st.set_page_config(page_title='Oxygen Research Studio', page_icon='🫧', layout='wide')
variables = ':root{' + ''.join(f'--{k}:{v};' for k,v in COLORS.items()) + '}'
st.markdown(f'<style>{variables}\n{Path(__file__).with_name("styles.css").read_text()}</style>', unsafe_allow_html=True)
st.markdown('<span class="offline">RECORDED DATA · OFFLINE DEMO</span>', unsafe_allow_html=True)
st.title('Oxygen Research Studio')
st.caption('Offline predictive oxygen research demonstration.')
st.caption('Scope: the banner, saved-policy playback and Research results show the preserved original policy. A separate completed follow-up detected one nonsevere event in three GaussianNB conditions, with per-record budget violations. No follow-up policy replaces this playback.')
st.warning('Research only. Does not recommend or control oxygen delivery. Experimental outputs are not treatment recommendations. Clinician review and prescribed oxygen protocols remain essential. No oxygen delivery is controlled.')
st.markdown('This dashboard replays existing physiological recordings, runs experimental prediction models, and compares their alerts with oxygen-saturation events observed later in the recordings.')
st.markdown('**Final research finding: all four learned models detected 0/14 events at 30 s; 0/15 at 5, 10 and 15 s.** No useful early warning was demonstrated under the tested policies. An alert is not an instruction to administer oxygen.')

@st.cache_data
def csv(path):
    return pd.read_csv(path)

@st.cache_data
def record_data(record):
    return load_record(record)

@st.cache_data
def predictions():
    return pd.read_csv(FINAL / 'outer_predictions.csv.gz', usecols=['Record','Model','Target','Budget','time_s','Score','Threshold','Prediction'])

availability = recordings()
available_ids = [record for record, present in availability.items() if present]
if not available_ids:
    st.error('No supported local recordings are available. Restore the local Signals and Numerics pairs; no data will be downloaded.')
    st.stop()
if st.session_state.get('r1', available_ids[0]) not in available_ids:
    st.session_state['r1'] = available_ids[0]


def move_record(offset):
    position = available_ids.index(st.session_state['r1'])
    st.session_state['r1'] = available_ids[position + offset]


st.selectbox('Select Patient Recording', available_ids,
             format_func=lambda record: f'Recording {record:02d}', key='r1')
previous, following = st.columns(2)
position = available_ids.index(st.session_state['r1'])
previous.button('Previous Recording', disabled=position == 0, on_click=move_record, args=(-1,))
following.button('Next Recording', disabled=position == len(available_ids)-1, on_click=move_record, args=(1,))
st.caption('Recording identifiers do not establish patient identity. Switching recordings resets playback and causal history.')
# Migrate navigation retained by an already-open three-case dashboard.
if st.session_state.get('page','Recording playback') not in ('Recording playback','Research results','About and limitations'):
    st.session_state.pop('page',None)
with st.sidebar:
    st.header('Explore recordings')
    page = st.radio('Navigation', ['Recording playback','Research results','About and limitations'], key='page')
    st.caption('1. Choose a recording. 2. Press Play. 3. Inspect the observed readings and experimental status.')
    st.caption('All data stay local. No training or policy selection takes place here.')


def keep_controls():
    # The UI uses one playback slot; retain controls across page navigation.
    for i in (1,):
        for prefix in ('m','t','mode','b','clock'):
            key = f'{prefix}{i}'
            if key in st.session_state:
                st.session_state[key] = st.session_state[key]


def engine_for(case):
    config = (st.session_state.get(f'r{case}',case),
              MODELS[st.session_state.get(f'm{case}','ELM')],
              st.session_state.get(f't{case}','new_onset'),
              st.session_state.get(f'mode{case}',MODES[0]),
              st.session_state.get(f'b{case}',5))
    record,model,target,mode,budget = config
    if st.session_state.get(f'config{case}') != config:
        wave,numeric = record_data(record)
        if mode == MODES[0]:
            engine = Replay(wave,numeric,saved=policy_rows(predictions(),record,model,target,budget))
        else:
            engine = Replay(wave,numeric,controller_factory=demo_factory(target,model))
        st.session_state[f'engine{case}'] = engine
        st.session_state[f'config{case}'] = config
        st.session_state[f'clock{case}'] = 0.0
        st.session_state[f'play{case}'] = False
    return st.session_state[f'engine{case}'],config


def status_card(info):
    st.markdown(f'<div class="status {info["tone"]}" role="status"><strong>{escape(info["icon"])} {escape(info["status"])}</strong><p>{escape(info["explanation"])}</p></div>',unsafe_allow_html=True)


def provenance(mode):
    if mode == MODES[0]:
        st.caption('SAVED HELD-OUT EVALUATION · This recording was excluded from that model fit. Showing previously saved decisions, not live inference. Complete future-window eligibility was determined retrospectively.')
    else:
        st.caption('DEMONSTRATION ONLY: full-data fit, earlier saved threshold. Not held-out performance or the final budget policy.')
        st.caption('Demo scores are not final-policy event results. No lead-time claims or event credits are computed for this full-data demonstration.')


def recording_card(case):
    with st.container(border=True):
        st.subheader(f'Recording {st.session_state["r1"]:02d} · Local recorded data')
        readings = st.container()
        status = st.container()
        st.selectbox('Prediction model',list(MODELS),key=f'm{case}',help='Choose an existing saved model. Nothing is trained or tuned.')
        with st.expander('Prediction question and playback source'):
            st.selectbox('What should the model predict?',list(TARGETS),format_func=TARGETS.get,key=f't{case}',help=TARGET_HELP)
            mode = st.selectbox('Playback source',MODES,format_func=MODE_LABELS.get,key=f'mode{case}',help='Saved held-out evaluation and full-data model demonstrations are different sources of evidence.')
            st.selectbox('Existing research budget (false starts/hour)',[1,5,10],index=1,key=f'b{case}',disabled=mode != MODES[0],help='Loads an already saved policy, not a clinical limit. Demo mode uses its own saved cutoff, so this control is disabled.')
        try:
            engine,config = engine_for(case)
        except Exception as exc:
            with readings: st.metric('Current oxygen saturation','Unavailable'); st.metric('Pulse rate','Unavailable')
            with status: st.error(f'Recording {st.session_state["r1"]:02d} · Prediction unavailable. {exc}. Restore the local recording or saved artifact, or choose another playback source. No model will be fitted.')
            return
        record,model,target,mode,budget = config
        a,b,c = st.columns(3)
        if a.button('▶ Play',key=f'go{case}',help='Advance recorded time; this is not live monitoring.'):
            st.session_state[f'play{case}'] = True
        if b.button('Ⅱ Pause',key=f'pause{case}'):
            st.session_state[f'play{case}'] = False
        if c.button('↺ Reset',key=f'reset{case}',help='Return this recording to zero and clear its causal history.'):
            st.session_state[f'play{case}'] = False
            st.session_state[f'clock{case}'] = 0.0
            engine.reset()
        cursor = st.slider('Playback time (seconds)',0.0,float(engine.wave.time_s.iloc[-1]),step=1.0,key=f'clock{case}')
        engine.seek(cursor)
        info = summary(engine,target)
        with readings:
            st.markdown(f'**BIDMC {record:02d} · {cursor:.1f} s** · Recorded observations')
            a,b = st.columns(2)
            a.metric('Current oxygen saturation',info['spo2'])
            b.metric('Pulse rate',info['pulse'])
            if not info['valid_spo2']: st.caption('Oxygen reading unavailable: missing, invalid or more than 1.01 s old.')
            if info['pulse']=='Unavailable': st.caption('Pulse rate unavailable: no confirmed PPG pulse within the last 2 s.')
        with status:
            st.caption('Prediction status · Experimental output, 30-second horizon')
            status_card(info)
            st.caption('Last valid model decision: '+(f'{info["last_valid"]:.3f} s' if info['last_valid'] is not None else 'Unavailable · no valid decision yet'))
        st.caption('Selected question: '+TARGETS[target]+'.')
        provenance(mode)
        st.caption('Signal quality is unvalidated: missingness and continuity checks only, not a clinical quality score.')


def figure_style(fig,height):
    fig.update_layout(template='plotly_white',height=height,margin=dict(l=15,r=15,t=15,b=45),
        paper_bgcolor=COLORS['paper'],plot_bgcolor=COLORS['paper'],
        font=dict(color=COLORS['ink'],size=14),
        legend=dict(orientation='h',yanchor='top',y=-.3,font=dict(color=COLORS['ink'],size=13)),
        hoverlabel=dict(bgcolor=COLORS['ink'],font_color=COLORS['paper'],font_size=14))
    fig.update_xaxes(color=COLORS['ink'],linecolor=COLORS['border'],gridcolor=COLORS['soft'],automargin=True)
    fig.update_yaxes(color=COLORS['ink'],linecolor=COLORS['border'],gridcolor=COLORS['soft'],automargin=True)
    return fig


def playback(case):
    if f'engine{case}' not in st.session_state or f'config{case}' not in st.session_state:
        st.info(f'Recording {st.session_state["r1"]:02d}: choose available local data above to start playback.'); return
    engine = st.session_state[f'engine{case}']
    record,model,target,mode,budget = st.session_state[f'config{case}']
    # Never show an earlier configuration after an unsuccessful change.
    if record != st.session_state.get(f'r{case}',case) or MODELS[st.session_state.get(f'm{case}','ELM')] != model or st.session_state.get(f'mode{case}',MODES[0]) != mode or st.session_state.get(f't{case}','new_onset') != target or st.session_state.get(f'b{case}',5) != budget:
        st.info(f'Recording {st.session_state["r1"]:02d}: playback unavailable for the current selection; resolve the recording error above.'); return
    st.subheader(f'Recording {record:02d} · Waveform and predictions')
    st.markdown(f'**Recorded time: {engine.time:.1f} seconds** · {model} · {TARGETS[target]}')
    st.caption('Observed PPG is a light-based pulse waveform, not an ECG. Dots mark pulses confirmed by the existing causal detector. Only the available recording prefix is shown.')
    left,right = st.columns([3,1])
    with left:
        w = engine.visible_wave()
        if len(w)<2: st.info('Press Play or move the recording timeline to see the recorded PPG waveform.')
        fig = go.Figure(go.Scatter(x=w.time_s,y=w.PLETH,name='Recorded PPG',line=dict(color=COLORS['purple'],width=2)))
        pp = [p for p in engine.pulses if engine.time-12<=p['peak_time_s']<=engine.time]
        if pp:
            tx=[p['peak_time_s'] for p in pp]
            fig.add_trace(go.Scatter(x=tx,y=np.interp(tx,engine.wave.time_s.iloc[:engine.index],engine.wave.PLETH.iloc[:engine.index]),mode='markers',marker=dict(color=COLORS['pulse'],size=7,symbol='circle'),name='Confirmed pulse'))
        fig.update_xaxes(title='Recorded time (s) · most recent 12 seconds')
        fig.update_yaxes(title='PPG amplitude (source units)')
        st.plotly_chart(figure_style(fig,360),width='stretch',theme=None,key=f'wave{case}')
    with right:
        st.markdown('**Prediction details**')
        info = summary(engine,target)
        current=info['decision']
        st.metric('Model score',display_number(current.score if current is not None else None,digits=5),help=SCORE_HELP)
        threshold = current.threshold if current is not None else (engine.controller.model.threshold if engine.controller else (engine.saved.Threshold.iloc[0] if not engine.saved.empty else None))
        st.metric('Decision threshold',display_number(threshold,digits=8),help=SCORE_HELP)
        st.caption('Threshold is the saved score cutoff. A missing decision has no current score; a known saved cutoff can still be shown. Displayed numbers are rounded; decisions use the original values.')
        st.caption('Prediction horizon: 30 seconds. Not a countdown to treatment.')
        provenance(mode)
    st.markdown('**Model outputs and later observed events**')
    p=engine.visible_predictions()
    timeline=go.Figure()
    if p.empty: st.info('No model scores available yet. Continue playback; missing or disallowed reading windows can also prevent a decision.')
    if not p.empty:
        timeline.add_trace(go.Scatter(x=p.time_s,y=p.score,name='Model score',line=dict(color=COLORS['purple'],width=2)))
        timeline.add_trace(go.Scatter(x=p.time_s,y=p.threshold,name='Saved cutoff',line=dict(color=COLORS['threshold'],dash='dash',width=2)))
    onsets=[e['Onset_s'] for e in engine.observed_events() if e['Qualifying_Onset']]
    timeline.add_trace(go.Scatter(x=onsets,y=[0]*len(onsets),mode='markers',marker=dict(color=COLORS['onset'],symbol='diamond',size=10),name='Observed drop onset'))
    for onset in onsets: timeline.add_vline(x=onset,line_color=COLORS['onset'],line_dash='dot')
    if mode==MODES[0]:
        try:
            starts=csv(FINAL/'alarm_starts.csv')
            starts=starts[(starts.Record==record)&(starts.Model==model)&(starts.Target==target)&(starts.Budget==budget)&(starts.Window_s==30)&(starts.Start_s<=engine.time)]
            timeline.add_trace(go.Scatter(x=starts.Start_s,y=[0]*len(starts),mode='markers',marker=dict(color=COLORS['pulse'],symbol='triangle-up',size=11),name='Saved alarm start'))
            verified=verified_events(csv(FINAL/'event_matches.csv'),record,model,target,budget,engine.time)
            missed=verified[verified.Detected.eq(False)]
            timeline.add_trace(go.Scatter(x=missed.Onset_s,y=[0]*len(missed),mode='markers',marker=dict(color=COLORS['danger_ink'],symbol='x',size=13),name='Verified missed event'))
            with st.expander('Retrospective verification · revealed after observed onset +30 s'):
                st.caption('This is later outcome evidence, not information available to an earlier prediction. Each event and alarm can receive at most one matched credit. Missed events have undefined lead time.')
                if verified.empty: st.info('No event verification available yet. Continue playback until at least 30 seconds after an observed onset.')
                else:
                    for row in verified.itertuples():
                        outcome='Matched' if row.Detected==True else ('Missed' if row.Detected==False else 'Not evaluable')
                        st.write(f'Observed onset: {row.Onset_s:.1f} s · {outcome} · Lead time: {display_number(row.Lead_Time_s,"s")} · {row.Evaluation_Status}')
        except (OSError,KeyError,ValueError) as exc: st.error(f'Recording {st.session_state["r1"]:02d}: saved event evidence unavailable. Restore run02 alarm/event tables. {exc}')
    else:
        starts=demo_alarm_starts(engine.outputs)
        timeline.add_trace(go.Scatter(x=starts,y=[0]*len(starts),mode='markers',marker=dict(color=COLORS['pulse'],symbol='triangle-up',size=11),name='Demo alarm start'))
        st.caption('Demonstration alert starts are not held-out event metrics. No matched/missed event counts or lead times are assigned to demo outputs.')
    timeline.update_xaxes(range=[0,max(engine.time,1)],title='Available recording time (s)')
    timeline.update_yaxes(title='Algorithmic score (uncalibrated)')
    st.plotly_chart(figure_style(timeline,310),width='stretch',theme=None,key=f'timeline{case}')
    st.caption('Triangles: model alarm starts. Diamonds/dotted lines: oxygen drops already observed. Crosses: missed events verified later. Symbol and text labels carry meaning in addition to color.')


def research_results():
    st.header('Study-wide research results')
    st.caption('These preserved pooled results describe the full study, not only the selected recording. Recording-specific outputs and outcomes appear on Recording playback.')
    st.subheader('Final alarm-budget evaluation · No useful early warning demonstrated')
    st.markdown('**All four learned models detected 0/14 eligible events at 30 seconds and 0/15 at 5, 10 and 15 seconds. All three severe-interest eligible events were missed.** Both original training targets and all tested budgets gave this result; detected-event lead times are undefined.')
    st.write('Budgets of 1, 5 or 10 false alarm starts per evaluable hour, with a 5% active-time cap, were research assumptions—not clinically validated limits. Some held-out recordings exceeded the constraints selected using training recordings.')
    a,b,c=st.columns(3)
    target=a.selectbox('Prediction question (results)',list(TARGETS),format_func=TARGETS.get,key='result_target',help=TARGET_HELP)
    budget=b.selectbox('Research false-start budget/hour',[1,5,10],index=1,key='result_budget')
    window=c.selectbox('Seconds before observed onset',[5,10,15,30],index=3,key='result_window',help='Look-back matching windows, not minimum guaranteed warning times. The model prediction horizon remains 30 seconds.')
    try:
        rows=result_rows(csv(FINAL/'summary.csv'),target,budget,window)
        # Vertical cards avoid a wide, horizontally scrolling technical table.
        for row in rows.itertuples():
            with st.container(border=True):
                st.markdown(f'**{row.Model}** · Saved final-policy evaluation')
                st.write(f'Events detected: {row.Detected}/{row.Eligible_Events} · Missed: {row.Missed_Events} · Median detected-event lead: {display_number(row.Lead_Median_s,"s")}')
                a,b,c=st.columns(3)
                a.metric('False alarm starts/hour',f'{row.False_Alarm_Starts_Per_Hour:.2f}',help='Off-event alarm starts divided by evaluable hours. A start is off-event if no qualifying onset follows within the chosen window.')
                b.metric('Time with alarm active',f'{100*row.Alarm_Active_Fraction:.2f}%',help='Positive left-held decision duration divided by eligible inter-pulse exposure; not the fraction of pulse rows with an alert.')
                c.metric('Evaluable time',f'{row.Exposure_s/3600:.3f} h',help='Valid eligible intervals between pulse decisions, not the full recording duration.')
                with st.expander(f'{row.Model} · timestamp classification details'):
                    st.write(f'Positive prediction precision: {display_number(row.Timestamp_Precision*100,"%") if np.isfinite(row.Timestamp_Precision) else "Undefined (no positive decisions)"}. Positive-outcome recall: {display_number(row.Timestamp_Recall*100,"%")}. These count pulse-level rows, not events.')
                    st.write(f'Off-event starts: {row.Off_Event_Alarm_Starts} · Exact evaluable exposure: {row.Exposure_s:.3f} s. One event and alarm receive at most one matching credit in [onset−window, onset).')
        st.caption('Always-alert is not budget compliant. Zero alerts never establish safety. Undefined means no detected events (lead) or no positive decisions (precision); unavailable means missing evidence.')
    except (OSError,KeyError,ValueError) as exc: st.error(f'Final results unavailable. Restore results/alarm_budget_experiment/run02/summary.csv. {exc}')
    st.subheader('Historical recall-oriented evaluation · A different operating policy')
    st.write('Earlier cutoffs were selected on training-only recording-grouped predictions to reach 92% timestamp recall. They produced frequent or nearly continuous alerts. The original reusable-positive metric detected 14/14 events, as did always-alert. Audited new-start matching gave onset ELM 13/14 with high burden; the final budget policy gave 0/14. These results are not interchangeable.')
    st.caption('Timestamp recall: fraction of positive-outcome pulse rows alerted. Precision: fraction of alerted rows with a positive outcome. False episodes/hour: legacy alert runs containing no positive target row per eligible hour; this differs from audited off-event alarm starts/hour.')
    try:
        for row in csv(ROOT/'results/final/new_onset/model_comparison.csv').itertuples():
            with st.expander(f'{row.Model} · Earlier new-drop timestamp metrics'):
                st.write(f'Recall: {row.Recall:.1%} · Precision: {row.Precision:.1%} · Legacy false episodes/hour: {row.False_Alerts_Per_Hour:.2f} · Alerted pulse rows: {row.Alert_Fraction:.1%} · Rows: {row.Samples}')
                st.caption('These are historical timestamp outcomes, not final-budget results or time-based active fraction.')
    except (OSError,KeyError,ValueError) as exc: st.error(f'Historical evidence unavailable. Restore the saved comparison table. {exc}')


def about():
    st.header('About and limitations')
    st.write('This study uses only 30 existing local recordings. Patient identities are unverified: leaving out one recording does not prove generalization to unseen patients. There are only 14 eligible 30-second events, 15 at shorter windows and three severe-interest events. Data were examined retrospectively; the model is not clinically validated.')
    st.write('An observed study onset is an adjacent valid >94% to ≤94% crossing. One reading suffices; it is not a clinically adjudicated sustained desaturation. A new-drop prediction requires current oxygen above 94%. The other target includes an ongoing low reading. Neither target tells a clinician when oxygen is required.')
    st.write('Physiological sampling delays and the 30-second horizon are distinct from computation time. PPG is not ECG. Missing data, old numeric readings, waveform gaps and unvalidated signal quality limit interpretation.')
    st.caption('Read docs/DASHBOARD_GUIDE.md for architecture, exact scientific definitions and manual visual checks. Historical evidence and integrity receipts remain preserved, including the three known documentation-hash test failures.')


@st.fragment(run_every=1)
def workspace():
    keep_controls()
    # Remove idle slots left by an older three-case session after a hot reload.
    for old in (2,3):
        for prefix in ('engine','config','play','clock','r','m','t','mode','b'):
            st.session_state.pop(f'{prefix}{old}', None)
    try:
        engine,config = engine_for(1)
    except Exception as exc:
        # Never retain an earlier recording's visible state after a failed switch.
        st.session_state.pop('engine1', None)
        st.session_state.pop('config1', None)
        st.session_state['play1'] = False
        st.error(f'Recording {st.session_state["r1"]:02d} · Prediction unavailable. {exc}. Restore the selected local artifacts or choose another source. No model will be fitted.')
        # Keep model/source controls available so the user can resolve the error.
        if page=='Recording playback': recording_card(1)
        elif page=='Research results': research_results()
        else: about()
        return
    if st.session_state.get('play1',False):
        end=float(engine.wave.time_s.iloc[-1])
        st.session_state['clock1']=min(end,st.session_state.get('clock1',0)+1)
        engine.seek(st.session_state['clock1'])
        if st.session_state['clock1']>=end: st.session_state['play1']=False
    if page=='Recording playback':
        recording_card(1)
        playback(1)
    elif page=='Research results': research_results()
    else: about()
workspace()
