"""Optional Streamlit interaction checks (install requirements-dashboard.txt)."""
import importlib.util
import unittest
from pathlib import Path

@unittest.skipUnless(importlib.util.find_spec('streamlit') and importlib.util.find_spec('plotly'), 'Optional dashboard dependencies unavailable')
class DashboardUITests(unittest.TestCase):
    def app(self):
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'),default_timeout=60).run()

    def test_recording_selector_updates_entire_single_recording_view(self):
        from dashboard.backend import recordings, load_record
        app=self.app()
        self.assertFalse(app.exception); self.assertFalse(app.error)
        selector=app.selectbox(key='r1')
        self.assertEqual(selector.label,'Select Patient Recording')
        self.assertEqual(selector.options,[f'Recording {r:02d}' for r,ok in recordings().items() if ok])
        self.assertEqual(len(app.slider),1)
        self.assertFalse(any('Case ' in h.value or 'Three recordings' in h.value for h in app.subheader))
        self.assertNotIn('Three-case comparison',app.radio(key='page').options)
        self.assertTrue(any('0/14 events at 30 s; 0/15' in m.value for m in app.markdown))
        self.assertTrue(any('not treatment recommendations' in w.value and 'No oxygen delivery is controlled' in w.value for w in app.warning))
        first=app.session_state['engine1']
        app.slider(key='clock1').set_value(12).run()
        app.button(key='go1').click().run()
        app.selectbox(key='r1').select(12).run()
        engine=app.session_state['engine1']
        self.assertIsNot(engine,first)
        self.assertIsNot(engine.state,first.state)
        self.assertEqual(app.session_state['config1'][0],12)
        self.assertEqual(engine.time,0)
        self.assertFalse(app.session_state['play1'])
        self.assertEqual(len(engine.pulses),0)
        self.assertEqual(app.slider(key='clock1').value,0)
        _,numeric=load_record(12)
        expected=f'{numeric.SpO2.iloc[0]:.0f}%'
        self.assertEqual(next(m.value for m in app.metric if m.label=='Current oxygen saturation'),expected)
        self.assertTrue(any('Recording 12' in h.value and 'Waveform' in h.value for h in app.subheader))
        self.assertTrue(all('Recording 01' not in h.value for h in app.subheader))
        app.selectbox(key='r1').select(1).run()
        self.assertIsNot(app.session_state['engine1'],first)
        self.assertEqual(app.session_state['engine1'].time,0)
        self.assertEqual(len(app.slider),1)
        self.assertNotIn('engine2',app.session_state)
        self.assertNotIn('engine3',app.session_state)
        # Hot reload from an older workspace must not retain its navigation/slots.
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py'),default_timeout=60)
        app.session_state['page']='Overview'
        app.session_state['engine2']=object()
        app.session_state['engine3']=object()
        app.run()
        self.assertEqual(app.radio(key='page').value,'Recording playback')
        self.assertNotIn('engine2',app.session_state)
        self.assertNotIn('engine3',app.session_state)
        self.assertEqual(len(app.slider),1)
        self.assertFalse(app.exception); self.assertFalse(app.error)

    def test_demo_reset_model_selection_and_record_switch_isolation(self):
        app=self.app()
        app.selectbox(key='mode1').select('Causal full-data demo model').run()
        app.slider(key='clock1').set_value(3).run()
        original=app.session_state['engine1']
        self.assertGreater(len(original.outputs),0)
        self.assertTrue(any('DEMONSTRATION ONLY' in c.value and 'Not held-out performance' in c.value for c in app.caption))
        self.assertTrue(any('No lead-time claims or event credits' in c.value for c in app.caption))
        app.selectbox(key='r1').select(17).run()
        switched=app.session_state['engine1']
        for attr in ('controller','state','outputs'):
            self.assertIsNot(getattr(switched,attr),getattr(original,attr))
        self.assertIsNot(switched.controller.feature_state,original.controller.feature_state)
        self.assertIsNot(switched.controller.waveform_state,original.controller.waveform_state)
        self.assertIsNot(switched.controller.model,original.controller.model)
        self.assertEqual(switched.time,0)
        self.assertEqual(len(switched.outputs),0)
        app.slider(key='clock1').set_value(3).run()
        app.button(key='go1').click().run()
        app.button(key='pause1').click().run()
        self.assertFalse(app.session_state['play1'])
        app.button(key='reset1').click().run()
        self.assertEqual(app.session_state['engine1'].time,0)
        app.selectbox(key='m1').select('Logistic Regression').run()
        self.assertEqual(app.session_state['engine1'].controller.model.name,'Logistic')
        self.assertEqual(app.session_state['engine1'].time,0)
        self.assertFalse(app.exception); self.assertFalse(app.error)

    def test_navigation_plain_labels_and_missing_artifact(self):
        from unittest.mock import patch
        app=self.app()
        self.assertTrue(any('Insufficient valid signal' in m.value for m in app.markdown))
        self.assertTrue(any('Last valid model decision: Unavailable' in c.value for c in app.caption))
        with patch('dashboard.backend.demo_factory',side_effect=FileNotFoundError('test missing trusted model')):
            app.selectbox(key='mode1').select('Causal full-data demo model').run()
            self.assertTrue(any('Recording 01' in e.value and 'Prediction unavailable' in e.value and 'Restore' in e.value for e in app.error))
            self.assertNotIn('engine1',app.session_state)
            self.assertFalse(app.exception)
        app.selectbox(key='mode1').select('Saved held-out final policy').run()
        for page in ['Research results','About and limitations','Recording playback']:
            app.radio(key='page').set_value(page).run()
            self.assertFalse(app.exception,page); self.assertFalse(app.error,page)
            if page=='Research results':
                self.assertTrue(any('All three severe-interest eligible events were missed' in m.value for m in app.markdown))
                self.assertTrue(any('Undefined' in m.value for m in app.markdown))
                app.selectbox(key='r1').select(30).run()
                self.assertEqual(app.session_state['config1'][0],30)
        self.assertEqual(app.selectbox(key='r1').value,30)
        self.assertEqual(len(app.slider),1)

    def test_previous_next_and_available_record_filter(self):
        from unittest.mock import patch
        with patch('dashboard.backend.recordings',return_value={1:True,12:False,17:True,30:True}):
            app=self.app()
            self.assertEqual(app.selectbox(key='r1').options,['Recording 01','Recording 17','Recording 30'])
            previous=lambda: next(b for b in app.button if b.label=='Previous Recording')
            following=lambda: next(b for b in app.button if b.label=='Next Recording')
            self.assertTrue(previous().disabled)
            following().click().run()
            self.assertEqual(app.selectbox(key='r1').value,17)
            following().click().run()
            self.assertEqual(app.selectbox(key='r1').value,30)
            self.assertTrue(following().disabled)
            previous().click().run()
            self.assertEqual(app.selectbox(key='r1').value,17)
            self.assertEqual(app.session_state['engine1'].time,0)
            self.assertFalse(app.exception);self.assertFalse(app.error)

if __name__=='__main__': unittest.main()
