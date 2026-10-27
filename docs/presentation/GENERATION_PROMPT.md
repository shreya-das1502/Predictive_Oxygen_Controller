# Copy-paste prompt: project presentation with proposed PID integration

Create exactly **20 academic slides** using the accompanying `PPT_Project_Template.pptx`. Use the template as a layout, not as scientific evidence. Audience: a professor assessing the project. Target 9–10 minutes plus questions. Provide finished slide content, speaker notes, captions, citations and editable diagrams.

## Design and layout

Preserve the template's 4:3 aspect ratio (10 ×7.5 inches), white background and full-width dark-blue top/bottom panels, colour `#092034`. Header approximately0.98 inches; footer approximately0.315 inches starting at7.185 inches. Use Times New Roman for all editable text: titles28–32pt, body20–22pt, tables17–20pt, captions15–17pt and references15–16pt. White titles/numbers on blue; dark body text. Keep0.4-inch side margins. Reserve visual areas BEFORE text. Never squeeze a figure or remove axes to fit extra PID text. Put detail in speaker notes. Existing raster charts may retain embedded fonts; redraw faithfully from CSV if every chart label must be Times New Roman. Blue=SpO₂ history; orange=combined.

Personal details must remain placeholders: [Student name], [Registration number], [Department], [Institution], [Guide name], [Presentation date]. Remove instructional text in the template. Use no invented device photos, SHAP plots, loss curves, reference annotations, simulation curves or accuracy claims. Use editable PowerPoint shapes for diagrams. Missing images must have a clearly named placeholder, not a fabricated substitute.

## Scientific scope: do not blur prediction and PID

Title: **Towards Predictive PID-Based Oxygen Control**.
Subtitle: **Evaluating PPG's Incremental Value for Early Desaturation Warning**.
Scope: **Completed prediction study • Proposed PID integration**.

The repository name is Predictive_Oxygen_Controller. Completed work is retrospective prediction/warning evaluation, not demonstrated closed-loop oxygen delivery. PID integration is a proposed extension. Do not claim an implemented or validated PID loop, tuned gains, actuator performance, improved control stability, reduced physical delivery delay, clinical benefit or deployment readiness. Do not equate the94% research threshold with a clinical control target.

PID background: error `e(t)=r(t)−y(t)`; ideal law `u(t)=Kp e(t)+Ki∫₀ᵗe(τ)dτ+Kd de(t)/dt`. Explain proportional/current error, integral/accumulated error and derivative/error change. This conceptual law is not an implementation result. Practical control needs discrete-time sampling, derivative filtering, constraints, anti-windup and a justified plant model. No gains are supplied. Prediction scores are not oxygen commands. No prediction-to-control rule has been defined or tested here.

## Evidence and immutable numerical facts

Primary source: `experiments/final_warning_leadtime_20261028T005900_IST/REPORT.md`, plus its event/recording/paired summaries, figures, limitations and manifest. Supporting methods: `experiments/incremental_value_20261027/REPORT.md`; independent audit: `experiments/final_audit_20261027T215012_IST/REPORT.md`. Do not mix results from older policies. See `IMAGE_PLACEMENT.md` for exact assets.

-30 existing local BIDMC recordings, approximately8min each;14,400s total source duration. PPG125Hz; numeric SpO₂ approximately1Hz. Not verified as30 independent patients.
-Target: minimum valid observed SpO₂≤94% in `(t,t+30s]`. This is a monitor-derived reference, not adjudicated arterial oxygen truth.
-22 low episodes;7 left-censored/unknown-prior;15 qualifying onsets;1 lacks complete30s lead-in. Original eligible events14 across6 recordings. Sustained truth requires≥3 consecutive low observations spanning≥2s:13 eligible events across5 recordings. It removes record7/onset367.
-Inputs: history7 features, PPG10, combined17. Final timing comparison is history versus combined. Preceding PPG-only comparison detected0/14 original events for each tested classifier under the specified policy.
-Models: Logistic, GaussianNB, ELM; fixed existing fits. Outer leave-one-recording-out; training-only grouped3-fold inner evaluation; training-only imputation/scaling; training-OOF fold-specific thresholds maximize event detections subject to≤5 off-event starts/evaluable hour and≤5% active fraction. These are research constraints, not clinical standards. Do not use the older timestamp-recall threshold policy.
-Match chronological events to earliest unused ACTUAL alarm-episode start in `[onset−30,onset)`, one-to-one, no reuse, no clipped/restarted alarms at event boundaries. Misses have undefined lead, not zero. Existing pre-onset matching does not measure post-onset response latency.
-Lead `L=onset−warning`; paired difference `ΔL=L_combined−L_history` only when both detect. Positive difference means combined earlier.
-Evaluable current-normal exposure10,329.032s=2.869176h across26 recordings;4 zero-exposure recordings remain in tables. Do not use4 source hours as false-alarm denominator. Future completeness is retrospective evaluability, not an online future-aware gate.
-Uncertainty:2,000 whole-recording bootstrap draws, seed20261027; shared multiplicities; conditional on saved overlapping CV fits, no model refits or independent-patient inference.

| Family | Original detection history→combined | Sustained detection | Original≥10s | Sustained≥10s |
|---|---|---|---|---|
| Logistic |9/14→10/14|9/13→9/13|6/14→6/14|6/13→5/13|
| GaussianNB |10/14→6/14|10/13→6/13|6/14→4/14|6/13→4/13|
| ELM |6/14→1/14|6/13→1/13|3/14→1/14|3/13→1/13|

Logistic paired timing:7 events/4 recordings;2 earlier,3 equal,2 later. Median additional lead0.000s,95% interval[−4.000,+10.064]s. Mean+3.808s,[−4.000,+10.219]s approximately. Same paired subset under both definitions. This conditional subset cannot establish overall superiority.

Logistic original gains:7:367,12:210,26:144; losses30:72,30:98. Not the same9 plus1. Record26/onset310: history lead2.368s, combined24.312s, delta+21.944s; warning timestamps307.632s and285.688s. Record30/onset72: history lead17.136s (warning54.864s), combined no qualifying warning; no numerical combined lead.

Logistic original: misses5→4; detected<10s3→4; missed OR<10s8/14 each. Sustained: misses4 each; detected<10s3→4; missed OR<10s7/13→8/13.

Logistic original alarm burden: history10 off-event starts/2.869176h=3.485/h; combined14=4.879/h. Total alarm starts29→33. Active duration140.248→98.568s; fractions1.358%→0.954%. Both pooled-original budgets pass, but individual recording constraints fail. Start-budget violations3 recordings each; activity-cap violations4 history/3 combined (overlapping counts, not additive). Record29 has6.264s exposure and100% activity in both. Combined sustained false starts5.577/h exceed pooled budget. Off-event false alarms differ from unmatched redundant starts near true events.

Verification:6/6 new focused tests passed;67 existing relevant tests run,66 passed and1 missing-artifact error (`results/final/future_low/causal_dataset.csv`, not needed by this analysis). Do not claim full-suite success.12 outputs reproduced byte-identically;5,052 pre-existing files unchanged in that analysis. No human pulse references yet;60-window review prepared but pending actual reviewers. No pulse-detection accuracy can be claimed.

## Required slides, visual allocation and speaker notes

###1 Title
Use title/subtitle/scope above, personal placeholders. Small block motif: measurements→prediction→warning evaluation. Notes: original motivation is PID oxygen control; completed work evaluates the prediction subsystem.

###2 Agenda
Five groups: control motivation; predictive question; data/models/evaluation; timing/coverage/burden; proposed integration/limitations. Five editable roadmap blocks.

###3 Introduction: feedback and anticipation
Four bullets: PID responds to measured tracking error; prediction may anticipate deterioration; proposed architecture combines feedback and supervision; present study tests predictive information. Visual: adjacent Feedback and Prediction blocks→Proposed supervised control framework. Caption: combined control not tested. Notes: prediction estimates an outcome; control selects an action; earlier warnings alone cannot establish control improvement. Cite[3–5,8].

###4 Problem and motivation
Question: Does adding PPG provide earlier useful warnings than SpO₂ history alone? Three pillars: detection coverage, warning lead, alarm burden. Supporting line: tests whether PPG justifies additional predictive-supervisor complexity. Notes: timestamp accuracy can obscure rare events and continuous alerting. Never say PPG improves PID.

###5 Literature survey
Table with Pimentel2017(dataset/PPG respiratory context), Allen2007(PPG physiology), Lundberg2018(different-setting hypoxaemia prediction), Fawzy2022(measurement discrepancies). Third column explicitly states each limitation; none validates this model. Bottom: matched-data incremental-information evaluation is the project question. Notes: informed by multiple sources, not replication of one post2020 paper; no global-first claim. PID textbook is background[8].

###6 Aim and objectives
Aim: evaluate predictive subsystem for proposed PID framework. Completed objectives: matched inputs/models; detection and≥10s lead; paired differences and recording uncertainty. Distinct dashed box: future closed-loop PID integration. Label proposed/not evaluated.

###7 Dataset and setup
Left: source30 recordings;~4h;125Hz PPG;~1Hz numerics;30s horizon. Right editable funnel:22 episodes→15 qualifying→14 original eligible→13 sustained. Side exclusions7 unknown-prior,1 incomplete lead-in,1 isolated. Notes:14 events/6 records vs13/5; no patient independence; source vs evaluable hours. Cite[1–3,9,10].

###8 System architecture: evaluated prediction and proposed control
Reserve almost all body area for diagram. Upper solid-border section: PPG→causal features and SpO₂→past history→predictor→warning. Separate future-reference branch→evaluation ONLY, no future data into inference.
Lower dashed section: target r→summing junction(r−y)→PID→constrained actuator command→oxygen actuator/patient model→measured y→feedback. Warning dashed→predictive supervisor dashed→coordination interface; integration rule NOT YET DEFINED. Label lower section proposed/not tested. Legend solid=evaluated,dashed=proposed. No direct warning-to-oxygen command. Notes: supervisor could influence a future design but no mechanism is selected here.

###9 Signal processing and target mathematics
Two brief feature branches plus equation `y(t)=1[min_{t<u≤t+30}SpO₂(u)≤94]`. Small past/future timeline. PPG:causal0.5–5Hz Butterworth; history:past-only10/30s summaries. Notes:prototype order3,digital band-pass order6,not zero-phase; fixed0.03 detector amplitude in exported units,minimum spacing0.30s,confirmation8ms. `IBI=peak_i−peak_(i−1)`,rate60/IBI. PPG names:PPG_Amplitude,Pulse_Interval_s,Pulse_Rate_BPM,Amplitude_Slope,Rate_Mean_5,Rate_Std_10,Rate_Change_5,Amplitude_Mean_5,Amplitude_Std_10,Interval_Std_10. Rolling suffixes are pulse counts,not seconds. History:current,mean10/30,OLS trend10/30,sample SD30,latest measurement age. OLS slope=Σ[(t−mean t)(s−mean s)]/Σ(t−mean t)². Prediction target is not an actuator command.

###10 Models and choices
Three-row table:Logistic transparent linear reference(C1,balanced);GaussianNB Gaussian class likelihood/conditional independence(priors0.5/0.5,smoothing1e−6);ELM random tanh hidden features/ridge output(50 units,λ1,positive weight5,seed42). Footer:predictors,not PID controllers;7/10/17 inputs. Notes:logisticσ(β₀+βᵀz);GNB normalized prior×product of Gaussian likelihoods;ELM H=tanh(ZW+b),β=(HᵀDH+λI)⁻¹HᵀDy,score=hβ. Scores not calibrated risks. Impute training medians;z=(imputed x−training mean)/training SD. No new classifier sweep. Cite[7,10].

###11 Implementation and fair evaluation
Main visual editable flowchart:hold out recording→inner grouped training→train-only transforms/model→training OOF scores→threshold under both constraints→held-out saved scores→actual alarm episodes→one-to-one matching→paired metrics/cluster bootstrap. Side note:describes production of existing saved predictions;final study does not retrain. Small conceptual PID equation box only;put full explanation in notes to preserve flowchart space. Notes:PID ideal law and parameter definitions;practical filtering/anti-windup needed;no gains or tested loop. Include lead/delta equations and threshold optimization objective. Cite[8–10].

###12 Detection results
Editable three-row table of original and sustained detection counts above. Takeaway:only logistic small original gain,absent under sustained truth;other families lose coverage. Small note:preceding PPG-only0/14. Notes:retain event swaps;no generic accuracy or classifier-winner claim.

###13 Does PPG warn earlier?
Reserve65–70% of content height for IMAGE A top-left Logistic/original panel,including axes,legend and Missed band. Below:7 shared/4 records;2 earlier/3 equal/2 later;median0s,CI−4 to+10.064. Notes:blue history/orange combined;crosses are separate misses,not negative/zero lead;conditional paired subset;mean+3.808 with interval crossingzero;warning lead is not actuator response.

###14 At least10 seconds warning
Reserve approximately65% for IMAGE B top row,both Logistic panels,original left/sustained right;preserve shading/axes/10s marker. Table:original6/14 each;sustained6/13 vs5/13. Takeaway:no original gain,worse sustained coverage. Notes:all eligible events denominator;research benchmark,not established oxygen-response requirement;missed or insufficient8/14 each vs7/13→8/13;whole-record conditional intervals.

###15 Alarm burden
Table original logistic:3.485→4.879 off-event/h;29→33 starts;140.248→98.568 active seconds. Two callouts:3 start-budget violations each;4→3 activity violations. Footer:combined sustained5.577/h fails pooled budget. Optional IMAGE E only if readable;otherwise retain table. Notes:2.869176h denominator;overlapping violations not additive;record29short100%active;research budgets not safety/stability standards.

###16 Actual gain and actual loss
Two columns. Left26:310:2.368→24.312s,+21.944. Right30:72:17.136s→missed. Use editable onset-relative timelines and/or IMAGES C/D preserving axes. If figures too dense,keep timelines on slide/full figures in notes. Caption:event gains coexist with losses. Notes:warning times307.632/285.688 and54.864s;no fabricated missing lead;examples do not identify physiological cause or establish improved PID performance.

###17 Applications and contributions
Left demonstrated:causal pipeline,matched input ablation,event-level timing/burden,reproducible uncertainty. Right proposed:predictive supervision of PID,conventional-PID comparator,delay-aware constrained simulation. Novelty:auditable evaluation and project-specific evidence,not a new classifier or proven controller. Notes:no first-ever/patent claim;publication possible only with justified scope and stronger reference/generalization evidence,no acceptance promise.

###18 Conclusion and next validation
Conclusion:PPG additional useful-warning benefit not established. Status:prediction evaluation complete;PID integration not established. Next:real human references/clinical review→justified plant model→prespecified supervision/PID→closed-loop comparison. Small diagram conventional PID versus PID+history prediction versus PID+combined prediction. Notes:match disturbances/delays/initial conditions/constraints/tuning;measure IAE=∫|e|dt,overshoot,settling,threshold-excursion duration/severity,control effort,saturation;anti-windup/filtering. No fabricated plant/gains/curves;simulation≠clinical proof. Retain6newpasses/12reproducedoutputs and67existing66pass1error;no detectoraccuracy beforehumanreferences. Predictionlead≠alarmresponse≠deliverylatency≠clinicalbenefit.

###19 References
Use concise IEEE-style references below,in two readable columns. Full citations/URLs in notes;internal reports labelled internal,not peer reviewed. General scikit-learn citation(Pedregosa etal.,JMLR12,2825–2830,2011) may appear in notes in addition. No invented citations.

###20 Thank you / Questions
Takeaway:Validate the predictive information first;then test whether it improves closed-loop control. Notes:be ready to explain zero median,undefined misses,10s denominators,conditional uncertainty,budgetviolations and unmeasured delivery response.

## References

[1] M.A.F.Pimentel etal.,“Toward a Robust Estimation of Respiratory Rate From Pulse Oximeters,”IEEE Trans.Biomed.Eng.,64(8),1914–1926,2017.doi:10.1109/TBME.2016.2613124.
[2] BIDMC PPG and Respiration Dataset,PhysioNet,v1.0.0,2018.doi:10.13026/C2208R.https://physionet.org/content/bidmc/1.0.0/.
[3] A.L.Goldberger etal.,“PhysioBank,PhysioToolkit,and PhysioNet:Components of a New Research Resource for Complex Physiologic Signals,”Circulation101(23),e215–e220,2000.doi:10.1161/01.CIR.101.23.e215.
[4] J.Allen,“Photoplethysmography and its application in clinical physiological measurement,”Physiol.Meas.28(3),R1–R39,2007.doi:10.1088/0967-3334/28/3/R01.
[5] S.M.Lundberg etal.,“An explainable machine-learning algorithm for hypoxaemia prediction during surgery,”Nature Biomedical Engineering2,749–760,2018.doi:10.1038/s41551-018-0304-0.
[6] A.Fawzy etal.,“Racial and Ethnic Discrepancy in Pulse Oximetry and Delayed Identification of Treatment Eligibility Among Patients With COVID-19,”JAMA Internal Medicine182(7),730–738,2022.doi:10.1001/jamainternmed.2022.1906.Measurement limitations,not proof of incremental PPG prediction.
[7] G.-B.Huang,Q.-Y.Zhu,and C.-K.Siew,“Extreme learning machine:Theory and applications,”Neurocomputing70(1–3),489–501,2006.doi:10.1016/j.neucom.2005.12.126.
[8] K.J.Åström and T.Hägglund,Advanced PID Control.ISA,2006.Background theory,not project validation.
[9] Predictive_Oxygen_Controller,“Does Adding PPG Reduce Desaturation Warning Delay?”,internal report,28Oct2026,experiments/final_warning_leadtime_20261028T005900_IST/REPORT.md.
[10] Predictive_Oxygen_Controller,“Matched SpO₂-history / PPG / combined incremental-information experiment,”internal report,Oct2026,experiments/incremental_value_20261027/REPORT.md.

## Final checks and defence notes

Exactly20slides;template4:3;white/blue/TNR;allvisualslegible;editable diagrams;source captions;all counts andmissingvalues accurate;bothtruthdefinitions separate;no invented PID/clinicalresults. Do not substitute the both-detected subset for all events. Humanannotationspending. State results honestly:negative/mixed evidence can be a strong bounded research contribution. Do not generalize “PPG is useless.” The project is not a direct replication of one recent paper. Recent measurement literature informs limitations. A publication requires appropriate scope,referencequality andvalidation;acceptance cannot be promised. Keep models/results unchanged. To demonstrate PID as a core contribution,additional implemented closed-loop work is needed;architecture alone does not satisfy that objective.
