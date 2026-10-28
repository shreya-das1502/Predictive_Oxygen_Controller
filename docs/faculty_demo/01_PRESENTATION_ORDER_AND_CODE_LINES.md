# What to show, in exact order

All paths below are relative to Predictive_Oxygen_Controller/. The ZIP contains identical source copies beneath `repository/` and readable numbered copies beneath `line_numbered/`. Press Ctrl+G in VS Code to jump to a line. Line ranges below mean which block to show; they are not instructions to execute/retrain it.

## A. Slides and code demonstration

| Order / PPT slide | Repository file | Lines to show | What to say |
|---|---|---|---|
|1 / slides1–6|`experiments/incremental_value_20261027/config.json`|4–24;32–36|Thirty-second future-low target;fixed recording-grouped evaluation;training-only alarm policy;research budgets,not clinical safety limits.|
|2 / slide7,dataset|`experiments/incremental_value_20261027/study.py`|51–56|Read thirty existing numeric recordings.|
|3 / slide7,dataset pipeline|same `study.py`|58–76|Read PLETH,validate waveform clock,extract pulses,add history/labels,retain matched rows.|
|4 / slide9,labels|`src/labeling.py`|6–24|Future interval(t,t+30],≤94% label;incomplete/invalid reference becomes unknown,notnegative. Line14 defines interval;lines16–23 enforce completeness and label.|
|5 / slide9,event definition|same `labeling.py`|27–50|Observed low episodes;valid preceding normal reading required for a qualifying onset. Lines42–49 construct events.|
|6 / slide9,clock integrity|`src/preprocessing.py`|80–93|Validate documented rounded125Hz export before using a row-index clock;do not arbitrarily deduplicate waveform samples.|
|7 / slide9,causal filtering|same `preprocessing.py`|40–57|Line45 designs Butterworth band-pass;line55 uses stateful causal sosfilt. Prototype order3 gives band-pass digitalorder6.|
|8 / slide9,pulse detector|same `preprocessing.py`|16–38|Lines20–23 maximum/amplitude/spacing;25–30 interval,rate and amplitude slope;peak is confirmed using the next sample.8ms isconfirmationdelay,not totalmeasurementlatency.|
|9 / slide9,PPG rollingfeatures|`src/features.py`|7–34|Finite/physical consistencychecks;history≤10pulses;last5means,last10sampleSD;rate change5pulsesback. Pulsecounts,notseconds.|
|10 / slide9,SpO₂ history|`experiments/incremental_value_20261027/study.py`|16–17;22–39|Sevenhistory/tenPPG/seventeencombined;line29 uses at-or-before observationboundary;trailing10/30smeans,trends,SD,andage.|
|11 / slide10,settings|`src/config.py`|14–22|PPG names and fixedparameters;present Logistic,GaussianNB,ELM. Tree exists in genericconfig but isnot a fourth finalcomparisonmodel.|
|12 / slide10,ELM|`src/model.py`|14–28|Randomtanh hiddenfeatures;weightedridgeoutput;fixedseed;notallweights trained andnotcalibratedprobability.|
|13 / slide10,model pipeline|same `model.py`|30–50|Lines34–36 imputer/scaler/classifier;line41 fits transforms ontraining;47–48 transforms inference withoutrefitting and scores.|
|14 / slide11,input selection|`experiments/incremental_value_20261027/run.py`|56–67|Use selected7/10/17features;training-only OOF thresholdselection.|
|15 / slide11,outervalidation|same `run.py`|76–100|85–86 outerrecordseparation;88–92 groupedinnerfolds;94–98 thresholdfrozen BEFORE heldoutscoring;100 actualsaveddecisions.|
|16 / slide11,thresholdobjective|`src/alarm_budget_experiment.py`|73–100|Candidatealarmstatistics;97–100 jointlyenforce bothbudgets then deterministicobjective/ties. Do not confuse this witholdertimestamprecallpolicy.|
|17 / slide11,alarms|`src/reliability_evaluation.py`|90–103|Actualpositiveepisode starts;noreset at eventwindowedge.|
|18 / slide11,eligibility/burden|same `reliability_evaluation.py`|116–138|Eligibleonsetrules;off-eventstarts differfromunmatchedredundantstarts.|
|19 / slide11,matching|same `reliability_evaluation.py`|140–159|148 earliestunusedactualstart in[onset−30,onset);151 record-use;156–157 detectionandlead. MissesremainNaN.|
|20 / slides12–16,recordmetrics|`experiments/incremental_value_20261027/study.py`|102–124|Alternateeventdefinition,normalexposuretopology and unchangedrecordeventevaluation.|
|21 / slide13,uncertainty|`experiments/final_warning_leadtime_20261028T005900_IST/analyze.py`|18–26;113–130|paired_bootstrap repeatswhole-record contributions;shared2000draws and explicitall-eventtimelydenominators.|
|22 / slides13–14,lead arithmetic|same `analyze.py`|94–101|Missingwarningkeptmissing;additionalleaddefined only whenbothdetect.|
|23 / slide18,checks|same `analyze.py`|145–156|Uniqueness,missingsemantics,timestampmath,nonreusedwarnings,countidentities.|
|24 / slide18,tests|`experiments/final_warning_leadtime_20261028T005900_IST/tests/test_analysis.py`|10–33|One-to-oneboundarytests,clusterweights,undefinedpairs,table/plotconsistency.|
|25 / PIDquestiononly|`src/controller.py`|1;8–35|Researchalertwrapper,NOTPID oroxygenactuation. OriginalPPGfeaturepath,not the17featurecombinedinferencepipeline.|

### Short live walkthrough

Do not open25blocks during a10-minpresentation. Present theslides and use only these if a livecode demonstration isrequested:
1. `src/labeling.py`,lines14–23:whatpredictionmeans.
2. `src/preprocessing.py`,lines40–57:causalfilter.
3. `study.py`,lines22–39:past-onlyhistory.
4. `run.py`,lines85–100:recordinggrouping andthresholdbeforetest.
5. `reliability_evaluation.py`,lines140–159:honestwarningcredit.
6. Showthefiveimagesbelow and correspondingCSVresults.

## B. Exact image order

| Order | PPT slide | Filename INSIDE ZIP | Repository original | Main explanation |
|---|---|---|---|---|
|1|13|`images/01_warning_lead_time_comparison.png`|`experiments/final_warning_leadtime_20261028T005900_IST/warning_lead_time_comparison.png`|Showtop-leftLogistic/originalpanel.2earlier/3equal/2later;medianextra0s;missesareseparate. Keepfull6panelimageforquestions.|
|2|14|`images/02_timely_warning_comparison.png`|`experiments/final_warning_leadtime_20261028T005900_IST/timely_warning_comparison.png`|ToprowbothLogisticpanels;≥10soriginal6/14each;sustained6/13historyvs5/13combined.|
|3|15|`images/03_per_record_alarm_constraints.png`|`experiments/final_audit_20261027T215012_IST/figures/per_record_alarm_constraints.png`|Individualrecordbudgetviolations;pooledmeansarenotsafetyproof. Usetableonslideifplottoodense.|
|4|16,left|`images/04_gain_record26_event310.png`|`experiments/final_audit_20261027T215012_IST/figures/event_windows/record26_event310.png`|History2.368s,combined24.312s;gain21.944s.|
|5|16,right|`images/05_loss_record30_event72.png`|`experiments/final_audit_20261027T215012_IST/figures/event_windows/record30_event72.png`|History17.136s,combinedmissed;neverassignzero lead.|

Crop only namedpanels whilekeepingaxes/legend/Missedband. Do notstretch. Ifplotsillegible,usethePPTeditabletimelines/fullfiguresinquestions. Diagramsshowarchitecture,notexperimentalPIDresults. NonewPIDresponseimageexists.

## C. Saved tables to open

All beneath `experiments/final_warning_leadtime_20261028T005900_IST/` inrepositoryand`repository/`snapshot:
-`overall_summary.csv`:slide12detectionandpooledburden.
-`paired_lead_time_summary.csv`:slide13conditionaltimingandintervals.
-`timely_warning_summary.csv`:slide14all-eventleadrequirements.
-`recording_level_summary.csv`:slide15individualburden/exposure.
-`event_level_lead_times.csv`:slide16actualeventtimes/status.
-`REPORT.md`:fullmethods/results/limitations.
-`manifest.json`:provenanceandverification.

For CSVs,filterbyModel,DefinitionandModality;do notreadanELMrowasLogistic. CSVblankmeansundefined,notzero. Useoriginalandsustainedseparately.

## D. Before presenting

-OpenPPTandreadscope:predictioncompleted,PIDproposed.
-Checkcitationnumbers:slide3mentions[11],notlistedinthevisiblebibliography;PIDtextbookis[8]. Internalreports[9]/[10]needbibliography/notesentries. UploadedPPTretainedunchanged.
-Know30recordings≠verified30patients;20150feature rows≠events.
-Knowfuture-lowlabelsincludeongoinglows;earlywarningevaluationusescurrent-normalopportunities.
-Know30s horizon≠guaranteed30s lead;10srequirementisresearchbenchmark.
-Knoworiginal14events/6records;sustained13/5;logisticpaired7events/4records.
-Know0medianadditionalleadandnogaininoriginal≥10scoverage.
-Knowhumanpulsereferencespending;donotclaimdetectorprecision/recall.
-Do notruntrainingduringdemo. Newfocusedtests6pass;existingrelevant67run/66pass/1missinghistoricalartifacterror,notfullsuitegreen.
-Do notshowcredentials/configurationdirectories(.aws,etc.).
