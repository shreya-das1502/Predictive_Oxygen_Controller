# Presentation images and layout

All paths are repository-relative. Existing figures are derived from observed signals or saved predictions, not generated illustrations. Upload these files and the template alongside GENERATION_PROMPT.md to a presentation generator.

| Slide | File | Placement |
|---|---|---|
|13|`experiments/final_warning_leadtime_20261028T005900_IST/warning_lead_time_comparison.png`|Top-left Logistic/original panel;65–70% of content area;retain axes,legend,Missed band.|
|14|`experiments/final_warning_leadtime_20261028T005900_IST/timely_warning_comparison.png`|Top row,both Logistic panels;approximately65% of content height;retain uncertainty shading and axes.|
|16,left|`experiments/final_audit_20261027T215012_IST/figures/event_windows/record26_event310.png`|Gain example:history2.368s,combined24.312s. If unreadable in half-slide,use editable timeline and full figure in notes.|
|16,right|`experiments/final_audit_20261027T215012_IST/figures/event_windows/record30_event72.png`|Loss example:history17.136s,combinedmissed. Never assign a numerical missed lead.|
|15,optional|`experiments/final_audit_20261027T215012_IST/figures/per_record_alarm_constraints.png`|Supporting individual-recording burden evidence;prefer the verified table if figure too dense.|

No stretching or artificial result plots. A generator may faithfully replot supplied CSVs with Times New Roman to improve slide legibility. Leave a named placeholder if an asset is unavailable.

Create these as editable PowerPoint shapes,not invented experiment outputs:slide3feedback/prediction diagram;slide4metric pillars;slide7eventfunnel;slide8evaluated prediction/proposedPID architecture;slide9past/futuretimeline;slide11evaluationflowchart;slide16relative warningtimelines;slide18proposedcontrollercomparison. Layout specifications are in GENERATION_PROMPT.md. Detailed PID mathematics stays in notes to preserve slide11flowchart space.

`presentation_assets.zip` bundles the five unchanged PNGs,template,prompt,thisguide andPIDscope. It does not claim newly rendered diagrams or a completed PPT.
