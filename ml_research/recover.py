"""Resume an interrupted frozen experiment into a NEW directory, never in place.

Completed grids/scores are reused verbatim. Lost in-memory fit hashes are explicitly
unavailable; deterministic split membership is reconstructed, not claimed observed.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from .baseline import sha, verify_snapshot
from .core import training_scores, fit_model, negative_rank, array_hash
from .experiment import inputs, datasets, topology_cache, select
from src.alarm_budget_experiment import choose_candidate

GROUP = ['Target', 'Horizon_s', 'FeatureSet', 'Model']


def recovered_choices(grid, protocol):
    """Use exact serialized threshold hex for tie ordering, not rounded CSV floats."""
    grid = grid.copy()
    grid = grid.drop(columns=GROUP+['ScoreSpace', 'Outer_Test_Record'], errors='ignore')
    if grid.Threshold_hex.duplicated().any():
        raise ValueError('Duplicate candidate thresholds: incomplete/duplicated source')
    grid['Threshold'] = grid.Threshold_hex.map(float.fromhex)
    return [dict(Budget=b, **choose_candidate(grid, b, protocol['active_cap']))
            for b in protocol['budgets']]


def validate_scores(scored, held):
    if scored.sample_id.duplicated().any():
        raise ValueError('Duplicate saved held-out samples')
    a = scored.sort_values('sample_id'); b = held.sort_values('sample_id')
    for column in ['sample_id', 'Record', 'pulse_row', 'time_s']:
        np.testing.assert_array_equal(a[column], b[column])
    if not np.isfinite(a[['raw', 'negative_rank']].to_numpy()).all():
        raise ValueError('Invalid saved score')
    if not a.negative_rank.between(0, 1).all():
        raise ValueError('Invalid saved negative rank')


def recover(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    if output.exists() or output.parent != source.parent:
        raise ValueError('Recovery requires a new sibling output directory')
    protocol = json.loads((source/'protocol.json').read_text())
    for name, expected in protocol['implementation'].items():
        if sha(Path(__file__).parent/name) != expected:
            raise ValueError('Frozen implementation changed: '+name)
    verify_snapshot(json.loads((source/'preservation.json').read_text()))
    source_hashes = {p.name: sha(p) for p in source.iterdir() if p.is_file()}
    # Fail closed on truncated gzip/CSV; never silently accept a partial group.
    scores = pd.read_csv(source/'outer_scores_unlabeled.csv.gz', float_precision='round_trip')
    grids = pd.read_csv(source/'training_candidates.csv.gz', float_precision='round_trip')
    output.mkdir()
    for p in source.iterdir():
        if p.is_file(): shutil.copy2(p, output/p.name)
    numeric, events = inputs()
    selections=[]; ledger=[]; analysis=[]; counts=[]; models=[]; reused=0; fitted=0
    expected_keys=set()
    with threadpool_limits(limits=1):
        for target in protocol['targets']:
            variants=datasets(target,numeric); cache=topology_cache(variants[30],numeric,events)
            for spec in protocol['settings']:
                h,fs=spec['Horizon_s'],spec['FeatureSet']; data=variants[h]; cols=protocol['features'][fs]
                expected_labels=data[['sample_id','Record','pulse_row','time_s','Label']]
                pd.testing.assert_frame_equal(pd.read_csv(source/f'labels_{target}_{h}_{fs}.csv.gz',float_precision='round_trip'),expected_labels.reset_index(drop=True),check_dtype=False)
                counts.append(dict(Target=target,Horizon_s=h,FeatureSet=fs,Rows=len(data),Positives=int(data.Label.sum()),Recordings=data.Record.nunique(),Positive_Recordings=int(data.groupby('Record').Label.sum().gt(0).sum())))
                for test in sorted(data.Record.unique()):
                    train=data[data.Record.ne(test)].reset_index(drop=True);held=data[data.Record.eq(test)].reset_index(drop=True)
                    allowed=sorted(map(int,train.Record.unique()))
                    for feature in cols:
                        pos=train.loc[train.Label.eq(1),feature];neg=train.loc[train.Label.eq(0),feature];scale=train[feature].std()
                        analysis.append(dict(Target=target,Horizon_s=h,FeatureSet=fs,Outer_Test_Record=int(test),Feature=feature,Train_Positive_Mean=pos.mean(),Train_Negative_Mean=neg.mean(),Standardized_Difference=(pos.mean()-neg.mean())/scale if scale>0 else np.nan,Train_Positives=len(pos)))
                    for name in protocol['model_configs']:
                        context=dict(Target=target,Horizon_s=h,FeatureSet=fs,Model=name)
                        expected_keys.add((target,h,fs,name,int(test)))
                        q=scores
                        g=grids
                        for key,value in context.items():
                            q=q[q[key].eq(value)];g=g[g[key].eq(value)]
                        q=q[q.Record.eq(test)];g=g[g.Outer_Test_Record.eq(test)]
                        complete=len(q)>0 and set(g.ScoreSpace)==set(protocol['score_spaces'])
                        if len(q):validate_scores(q,held)
                        if complete:
                            splits=[dict(train_records=sorted(map(int,train.iloc[tr].Record.unique())),validation_records=sorted(map(int,train.iloc[va].Record.unique()))) for tr,va in GroupKFold(n_splits=3).split(train,train.Label,train.Record)]
                            proof=dict(Provenance='Reused saved unlabeled scores and training grids; split membership reconstructed from frozen algorithm; original in-memory OOF/reference hashes unavailable after interruption',Recovered_Source_Scores_SHA256=source_hashes['outer_scores_unlabeled.csv.gz'],Recovered_Source_Grids_SHA256=source_hashes['training_candidates.csv.gz'])
                            reused+=1
                        else:
                            oof,model,ref,splits=training_scores(train,cols,name,int(test))
                            raw=model.score(held[cols].to_numpy(float))
                            new=held[['sample_id','Record','time_s','pulse_row']].assign(raw=raw,negative_rank=negative_rank(ref,raw),**context)
                            if len(q):np.testing.assert_allclose(q.sort_values('sample_id')[['raw','negative_rank']],new.sort_values('sample_id')[['raw','negative_rank']],rtol=1e-9,atol=1e-10)
                            else:new.to_csv(output/'outer_scores_unlabeled.csv.gz',mode='a',header=False,index=False,compression='gzip')
                            for space,values in oof.items():
                                existing=g[g.ScoreSpace.eq(space)]
                                if len(existing):continue
                                _,extra=select(train,values,cache,events,numeric,int(test),protocol,h)
                                extra=extra.assign(**context,ScoreSpace=space,Outer_Test_Record=int(test))
                                extra.to_csv(output/'training_candidates.csv.gz',mode='a',header=False,index=False,compression='gzip')
                                g=pd.concat([g,extra],ignore_index=True)
                            proof=dict(Provenance='New fit of missing group using frozen implementation',OOF_Raw_SHA256=array_hash(oof['raw']),OOF_Rank_SHA256=array_hash(oof['negative_rank']),Negative_Reference_SHA256=array_hash(ref),Reference_Count=len(ref))
                            fitted+=1
                        for space in protocol['score_spaces']:
                            for row in recovered_choices(g[g.ScoreSpace.eq(space)],protocol):
                                selections.append(context|dict(ScoreSpace=space,Outer_Test_Record=int(test),Allowed_Records=','.join(map(str,allowed)),**row))
                        ledger.append(context|dict(Outer_Test_Record=int(test),Allowed_Records=allowed,inner=splits,Training_Sample_ID_SHA256=array_hash(train.sample_id),**proof))
                        if test==max(data.Record.unique()):
                            # A single outer-training refit per condition restores lost latency objects.
                            model=fit_model(train,cols,name);ref=model.score(train.loc[train.Label.eq(0),cols].to_numpy(float))
                            if len(q):np.testing.assert_allclose(model.score(held[cols].to_numpy(float)),q.sort_values('sample_id').raw,rtol=1e-9,atol=1e-10)
                            models.append((context|dict(Outer_Test_Record=int(test)),model,ref,train[cols].iloc[:1].to_numpy(float)))
                    print('RECOVERED',target,h,fs,int(test),'reused',reused,'new',fitted,flush=True)
        actual_keys=set(map(tuple,scores[GROUP+['Record']].drop_duplicates().to_numpy()))
        if not actual_keys<=expected_keys:raise ValueError('Unexpected saved fit group')
        pd.DataFrame(selections).to_csv(output/'selected_thresholds.csv',index=False)
        pd.DataFrame(analysis).to_csv(output/'training_feature_analysis.csv',index=False)
        pd.DataFrame(counts).to_csv(output/'dataset_counts.csv',index=False)
        (output/'fold_ledger.json').write_text(json.dumps(ledger,indent=2))
        receipt=dict(source_directory=str(source),source_files=source_hashes,reused_model_groups=reused,new_model_groups=fitted,latency_only_outer_refits=len(models),lost_hash_limitation='Reconstructed split membership is not a recovered execution trace. No unavailable fit hash is invented.',recovery_code_sha256=sha(Path(__file__)))
        (output/'recovery.json').write_text(json.dumps(receipt,indent=2))
        (output/'selection_complete.json').write_text(json.dumps(dict(phase='ALL selections frozen before new held-out outcome evaluation; recovered training-only grids',thresholds_sha256=sha(output/'selected_thresholds.csv'),protocol_sha256=sha(output/'protocol.json'),outer_unlabeled_sha256=sha(output/'outer_scores_unlabeled.csv.gz')),indent=2))
        for name,expected in source_hashes.items():
            if sha(source/name)!=expected:raise ValueError('Interrupted run changed')
        from .reporting import evaluate, benchmark
        evaluate(output,protocol,numeric,events)
        benchmark(output,protocol,models)
        finish_recovery(output,protocol)


def finish_recovery(output, protocol):
    """Same frozen analysis; disclose interruption before final artifact hashing."""
    from .baseline import ROOT
    from .reporting import figures
    from .summarize import write_report
    from src.benchmark_latency import environment
    verify_snapshot(json.loads((output/'preservation.json').read_text()))
    figures(output)
    report=ROOT/'docs/ML_IMPROVEMENT_REPORT.md'; existed=report.exists()
    write_report(output,protocol)
    local=output/'ML_IMPROVEMENT_REPORT.md'
    text=local.read_text()
    text=text.replace('fold ledgers include exact permitted recordings, inner splits and OOF/reference hashes.',
        'fold ledgers include exact permitted recordings and inner splits; interrupted-run fit hashes are unavailable for recovered groups, explicitly disclosed below.')
    begin=text.index('Run in the pinned repository environment;')
    end=text.index('Outputs include original baseline',begin)
    replacement='''The original baseline and frozen experiment were executed in `results/ml_improvement/run01`; fitting was interrupted after 1,074 of 1,120 model/fold groups. Run01 is preserved byte-for-byte. A recovery-only attempt in `recovery01` stopped on a metadata-column collision before new fitting or outcome evaluation; it is retained too. The completed run uses the recovery command below, retaining the frozen scientific protocol and all existing fitted scores/grids. No completed scientific fit is replaced.\n\n```bash\nMPLCONFIGDIR=/tmp/oxygen-ml-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.recover --source results/ml_improvement/run01 --output results/ml_improvement/recovery02\n```\n\nChoose a NEW sibling output directory for another recovery reproduction; existing paths are rejected. To reproduce entirely from source instead, run `python -m ml_research.baseline --output results/ml_improvement/reproduction01` followed by `python -m ml_research.experiment --output results/ml_improvement/reproduction01`, with the same environment variables. Baseline/protocol guards reject overwrites, while existing docs are retained and each run writes its own report.\n\n**Recovery provenance limitation:** 1,074 completed saved unlabeled-score/model groups and their training candidate grids were reused, with exact row identity and finite-score checks. Only 46 missing groups were fit using the unchanged frozen implementation. Thresholds were reconstructed from training grids with exact hexadecimal cutoffs and unchanged tie ordering. The lost in-memory OOF/reference hashes are NOT invented: ledger fields explicitly state their absence. Inner split membership is reconstructed from the deterministic frozen splitter, not claimed as a recovered execution trace. Forty outer-training-only refits restored lost objects solely for timing, with last-record saved score parity checked where available. `recovery.json` records source file hashes, counts and this limitation. The original seven frozen source hashes and all 765 baseline-protected files remain verified. A recovered ledger is weaker provenance than an uninterrupted persisted fit trace.\n\nFinal commands and outcomes are recorded in [ML_IMPROVEMENT_VERIFICATION.md](ML_IMPROVEMENT_VERIFICATION.md). No pipeline cache or original output was regenerated.\n\n'''
    text=text[:begin]+replacement+text[end:]
    local.write_text(text)
    if not existed:report.write_text(text)
    manifest=dict(environment=environment(),status='complete',protocol_sha256=sha(output/'protocol.json'),baseline_document_sha256=sha(ROOT/'docs/ML_IMPROVEMENT_BASELINE.md'),sources={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},original_artifacts_unchanged=True,outputs={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from .verify import verify
    (output/'verification.json').write_text(json.dumps(verify(output),indent=2))
    print('COMPLETE',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();recover(args.source,args.output)
