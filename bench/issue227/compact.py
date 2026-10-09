#!/usr/bin/env python3
"""Retain raw samples, decisions and provenance without execution logs."""
import argparse
import csv
import json
from pathlib import Path


def main(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'timings.csv').open('w') as timing, (args.output/'decisions.jsonl').open('w') as decision, \
         (args.output/'provenance.jsonl').open('w') as provenance:
        writer=csv.writer(timing,lineterminator='\n')
        writer.writerow(['run','case','shape','split','path','phase','p50_ms','p95_ms','samples_ms'])
        def distribution(run,case,shape,split,path,phase,value):
            writer.writerow([run,case,'x'.join(map(str,shape)),split,path,phase,
                round(value['p50_ms'],6),round(value['p95_ms'],6),
                json.dumps([round(x,6) for x in value['samples_ms']],separators=(',',':'))])
        for file in sorted(args.input.glob('*.json')):
            if file.name.endswith('.profile.json'):continue
            result=json.loads(file.read_text())
            run=file.stem
            provenance.write(json.dumps(dict(run=run,**{k:v for k,v in result.items()
                if k in ('hardware','metadata','configuration','options','limits','round','protocol',
                    'benchmark_sha256','isolation','source_revision','scenario','controlled_load')}),separators=(',',':'))+'\n')
            if run.startswith('decode'):
                for case,row in result['cases'].items():
                    for path,value in row['paths'].items():
                        for phase in ('end_to_end','transfer','kv_roundtrips'):
                            dist=value if phase=='end_to_end' else value.get(phase)
                            if dist:distribution(run,case,[],'model',path,phase,dist)
                        record=dict(run=run,case=case,path=path,
                            configuration={k:v for k,v in row['configuration'].items() if k!='preparation'},
                            preparation=row['configuration'].get('preparation',{}).get(path),
                            first_completed_ms=value['first_completed_ms'],
                            first_generated_kernels=value['first_generated_kernels'],
                            first_compile_ms=value['first_stats']['compile_ms'],
                            warm_compile_callbacks=value['warm_compile_callbacks'],max_abs_error=value['max_abs_error'],
                            backend=value['warm_stats']['backend'])
                        decision.write(json.dumps(record,separators=(',',':'))+'\n')
            elif 'cases' in result:
                for case,rows in result['cases'].items():
                    if not isinstance(rows,list):continue
                    for row in rows:
                        for path,value in row['paths'].items():
                            path,phase=path.rsplit('_',1)
                            distribution(run,case,row['shape'],row['split'],path,phase,value)
                        for path,value in row.get('controls',{}).items():
                            distribution(run,case,row['shape'],row['split'],path,'warm',value)
                        if 'selection_only' in row:
                            # Keep the distribution, but 100 selector samples add little
                            # evidence beyond the requested raw transfer measurements.
                            dist=row['selection_only']
                            record={k:row[k] for k in ('shape','decision','automatic_cold','regret','path_regret')}
                            record.update(run=run,case=case,selection_only={k:dist[k] for k in ('p50_ms','p95_ms')})
                            decision.write(json.dumps(record,separators=(',',':'))+'\n')
            else:
                for key in ('foreign_profile','qualified_profile'):
                    if key not in result:continue
                    row=result[key]
                    distribution(run,'matrix_cast_h2d',[128,512],'context','automatic',key,row)
                    for path,value in row.get('forced',{}).items():
                        distribution(run,'matrix_cast_h2d',[128,512],'context',path,'warm',value)
                    decision.write(json.dumps(dict(run=run,phase=key,decision=row['decision']),separators=(',',':'))+'\n')
                for row in result.get('rows',[]):
                    distribution(run,'matrix_cast_h2d',row['shape'],'context_train',row['path'],row['state'],row)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
