"""Declared membership and provenance for consumer layout qualification."""
from pathlib import Path
from bench.issue268.contract import CPUS, environment, expected_environment, save, sha
from bench.issue268.contract import provenance as previous_provenance

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('native', 'materialized', 'lowered')
CASES = {'decode8': (8,4), 'decode64': (64,4), 'prefill256': (256,1)}
MICROS = (
    dict(name='canonical', length=8, heads=16, dim=64, visible=False, budget_mib=None),
    dict(name='held_small', length=31, heads=4, dim=32, visible=False, budget_mib=None),
    dict(name='held_long', length=4096, heads=16, dim=64, visible=False, budget_mib=None),
    dict(name='mixed', length=512, heads=16, dim=64, visible=True, budget_mib=None),
    dict(name='pressure', length=8192, heads=16, dim=64, visible=False, budget_mib=128),
)


def jobs():
    result = []
    for round_ in (1,2,3):
        for replay in (False,True):
            for case in (('decode8',) if replay else tuple(CASES)):
                for compute in ('eager','inductor'):
                    cell = f'model-r{round_}-{case}-{compute}-'+('replay' if replay else 'plain')
                    modes = VARIANTS[round_%3:] + VARIANTS[:round_%3]
                    runtimes = ('torch','sym') if round_%2 else ('sym','torch')
                    arms = [(rt,mode) for mode in modes for rt in runtimes]
                    if not replay: arms.insert(round_%len(arms),('torch','resident'))
                    for runtime,variant in arms:
                        result.append(dict(kind='model',round=round_,case=case,compute=compute,
                            replay=replay,runtime=runtime,variant=variant,callers=1,
                            cell=cell,id=cell+'-'+runtime+'-'+variant))
        for compute in ('eager','inductor'):
            cell=f'micro-r{round_}-{compute}'
            result.append(dict(kind='micro',round=round_,case='supplement',compute=compute,
                replay=False,runtime='shared',variant='all',callers=1,cell=cell,id=cell))
    return result


def protocol(job):
    return dict(dtype='float16',tf32=False,dynamic=False,grouping='grouped',prefetch='whole',
                slots=2,native_scratch_bytes=128<<20,replay_slots=2,replay_buckets=8,
                replay_bytes_per_region=256<<20,warmups=5,
                samples=30 if job['kind']=='model' else 100,
                inputs=8 if job['kind']=='model' else 4,
                micro_cases=list(MICROS) if job['kind']=='micro' else None)


def provenance(runtime, artifact, compiler):
    result=previous_provenance(runtime,artifact,compiler)
    result['source'].update({str(p.relative_to(ROOT)):sha(p)
        for p in (ROOT/'bench/issue269').glob('*.py')
        if p.name not in ('analyze.py','test_analysis.py')})
    return result
