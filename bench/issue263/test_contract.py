"""Resume and aggregation must not silently mix execution settings."""
import copy
import json
from types import SimpleNamespace

import pytest

from bench.issue263.contract import freeze_contract, validate_record


def fixture():
    launch=dict(artifact='/artifact',build='/build',python='/venv/bin/python',nsys='/nsys',
                affinity=[4,5],device='0')
    platform=dict(python='3.14',torch='2.14',torch_cuda='12.6',gpu='GPU')
    env=dict(CUDA_VISIBLE_DEVICES='0',CUDA_HOME=None,LD_LIBRARY_PATH='/build/libreloc',
             OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OMP_WAIT_POLICY='PASSIVE',
             GOMP_SPINCOUNT='0',TORCHINDUCTOR_COMPILE_THREADS='1')
    sources=dict(runtime_sha256={'runtime':'hash'},artifact_sha256={'input':'hash'},
                 repository_sha256={'legacy.py':'hash'},measurement_sha256={'support.py':'hash'})
    contract=dict(schema=1,launch=launch,platform=platform,environment=env,**sources)
    x=dict(configuration=dict(artifact='/artifact',build='/build',output='/campaign/job.json',
             round=1,variant='baseline',instrument='off',callers=4),
           metadata=dict(**platform,affinity=[4,5],torch_threads=2,interop_threads=1,
             environment=env|{'TORCHINDUCTOR_CACHE_DIR':'/campaign/job-cache'},
             runtime_sha256=sources['runtime_sha256'],artifact_sha256=sources['artifact_sha256'],
             repository_sha256=sources['repository_sha256'],harness_sha256=sources['measurement_sha256']),
           frozen_runner_sha256='runner')
    job=dict(name='job',kind='concurrency',round=1,variant='baseline',mode='off',profile=False)
    audit=dict(command=['taskset','-c','4,5','/venv/bin/python','concurrency.py','--artifact','/artifact','--build','/build'])
    contract['frozen_runners']={'concurrency':'runner'}
    return contract,copy.deepcopy(x),job,audit


def test_resume_accepts_same_contract_but_rejects_changed_launch(tmp_path):
    contract,*_=fixture()
    freeze_contract(tmp_path,contract)
    freeze_contract(tmp_path,copy.deepcopy(contract))
    for field,value in [('device','999'),('affinity',[0]),('python','/other/python'),('build','/other/build')]:
        changed=copy.deepcopy(contract);changed['launch'][field]=value
        with pytest.raises(ValueError,match='execution contract'):
            freeze_contract(tmp_path,changed)


def test_old_outputs_cannot_silently_acquire_a_new_execution_contract(tmp_path):
    (tmp_path/'job.json').write_text('{}')
    with pytest.raises(ValueError,match='existing results'):
        freeze_contract(tmp_path,fixture()[0])


def test_aggregation_normalizes_threads_and_per_job_cache():
    contract,x,job,audit=fixture()
    validate_record(contract,x,job,audit)
    x['metadata']['torch_threads']=8
    with pytest.raises(ValueError,match='threads'):
        validate_record(contract,x,job,audit)


@pytest.mark.parametrize('field,value',[('gpu','another GPU'),('python','3.15'),('torch','2.15'),('affinity',[0])])
def test_aggregation_rejects_mixed_platform(field,value):
    contract,x,job,audit=fixture();x['metadata'][field]=value
    with pytest.raises(ValueError,match=field):validate_record(contract,x,job,audit)


def test_aggregation_checks_environment_interpreter_and_frozen_runner():
    for change,match in [(lambda x,a:x['metadata']['environment'].update(OMP_WAIT_POLICY='ACTIVE'),'environment'),
                           (lambda x,a:a['command'].__setitem__(3,'/other/python'),'interpreter'),
                           (lambda x,a:x.update(frozen_runner_sha256='changed'),'frozen runner'),
                           (lambda x,a:x['metadata']['repository_sha256'].update({'legacy.py':'changed'}),'repository')]:
        contract,x,job,audit=fixture();change(x,audit)
        with pytest.raises(ValueError,match=match):validate_record(contract,x,job,audit)


def test_historical_sidecar_binds_original_records_without_rewriting(tmp_path):
    from bench.issue263.contract import load_contract
    from bench.issue263.support import BASELINE,baseline_sources,digest
    contract,*_=fixture();contract.update({k:baseline_sources()[k] for k in ('repository_sha256','frozen_runners')})
    (tmp_path/'matrix.json').write_text(json.dumps([{'name':'job'}]))
    record=tmp_path/'job.json';record.write_text('{"original":true}')
    bound={p.name:digest(p) for p in (record,tmp_path/'matrix.json')}
    sidecar=dict(kind='post-campaign-audit',baseline=BASELINE,contract=contract,bound_files_sha256=bound)
    (tmp_path/'historical-qualification.json').write_text(json.dumps(sidecar))
    assert load_contract(tmp_path)==(contract,True)
    record.write_text('{"original":false}')
    with pytest.raises(ValueError,match='job.json'):load_contract(tmp_path)
