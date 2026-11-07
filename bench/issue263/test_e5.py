from types import SimpleNamespace
from bench.issue263.e5 import install
from bench.issue263.instrument import Instrumentation


def test_off_mode_does_not_change_frozen_e5_classes():
    class Worker: pass
    class Snapshots: pass
    before=(dict(vars(Worker)),dict(vars(Snapshots)))
    artifact=SimpleNamespace(e5=SimpleNamespace(Worker=Worker,Snapshots=Snapshots))
    with Instrumentation('off') as meter: install(meter,artifact,None,'lifecycle')
    assert before==(dict(vars(Worker)),dict(vars(Snapshots)))
