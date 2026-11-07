import sqlite3
import pytest
from bench.issue263.trace import analyze_trace, interval_union, exclusive_segments


def test_union_does_not_add_overlap():
    assert interval_union([(0, 10), (5, 20), (30, 35)]) == [(0, 20), (30, 35)]


def test_nested_exclusive_partition():
    assert exclusive_segments(0, 20, [(2, 11, 'binding'), (4, 8, 'native'), (12, 14, 'wait')]) == [
        (0, 2, 'other'), (2, 4, 'binding'), (4, 8, 'native'), (8, 11, 'binding'),
        (11, 12, 'other'), (12, 14, 'wait'), (14, 20, 'other')]


def fixture(path, beyond=False):
    c = sqlite3.connect(path)
    c.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
    INSERT INTO StringIds VALUES(1,'cudaLaunchKernel'),(2,'kernel');
    CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,globalTid INTEGER,nameId INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,globalPid INTEGER,streamId INTEGER,deviceId INTEGER,contextId INTEGER,shortName INTEGER);''')
    pid = 9 << 24
    c.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL,?)', [(0,100,'request263/A',pid+1),(20,120,'request263/B',pid+2)])
    c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,1)', [(10,12,11,pid+1),(25,26,12,pid+2)])
    c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,7,0,1,2)', [(25,101 if beyond else 60,11,pid),(35,80,12,pid)])
    c.commit(); c.close()


def test_overlapping_requests_use_launch_thread_not_containment(tmp_path):
    p = tmp_path/'trace.sqlite'; fixture(p)
    result = analyze_trace(p)
    a,b = result['requests']
    assert [a['assigned_gpu_ns'],b['assigned_gpu_ns']] == [35,45]
    assert a['device_busy_ns'] == 55
    assert len(a['activities']) == len(b['activities']) == 1
    assert a['activities'][0]['launch_tid'] != b['activities'][0]['launch_tid']


def test_gpu_beyond_completion_is_rejected(tmp_path):
    p=tmp_path/'trace.sqlite'; fixture(p,True)
    with pytest.raises(ValueError,match='completion'):
        analyze_trace(p)


def test_missing_gpu_is_rejected(tmp_path):
    p=tmp_path/'trace.sqlite'; fixture(p)
    c=sqlite3.connect(p); c.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL'); c.commit(); c.close()
    with pytest.raises(ValueError,match='CUDA'):
        analyze_trace(p)


def test_reused_event_joins_latest_record_in_same_process_context(tmp_path):
    p=tmp_path/'trace.sqlite';fixture(p)
    c=sqlite3.connect(p);pid=9<<24
    c.executescript('''CREATE TABLE CUPTI_ACTIVITY_KIND_CUDA_EVENT(globalPid INTEGER,contextId INTEGER,eventId INTEGER,streamId INTEGER,correlationId INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start INTEGER,end INTEGER,globalPid INTEGER,contextId INTEGER,eventId INTEGER,streamId INTEGER,correlationId INTEGER,syncType INTEGER);
    CREATE TABLE ENUM_CUPTI_SYNC_TYPE(id INTEGER,label TEXT);
    INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES(2,'Stream wait sync');
    INSERT INTO StringIds VALUES(3,'cudaEventRecord'),(4,'cudaStreamWaitEvent');''')
    c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)',[(5,6,20,pid+1,3),(15,16,21,pid+1,3),(18,19,22,pid+1,4)])
    c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_CUDA_EVENT VALUES(?,1,99,?,?)',[(pid,7,20),(pid,8,21)])
    c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES(20,60,?,1,99,9,22,2)',(pid,))
    c.commit();c.close()
    a=analyze_trace(p)['requests'][0]
    assert len(a['dependencies'])==1
    assert a['dependencies'][0]['record']['streamId']==8
    assert a['dependencies'][0]['record']['correlationId']==21


def test_correlation_ids_are_scoped_to_process(tmp_path):
    p=tmp_path/'trace.sqlite';fixture(p)
    c=sqlite3.connect(p);other=10<<24
    c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(9,10,11,?,1)',(other+1,))
    c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(30,50,11,?,7,0,1,2)',(other,))
    c.commit();c.close()
    a=analyze_trace(p)['requests'][0]
    assert a['assigned_gpu_ns']==35
    assert len(a['activities'])==1
