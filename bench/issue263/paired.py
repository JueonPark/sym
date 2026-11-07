"""Process-paired effects; request percentiles are descriptive, not replicates."""
import math
import random
import statistics


def percentile(values, q):
    values=sorted(values)
    if not values or not 0 <= q <= 100: raise ValueError('empty values or invalid percentile')
    pos=(len(values)-1)*q/100
    lo=int(pos); hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def distribution(values):
    return dict(n=len(values), p50=percentile(values,50), p95=percentile(values,95), mean=statistics.mean(values))


def paired_summary(rows, *, seed=263, resamples=10000):
    if not rows: raise ValueError('no process pairs')
    seen=set(); effects=[]
    for row in rows:
        if not all(k in row for k in ('round','baseline','candidate')): raise ValueError('incomplete pair')
        if row['round'] in seen: raise ValueError('duplicate process round')
        seen.add(row['round'])
        a,b=row['baseline'],row['candidate']
        if not all(math.isfinite(x) and x>0 for x in (a,b)): raise ValueError('nonpositive/nonfinite timing')
        effects.append(100*(a-b)/a)
    rng=random.Random(seed)
    boot=[statistics.mean(rng.choices(effects,k=len(effects))) for _ in range(resamples)]
    return dict(process_pairs=len(rows), reduction_percent=statistics.mean(effects),
                ci95_percent=[percentile(boot,2.5),percentile(boot,97.5)],
                paired_reductions_percent=effects, pairs=rows, seed=seed,resamples=resamples,
                estimator='mean of process-paired percentage reductions; percentile bootstrap over processes')
