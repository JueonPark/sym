Exploratory batches, excluded from headline comparisons.
An unrelated CPU-heavy build and later simulation jobs affected these runs.
At 15:18 local time the host load average was 32.96 with many compiler workers.
Some Torch KV restore totals rose from about 14 ms to about 123 ms over 111 calls.
Do not attribute these differences to the merged patches. All workload outputs
passed correctness. Final measurements and profiles were collected separately
after the observed jobs ended, with a monitor counting both process classes.

This published directory retains timing reports, logs, profile workload reports
and available monitors. Duplicate large Nsight/SQLite traces remain locally at
/tmp/sym-main-profile-results/contended on the experiment host.
