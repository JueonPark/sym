# Consumer layout qualification (#269)

The campaign contains 162 model arms over three independent rounds. 162 completed successfully; 0 recorded an actual OOM. 0/24 declared parent targets pass the requirement of at least 5% lower p50 than both matched Torch controls in every round.

[Complete tables](TABLES.md) · [Protocol](PROTOCOL.md) · [API contract](../../docs/consumer-layout.md) · [Review disposition](REVIEW.md) · [Public artifact](https://github.com/JueonPark/sym/wiki/Artifacts-Issue-269)

The implementation is default off. It deletes private inverse KV copies before fresh concatenation, using an ordinary FX rewrite and explicit packed wire restore. It does not change quantized weight rounding or GEMM precision. Torch and Sym receive identical lowering and replay opportunities. Existing Inductor fusion is not a Sym-specific contribution.

## Completed model results

| Sym lowered against | Per-round improvements | Comparisons | Median ratio (control / lowered) |
|---|---:|---:|---:|
| native | 18 | 24 | 1.006 |
| materialized | 13 | 24 | 1.001 |

The stricter comparison qualifies 0/8 lowered Sym regimes against the fastest observed offload control in all three rounds. A matched-mode win alone does not establish a benefit over the strongest alternative.

Across all observed offload alternatives (compute backend, representation and available replay mode), Sym lowered is at least 5% below the fastest Torch alternative in 0/24 per-round comparisons. Replay alternatives have additional retained-memory costs; this broader comparison is separate from the matched-mode parent test.

| Case | Compute | Replay | Sym lowered p50 range | Fastest Torch p50 range | Resident p50 range |
|---|---|---|---:|---:|---:|
| decode64 | eager | False | 22.101–22.669 | 17.704–17.747 | 13.730–13.877 |
| decode64 | inductor | False | 17.973–18.062 | 17.704–17.747 | 8.190–9.457 |
| decode8 | eager | False | 21.871–22.022 | 17.466–17.468 | 13.547–13.599 |
| decode8 | eager | True | 18.729–18.777 | 17.466–17.468 | 13.547–13.599 |
| decode8 | inductor | False | 17.870–18.810 | 17.466–17.468 | 8.113–8.292 |
| decode8 | inductor | True | 18.757–19.119 | 17.466–17.468 | 8.113–8.292 |
| prefill256 | eager | False | 5.581–5.657 | 4.848–4.853 | 3.968–4.043 |
| prefill256 | inductor | False | 4.883–4.905 | 4.848–4.853 | 2.316–2.391 |

Resident keeps weights and caches on the GPU and has no offload obligation. This small model fits; residency is a separate deployment alternative, not an equal-memory offload comparison. Prefill executes no KV restore and is a negative control. Held-out decode uses prompt 64 rather than prompt 8. All p95s, setup, compilation, physical peaks and per-round losses remain in the tables/raw data.

## Observed mechanism

These are separate ten-request diagnostic traces, never qualification timings. Allocation counts/bytes are positive Torch CUDA allocator events; native pools and physical cudaMalloc calls are separate scopes. Compare materialized versus lowered within each runtime. Transfer bytes come from profiler memcpy records. Logical eliminated read/write traffic is derived from tensor size, not a hardware DRAM-counter measurement.

| Compute | Runtime | Variant | Kernels/request | Allocation events/request | Allocated bytes/request | H2D bytes/request |
|---|---|---|---:|---:|---:|---:|
| eager | torch | materialized | 423 | 622 | 626266112 | 202358784 |
| eager | sym | materialized | 423 | 491 | 406081536 | 202358784 |
| eager | torch | lowered | 399 | 598 | 625823744 | 202358784 |
| eager | sym | lowered | 399 | 467 | 405639168 | 202358784 |
| eager | torch | native | 402 | 622 | 626266112 | 202358784 |
| eager | sym | native | 402 | 488 | 405639168 | 202358784 |
| inductor | torch | materialized | 283 | 342 | 624910336 | 202358784 |
| inductor | sym | materialized | 283 | 211 | 404725760 | 202358784 |
| inductor | torch | lowered | 283 | 342 | 624910336 | 202358784 |
| inductor | sym | lowered | 283 | 211 | 404725760 | 202358784 |
| inductor | torch | native | 286 | 366 | 625352704 | 202358784 |
| inductor | sym | native | 286 | 232 | 404725760 | 202358784 |

In these traces, eager lowering removes 24 kernels and 24 positive allocation events per request relative to transparent materialization. Relative to the native batched inverse conversion, it removes only three kernels. Inductor materialized and lowered paths both execute 283 kernels per request with identical allocation counts and bytes within each runtime. The rewrite therefore adds no observed kernel/allocation saving over ordinary Inductor fusion in this model.

### Actual replay execution

Counts below are warm region replays over 30 completed requests. A replay-enabled arm can fall back under its frozen admission policy; the label alone does not establish graph use.

| Compute | Runtime | Variant | Warm replays range | Retained graph MiB range |
|---|---|---|---:|---:|
| eager | torch | native | 720–720 | 280.477–280.477 |
| eager | torch | materialized | 720–720 | 280.477–280.477 |
| eager | torch | lowered | 720–720 | 280.477–280.477 |
| eager | sym | native | 720–720 | 280.477–280.477 |
| eager | sym | materialized | 720–720 | 280.477–280.477 |
| eager | sym | lowered | 720–720 | 280.477–280.477 |
| inductor | torch | native | 720–720 | 280.477–280.477 |
| inductor | torch | materialized | 720–720 | 280.477–280.477 |
| inductor | torch | lowered | 720–720 | 280.477–280.477 |
| inductor | sym | native | 720–720 | 280.477–280.477 |
| inductor | sym | materialized | 720–720 | 280.477–280.477 |
| inductor | sym | lowered | 720–720 | 280.477–280.477 |

The original decode restores lengths 8, 9, 10 across four layers and two caches. Relative to the unwrapped eager transparent materialized path, removing 24 inverse copies eliminates 442,368 logical allocation bytes and 884,736 logical read/write bytes per request. Inductor may have eliminated that allocation already; use its observed profile delta instead. The native path already batches all eight inverse conversions into one kernel per decode step, so it is a stronger eager materialization control than eight separate transparent copies. The rule's benefit must also be compared with that deployed path. This is separate from unchanged compact weight/wire traffic. Compare generated graphs, compiler caches and profile kernel names in the artifact; an FX node removal alone does not establish an Inductor kernel saving.

## Consumer cost and memory pressure

The supplements retain 90 shape/variant/round rows: 84 successes and 6 actual OOMs. Each successful row checks 100 changing-input cache results and held outputs bitwise. A visible intermediate forces fallback. The 128 MiB pressure case uses an actual CUDA reservation outside the Torch allocator.

| Compute | Shape | Variant | Completed p50 range | Consumer-only p50 range | OOM rounds |
|---|---|---|---:|---:|---:|
| eager | canonical | native | 0.302–0.305 | 0.140–0.142 | 0 |
| eager | canonical | materialized | 0.342–0.346 | 0.192–0.194 | 0 |
| eager | canonical | lowered | 0.305–0.307 | 0.157–0.157 | 0 |
| eager | held_small | native | 0.301–0.306 | 0.141–0.141 | 0 |
| eager | held_small | materialized | 0.337–0.343 | 0.191–0.192 | 0 |
| eager | held_small | lowered | 0.304–0.306 | 0.156–0.157 | 0 |
| eager | held_long | native | 1.596–1.609 | 0.277–0.303 | 0 |
| eager | held_long | materialized | 1.693–1.705 | 0.374–0.397 | 0 |
| eager | held_long | lowered | 1.644–1.661 | 0.329–0.339 | 0 |
| eager | mixed | native | 0.479–0.554 | 0.201–0.237 | 0 |
| eager | mixed | materialized | 0.566–0.585 | 0.248–0.461 | 0 |
| eager | mixed | lowered | 0.562–0.590 | 0.262–0.290 | 0 |
| eager | pressure | native | — | — | 3 |
| eager | pressure | materialized | 3.120–3.134 | 0.528–0.545 | 0 |
| eager | pressure | lowered | 3.040–3.049 | 0.452–0.458 | 0 |
| inductor | canonical | native | 0.315–0.316 | 0.149–0.151 | 0 |
| inductor | canonical | materialized | 0.300–0.304 | 0.150–0.151 | 0 |
| inductor | canonical | lowered | 0.301–0.305 | 0.150–0.152 | 0 |
| inductor | held_small | native | 0.312–0.314 | 0.150–0.151 | 0 |
| inductor | held_small | materialized | 0.300–0.303 | 0.149–0.152 | 0 |
| inductor | held_small | lowered | 0.298–0.302 | 0.149–0.152 | 0 |
| inductor | held_long | native | 1.592–1.612 | 0.296–0.300 | 0 |
| inductor | held_long | materialized | 1.604–1.609 | 0.292–0.302 | 0 |
| inductor | held_long | lowered | 1.606–1.616 | 0.288–0.302 | 0 |
| inductor | mixed | native | 0.534–0.587 | 0.243–0.246 | 0 |
| inductor | mixed | materialized | 0.547–0.561 | 0.236–0.280 | 0 |
| inductor | mixed | lowered | 0.554–0.561 | 0.255–0.291 | 0 |
| inductor | pressure | native | — | — | 3 |
| inductor | pressure | materialized | 2.950–2.968 | 0.380–0.382 | 0 |
| inductor | pressure | lowered | 2.954–2.963 | 0.377–0.386 | 0 |

The pressure case fails in all native rounds while both transparent materialized and lowered paths fit. Therefore the capacity improvement is not unique to the compiler rewrite. Eager lowered consumption also remains slower than the native pure-cat interval on canonical and held-out shapes; complete requests include the different restore costs shown in the tables.

Consumer-only intervals include the ordinary materialized or wire-aware GPU region on restored inputs. Restore intervals and complete H2D+consumer requests are also recorded. Native restore includes inverse conversion; materialized/lowered restore measures the shared raw H2D helper. Consumer intervals include dispatch/fencing and cover pure cat (native), inverse copies plus cat (materialized), or strided cat (lowered). These independent intervals cannot be subtracted from whole-model latency. Less allocation can coexist with slower strided consumption. Pressure outcomes include caller-held outputs and allocator growth; a logical admission limit does not guarantee physical success.

## Costs, correctness and limits

Setup is preparation + first request + all five warmups. Compile time is a component, not added again. Total(N) uses setup + N × median completed warm latency; break-even is reported only for a positive measured warm advantage. The fixture/oracle phase initializes libraries and constructs eight references before preparation, so this is not an application process-start benchmark. Each model process has a fresh Inductor cache; supplemental rows share a process/cache and are not independent cold starts.

| Sym lowered case | Compute | Replay | First request ms | Setup ms | Compile ms | Warm physical peak MiB |
|---|---|---|---:|---:|---:|---:|
| decode64 | eager | False | 1102.787–1113.714 | 2818.482–2833.951 | 85.269–86.127 | 944.438–944.438 |
| decode64 | inductor | False | 13002.043–13088.242 | 14698.878–14795.509 | 10834.810–10846.584 | 924.438–924.438 |
| decode8 | eager | False | 1100.498–1104.805 | 2820.711–2830.408 | 82.410–86.250 | 940.438–940.438 |
| decode8 | eager | True | 1173.990–1178.733 | 2957.462–2973.362 | 82.532–82.962 | 1466.438–1486.438 |
| decode8 | inductor | False | 12280.490–12351.335 | 13982.917–14073.762 | 10062.571–10130.001 | 920.438–920.438 |
| decode8 | inductor | True | 12445.297–12559.586 | 14235.696–14350.005 | 10136.282–10224.388 | 1486.438–1486.438 |
| prefill256 | eager | False | 408.208–409.171 | 2063.292–2068.789 | 38.327–38.537 | 944.438–944.438 |
| prefill256 | inductor | False | 5144.238–5185.545 | 6791.716–6833.786 | 4511.821–4549.345 | 944.438–944.438 |

Ranges span the three rounds. Complete per-arm first/setup/compile costs and break-even estimates remain in TABLES.md. Saved analysis also includes Total(1), Total(100) and Total(1000), separate preparation/warmup/retirement costs and phase-specific physical peaks.

Every successful model arm checks 30 outputs with the original tolerance and exact generated tokens, checks retained results after later requests and close, and records stable warm compile/capture counters. A separate post-measurement audit compares both caches from every actual decoder invocation against a materialized reference using identical arguments and the same compute backend: 256 exact cache checks for decode or 64 for prefill. No cross-compiler whole-model bitwise claim is made.

Physical memory uses 2 ms GPU-wide NVML samples, including context and any reservation, plus separate Torch allocation counters. Samples are lower bounds on transient peaks. Warm snapshots precede cache auditing/profiling; whole-process peaks include those later diagnostics and must not be interpreted as warm-serving peaks. Memory scopes overlap and are not additive.

Qualification is limited to Python 3.14.7/Torch 2.14.0+cu126, one RTX 2080Ti, CPU affinity 4–7,20–23, eight compute threads and three rounds. Static shape compilation is matched across arms; historical dynamic-compiler timings are not paired controls. Explicit wire-boundary integration, unknown/symbolic metadata exclusions, caller host-mutation obligations and supported inference-only scope remain. Dequantization inside GEMM and automatic whole-program discovery are deferred. The eligibility/cost interface describes proven logical savings, not a latency predictor or a physical-memory guarantee.
