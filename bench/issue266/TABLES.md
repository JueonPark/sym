# Complete issue 266 tables

Accepted campaign: **true**. All predeclared arms are retained. Times and memory are medians across successful rounds only; outcomes show the denominator. A dash means not applicable or no successful measurement, never zero.

E4 latency is a completed request; E5 latency is completed burst time divided by eight requests and p95 is request sojourn. Group latency is completed iterator delivery, excluding initial iterator submission. P95 is the median of the round p95s, not a pooled percentile.

Physical peaks are sampled NVML increments above the reservation/context baseline: E4 deployment excluding the oracle; E5/groups the warm loop including validation gaps. MiB means 2^20 bytes. Native device/host retention sums callers; arena storage is already inside Torch allocation. These scopes must not be added to estimate a physical peak.

## E4: completed timing and physical memory

| Arm | Success/OOM/admission | Median ms | P95 ms | Success physical MiB | OOM physical MiB | Admission physical MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L16/S16/B512 offload sym fresh | 3/0/0 | 17.4701 | 17.546 | 244 | — | — |
| L16/S16/B512 offload torch fresh | 3/0/0 | 18.3123 | 18.456 | 212 | — | — |
| L16/S16/B512 partial sym adaptive | 3/0/0 | 15.8849 | 15.9995 | 392 | — | — |
| L16/S16/B512 partial sym fresh | 3/0/0 | 13.8359 | 14.0397 | 444 | — | — |
| L16/S16/B512 partial sym packed | 3/0/0 | 15.2237 | 15.3443 | 392 | — | — |
| L16/S16/B512 partial sym recycle | 3/0/0 | 15.8797 | 16.0902 | 392 | — | — |
| L16/S16/B512 partial torch adaptive | 3/0/0 | 15.8306 | 15.9904 | 376 | — | — |
| L16/S16/B512 partial torch fresh | 3/0/0 | 14.6614 | 14.7292 | 412 | — | — |
| L16/S16/B512 partial torch packed | 3/0/0 | 15.026 | 15.1613 | 376 | — | — |
| L16/S16/B512 partial torch recycle | 3/0/0 | 15.7503 | 15.8461 | 376 | — | — |
| L16/S16/B512 resident torch fresh | 0/3/0 | — | — | — | 492 | — |
| L16/S16/B512 resident torch packed | 3/0/0 | 10.7997 | 10.8643 | 496 | — | — |
| L16/S16/B1024 offload sym fresh | 3/0/0 | 17.4941 | 17.5806 | 244 | — | — |
| L16/S16/B1024 offload torch fresh | 3/0/0 | 18.2497 | 18.3338 | 212 | — | — |
| L16/S16/B1024 partial sym adaptive | 3/0/0 | 15.9741 | 16.0847 | 392 | — | — |
| L16/S16/B1024 partial sym fresh | 3/0/0 | 13.8392 | 13.9928 | 444 | — | — |
| L16/S16/B1024 partial torch adaptive | 3/0/0 | 15.6945 | 15.8227 | 376 | — | — |
| L16/S16/B1024 partial torch fresh | 3/0/0 | 14.5867 | 14.6902 | 412 | — | — |
| L16/S16/B1024 resident torch fresh | 3/0/0 | 10.8264 | 10.8918 | 512 | — | — |
| L16/S16/B1024 resident torch packed | 3/0/0 | 12.0508 | 12.1264 | 496 | — | — |
| L16/S512/B512 offload sym fresh | 3/0/0 | 18.8304 | 19.2673 | 304 | — | — |
| L16/S512/B512 offload torch fresh | 3/0/0 | 19.3173 | 19.763 | 292 | — | — |
| L16/S512/B512 partial sym adaptive | 3/0/0 | 17.3781 | 18.4598 | 452 | — | — |
| L16/S512/B512 partial sym fresh | 3/0/0 | 15.0223 | 15.8818 | 504 | — | — |
| L16/S512/B512 partial torch adaptive | 3/0/0 | 17.3175 | 18.6551 | 436 | — | — |
| L16/S512/B512 partial torch fresh | 3/0/0 | 15.6267 | 16.896 | 472 | — | — |
| L16/S512/B512 resident torch fresh | 0/3/0 | — | — | — | 496 | — |
| L16/S512/B512 resident torch packed | 0/3/0 | — | — | — | 510 | — |
| L16/S512/B1024 offload sym fresh | 3/0/0 | 19.0811 | 23.1631 | 304 | — | — |
| L16/S512/B1024 offload torch fresh | 3/0/0 | 19.2798 | 20.7482 | 272 | — | — |
| L16/S512/B1024 partial sym adaptive | 3/0/0 | 17.1153 | 19.0365 | 452 | — | — |
| L16/S512/B1024 partial sym fresh | 3/0/0 | 14.6816 | 15.9555 | 504 | — | — |
| L16/S512/B1024 partial torch adaptive | 3/0/0 | 17.6114 | 18.7953 | 436 | — | — |
| L16/S512/B1024 partial torch fresh | 3/0/0 | 15.4844 | 17.0686 | 472 | — | — |
| L16/S512/B1024 resident torch fresh | 3/0/0 | 12.0849 | 12.9288 | 572 | — | — |
| L16/S512/B1024 resident torch packed | 3/0/0 | 12.802 | 12.9313 | 534 | — | — |
| L32/S16/B512 offload sym fresh | 3/0/0 | 34.1547 | 34.3814 | 246 | — | — |
| L32/S16/B512 offload torch fresh | 3/0/0 | 36.7763 | 36.9979 | 214 | — | — |
| L32/S16/B512 partial sym adaptive | 3/0/0 | 36.1652 | 36.287 | 394 | — | — |
| L32/S16/B512 partial sym fresh | 3/0/0 | 30.4743 | 30.6429 | 446 | — | — |
| L32/S16/B512 partial torch adaptive | 3/0/0 | 35.4252 | 35.5688 | 378 | — | — |
| L32/S16/B512 partial torch fresh | 3/0/0 | 32.8693 | 33.0892 | 414 | — | — |
| L32/S16/B512 resident torch fresh | 0/3/0 | — | — | — | 492 | — |
| L32/S16/B512 resident torch packed | 0/3/0 | — | — | — | 90 | — |
| L32/S16/B1024 offload sym fresh | 3/0/0 | 34.112 | 34.3994 | 246 | — | — |
| L32/S16/B1024 offload torch fresh | 3/0/0 | 36.4382 | 36.8091 | 234 | — | — |
| L32/S16/B1024 partial sym adaptive | 3/0/0 | 36.1185 | 36.2985 | 394 | — | — |
| L32/S16/B1024 partial sym fresh | 3/0/0 | 30.5059 | 30.6974 | 446 | — | — |
| L32/S16/B1024 partial torch adaptive | 3/0/0 | 35.4164 | 35.5507 | 378 | — | — |
| L32/S16/B1024 partial torch fresh | 3/0/0 | 32.925 | 33.2891 | 414 | — | — |
| L32/S16/B1024 resident torch fresh | 3/0/0 | 21.509 | 21.6013 | 914 | — | — |
| L32/S16/B1024 resident torch packed | 3/0/0 | 21.6248 | 21.7546 | 880 | — | — |
| L32/S512/B512 offload sym fresh | 3/0/0 | 36.1676 | 37.8661 | 336 | — | — |
| L32/S512/B512 offload torch fresh | 3/0/0 | 37.4135 | 40.5211 | 324 | — | — |
| L32/S512/B512 partial sym adaptive | 3/0/0 | 48.2748 | 51.3565 | 444 | — | — |
| L32/S512/B512 partial sym fresh | 0/3/0 | — | — | — | 510 | — |
| L32/S512/B512 partial sym packed | 3/0/0 | 39.5394 | 40.2037 | 484 | — | — |
| L32/S512/B512 partial sym recycle | 3/0/0 | 38.5204 | 41.4315 | 484 | — | — |
| L32/S512/B512 partial torch adaptive | 3/0/0 | 46.7649 | 49.0197 | 444 | — | — |
| L32/S512/B512 partial torch fresh | 3/0/0 | 33.565 | 36.8423 | 504 | — | — |
| L32/S512/B512 partial torch packed | 3/0/0 | 35.8498 | 36.1373 | 468 | — | — |
| L32/S512/B512 partial torch recycle | 3/0/0 | 38.0455 | 41.0797 | 468 | — | — |
| L32/S512/B512 resident torch fresh | 0/3/0 | — | — | — | 496 | — |
| L32/S512/B512 resident torch packed | 0/3/0 | — | — | — | 94 | — |
| L32/S512/B1024 offload sym fresh | 3/0/0 | 35.5995 | 36.5302 | 336 | — | — |
| L32/S512/B1024 offload torch fresh | 3/0/0 | 37.1247 | 40.7243 | 324 | — | — |
| L32/S512/B1024 partial sym adaptive | 3/0/0 | 37.8007 | 41.026 | 484 | — | — |
| L32/S512/B1024 partial sym fresh | 3/0/0 | 31.6624 | 32.3858 | 536 | — | — |
| L32/S512/B1024 partial torch adaptive | 3/0/0 | 37.7951 | 39.3633 | 468 | — | — |
| L32/S512/B1024 partial torch fresh | 3/0/0 | 33.593 | 36.6305 | 504 | — | — |
| L32/S512/B1024 resident torch fresh | 3/0/0 | 22.6834 | 24.6595 | 1004 | — | — |
| L32/S512/B1024 resident torch packed | 3/0/0 | 22.0441 | 24.9433 | 950 | — | — |

### Preparation and retirement

| Arm | Plan ms | Snapshot/preparation ms | Preload ms | First ms |
| --- | ---: | ---: | ---: | ---: |
| L16/S16/B512 offload sym fresh | 0.0614 | 475.6319 | 0.0797 | 37.9464 |
| L16/S16/B512 offload torch fresh | 0.0607 | 456.8588 | 0.071 | 21.0188 |
| L16/S16/B512 partial sym adaptive | 0.0602 | 475.5695 | 30.1158 | 20.2148 |
| L16/S16/B512 partial sym fresh | 0.0609 | 493.193 | 23.8798 | 17.9536 |
| L16/S16/B512 partial sym packed | 0.0615 | 473.9719 | 30.1968 | 20.1714 |
| L16/S16/B512 partial sym recycle | 0.0606 | 473.8497 | 30.2101 | 20.121 |
| L16/S16/B512 partial torch adaptive | 0.0607 | 455.5182 | 39.3162 | 16.92 |
| L16/S16/B512 partial torch fresh | 0.0606 | 454.6403 | 9.4315 | 16.4312 |
| L16/S16/B512 partial torch packed | 0.0612 | 456.9481 | 39.2725 | 16.8584 |
| L16/S16/B512 partial torch recycle | 0.0616 | 479.4689 | 39.4169 | 16.8557 |
| L16/S16/B512 resident torch fresh | — | — | — | — |
| L16/S16/B512 resident torch packed | 0.06 | 455.129 | 77.9059 | 11.2277 |
| L16/S16/B1024 offload sym fresh | 0.0621 | 474.632 | 0.0822 | 38.026 |
| L16/S16/B1024 offload torch fresh | 0.0606 | 456.065 | 0.0691 | 21.2233 |
| L16/S16/B1024 partial sym adaptive | 0.061 | 484.0987 | 30.2431 | 20.1428 |
| L16/S16/B1024 partial sym fresh | 0.0614 | 474.0362 | 23.8189 | 18.017 |
| L16/S16/B1024 partial torch adaptive | 0.0617 | 457.3183 | 39.3446 | 16.8117 |
| L16/S16/B1024 partial torch fresh | 0.0623 | 457.4831 | 9.4344 | 16.4147 |
| L16/S16/B1024 resident torch fresh | 0.0618 | 457.2208 | 18.0489 | 11.1878 |
| L16/S16/B1024 resident torch packed | 0.0609 | 466.5983 | 77.9712 | 11.2551 |
| L16/S512/B512 offload sym fresh | 0.0621 | 478.5454 | 0.0829 | 43.5523 |
| L16/S512/B512 offload torch fresh | 0.062 | 455.2765 | 0.07 | 25.6051 |
| L16/S512/B512 partial sym adaptive | 0.0611 | 480.0541 | 30.2735 | 25.2404 |
| L16/S512/B512 partial sym fresh | 0.0628 | 480.91 | 23.8052 | 22.56 |
| L16/S512/B512 partial torch adaptive | 0.0614 | 456.6365 | 39.3233 | 21.9069 |
| L16/S512/B512 partial torch fresh | 0.0615 | 457.3282 | 9.4852 | 20.8006 |
| L16/S512/B512 resident torch fresh | — | — | — | — |
| L16/S512/B512 resident torch packed | — | — | — | — |
| L16/S512/B1024 offload sym fresh | 0.0631 | 475.8767 | 0.0853 | 45.4627 |
| L16/S512/B1024 offload torch fresh | 0.0627 | 456.2505 | 0.0712 | 25.6132 |
| L16/S512/B1024 partial sym adaptive | 0.0619 | 475.6038 | 30.5957 | 25.4603 |
| L16/S512/B1024 partial sym fresh | 0.0617 | 476.2235 | 23.782 | 22.3678 |
| L16/S512/B1024 partial torch adaptive | 0.0613 | 456.4759 | 39.4358 | 21.8456 |
| L16/S512/B1024 partial torch fresh | 0.062 | 457.5781 | 9.4499 | 20.6233 |
| L16/S512/B1024 resident torch fresh | 0.0628 | 455.9562 | 18.0514 | 15.442 |
| L16/S512/B1024 resident torch packed | 0.0682 | 456.3259 | 77.9756 | 15.4971 |
| L32/S16/B512 offload sym fresh | 0.0703 | 942.5763 | 0.0858 | 61.4599 |
| L32/S16/B512 offload torch fresh | 0.0711 | 907.0316 | 0.1134 | 40.4903 |
| L32/S16/B512 partial sym adaptive | 0.0676 | 951.5595 | 33.8226 | 45.0668 |
| L32/S16/B512 partial sym fresh | 0.07 | 946.4237 | 26.0443 | 39.4015 |
| L32/S16/B512 partial torch adaptive | 0.0688 | 945.9595 | 40.5072 | 37.1993 |
| L32/S16/B512 partial torch fresh | 0.0694 | 925.0673 | 9.6159 | 35.4094 |
| L32/S16/B512 resident torch fresh | — | — | — | — |
| L32/S16/B512 resident torch packed | — | — | — | — |
| L32/S16/B1024 offload sym fresh | 0.0703 | 949.3415 | 0.0808 | 61.3506 |
| L32/S16/B1024 offload torch fresh | 0.0718 | 917.7428 | 0.1071 | 40.0674 |
| L32/S16/B1024 partial sym adaptive | 0.0704 | 958.9027 | 33.778 | 44.7102 |
| L32/S16/B1024 partial sym fresh | 0.0695 | 942.6253 | 27.2885 | 39.4552 |
| L32/S16/B1024 partial torch adaptive | 0.0697 | 912.3675 | 40.6267 | 37.5257 |
| L32/S16/B1024 partial torch fresh | 0.0699 | 921.8189 | 9.6398 | 34.9828 |
| L32/S16/B1024 resident torch fresh | 0.0694 | 923.2346 | 35.4954 | 22.5232 |
| L32/S16/B1024 resident torch packed | 0.0695 | 927.9743 | 157.0167 | 22.6221 |
| L32/S512/B512 offload sym fresh | 0.0773 | 940.1308 | 0.1071 | 71.2665 |
| L32/S512/B512 offload torch fresh | 0.0759 | 909.4163 | 0.1361 | 48.5947 |
| L32/S512/B512 partial sym adaptive | 0.0749 | 938.6647 | 33.9139 | 58.8294 |
| L32/S512/B512 partial sym fresh | — | — | — | — |
| L32/S512/B512 partial sym packed | 0.0758 | 946.2759 | 34.0214 | 53.7464 |
| L32/S512/B512 partial sym recycle | 0.0763 | 943.6599 | 33.8704 | 56.1717 |
| L32/S512/B512 partial torch adaptive | 0.0758 | 908.286 | 40.509 | 54.2159 |
| L32/S512/B512 partial torch fresh | 0.0769 | 906.9138 | 9.6183 | 42.6131 |
| L32/S512/B512 partial torch packed | 0.0772 | 910.4816 | 40.2929 | 45.4948 |
| L32/S512/B512 partial torch recycle | 0.0754 | 910.2136 | 40.3666 | 47.9584 |
| L32/S512/B512 resident torch fresh | — | — | — | — |
| L32/S512/B512 resident torch packed | — | — | — | — |
| L32/S512/B1024 offload sym fresh | 0.0707 | 959.4194 | 0.1029 | 71.654 |
| L32/S512/B1024 offload torch fresh | 0.0699 | 910.8097 | 0.1422 | 48.1994 |
| L32/S512/B1024 partial sym adaptive | 0.0694 | 938.8265 | 33.882 | 55.9431 |
| L32/S512/B1024 partial sym fresh | 0.0712 | 952.0775 | 27.8332 | 47.6011 |
| L32/S512/B1024 partial torch adaptive | 0.0698 | 910.8253 | 40.1018 | 47.5049 |
| L32/S512/B1024 partial torch fresh | 0.0702 | 911.3549 | 9.6415 | 43.1199 |
| L32/S512/B1024 resident torch fresh | 0.0696 | 909.0948 | 35.5429 | 29.7451 |
| L32/S512/B1024 resident torch packed | 0.0714 | 916.8749 | 156.9842 | 29.8136 |

### Retained owner and allocator scopes at warm end

| Arm | Torch allocated MiB | Torch reserved MiB | Native device MiB | Native host MiB | Arena allocated MiB | Arena live MiB | Torch pinned active MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L16/S16/B512 offload sym fresh | 36.1587 | 154 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S16/B512 offload torch fresh | 12.1587 | 156 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B512 partial sym adaptive | 276.1587 | 298 | 30.8203 | 0.082 | 72 | 24 | 208 |
| L16/S16/B512 partial sym fresh | 228.1587 | 354 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S16/B512 partial sym packed | 228.1587 | 298 | 30.8203 | 0.082 | 24 | 24 | 208 |
| L16/S16/B512 partial sym recycle | 276.1587 | 298 | 30.8203 | 0.082 | 72 | 24 | 208 |
| L16/S16/B512 partial torch adaptive | 276.1587 | 320 | 0 | 0 | 72 | 0 | 208.625 |
| L16/S16/B512 partial torch fresh | 205.1587 | 356 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B512 partial torch packed | 204.1587 | 320 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B512 partial torch recycle | 276.1587 | 320 | 0 | 0 | 72 | 0 | 208.625 |
| L16/S16/B512 resident torch fresh | — | — | — | — | — | — | — |
| L16/S16/B512 resident torch packed | 396.1587 | 418 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B1024 offload sym fresh | 36.1587 | 154 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S16/B1024 offload torch fresh | 12.1587 | 156 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B1024 partial sym adaptive | 276.1587 | 298 | 30.8203 | 0.082 | 72 | 24 | 208 |
| L16/S16/B1024 partial sym fresh | 228.1587 | 354 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S16/B1024 partial torch adaptive | 276.1587 | 320 | 0 | 0 | 72 | 0 | 208.625 |
| L16/S16/B1024 partial torch fresh | 205.1587 | 356 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B1024 resident torch fresh | 397.1587 | 456 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S16/B1024 resident torch packed | 396.1587 | 418 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B512 offload sym fresh | 37.1309 | 214 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S512/B512 offload torch fresh | 13.1309 | 236 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B512 partial sym adaptive | 277.1309 | 358 | 30.8203 | 0.082 | 72 | 24 | 208 |
| L16/S512/B512 partial sym fresh | 229.1309 | 414 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S512/B512 partial torch adaptive | 277.1309 | 380 | 0 | 0 | 72 | 0 | 208.625 |
| L16/S512/B512 partial torch fresh | 206.1309 | 416 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B512 resident torch fresh | — | — | — | — | — | — | — |
| L16/S512/B512 resident torch packed | — | — | — | — | — | — | — |
| L16/S512/B1024 offload sym fresh | 37.1309 | 214 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S512/B1024 offload torch fresh | 13.1309 | 216 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B1024 partial sym adaptive | 277.1309 | 358 | 30.8203 | 0.082 | 72 | 24 | 208 |
| L16/S512/B1024 partial sym fresh | 229.1309 | 414 | 27.5703 | 0.0703 | 0 | 0 | 208 |
| L16/S512/B1024 partial torch adaptive | 277.1309 | 380 | 0 | 0 | 72 | 0 | 208.625 |
| L16/S512/B1024 partial torch fresh | 206.1309 | 416 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B1024 resident torch fresh | 398.1309 | 516 | 0 | 0 | 0 | 0 | 208.625 |
| L16/S512/B1024 resident torch packed | 397.1309 | 478 | 0 | 0 | 0 | 0 | 208.625 |
| L32/S16/B512 offload sym fresh | 36.1587 | 156 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S16/B512 offload torch fresh | 12.1587 | 158 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S16/B512 partial sym adaptive | 276.1587 | 300 | 30.8203 | 0.082 | 72 | 24 | 416 |
| L32/S16/B512 partial sym fresh | 228.1587 | 356 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S16/B512 partial torch adaptive | 276.1587 | 322 | 0 | 0 | 72 | 0 | 417.25 |
| L32/S16/B512 partial torch fresh | 205.1587 | 358 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S16/B512 resident torch fresh | — | — | — | — | — | — | — |
| L32/S16/B512 resident torch packed | — | — | — | — | — | — | — |
| L32/S16/B1024 offload sym fresh | 36.1587 | 156 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S16/B1024 offload torch fresh | 12.1587 | 178 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S16/B1024 partial sym adaptive | 276.1587 | 300 | 30.8203 | 0.082 | 72 | 24 | 416 |
| L32/S16/B1024 partial sym fresh | 228.1587 | 356 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S16/B1024 partial torch adaptive | 276.1587 | 322 | 0 | 0 | 72 | 0 | 417.25 |
| L32/S16/B1024 partial torch fresh | 205.1587 | 358 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S16/B1024 resident torch fresh | 781.1587 | 858 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S16/B1024 resident torch packed | 780.1587 | 804 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B512 offload sym fresh | 37.1309 | 246 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S512/B512 offload torch fresh | 13.1309 | 268 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B512 partial sym adaptive | 253.1309 | 366 | 17.0352 | 0.0469 | 48 | 24 | 416 |
| L32/S512/B512 partial sym fresh | — | — | — | — | — | — | — |
| L32/S512/B512 partial sym packed | 229.1309 | 390 | 30.8203 | 0.082 | 24 | 24 | 416 |
| L32/S512/B512 partial sym recycle | 277.1309 | 390 | 30.8203 | 0.082 | 72 | 24 | 416 |
| L32/S512/B512 partial torch adaptive | 253.1309 | 388 | 0 | 0 | 48 | 0 | 417.25 |
| L32/S512/B512 partial torch fresh | 206.1309 | 448 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B512 partial torch packed | 205.1309 | 412 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B512 partial torch recycle | 277.1309 | 412 | 0 | 0 | 72 | 0 | 417.25 |
| L32/S512/B512 resident torch fresh | — | — | — | — | — | — | — |
| L32/S512/B512 resident torch packed | — | — | — | — | — | — | — |
| L32/S512/B1024 offload sym fresh | 37.1309 | 246 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S512/B1024 offload torch fresh | 13.1309 | 268 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B1024 partial sym adaptive | 277.1309 | 390 | 30.8203 | 0.082 | 72 | 24 | 416 |
| L32/S512/B1024 partial sym fresh | 229.1309 | 446 | 27.5703 | 0.0703 | 0 | 0 | 416 |
| L32/S512/B1024 partial torch adaptive | 277.1309 | 412 | 0 | 0 | 72 | 0 | 417.25 |
| L32/S512/B1024 partial torch fresh | 206.1309 | 448 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B1024 resident torch fresh | 782.1309 | 948 | 0 | 0 | 0 | 0 | 417.25 |
| L32/S512/B1024 resident torch packed | 781.1309 | 894 | 0 | 0 | 0 | 0 | 417.25 |

### Warm allocation and reuse counter deltas

| Arm | Native device allocations | Native context creations | Arena allocations | Arena reuses | Arena barriers | Arena refused attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L16/S16/B512 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B512 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B512 partial sym adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B512 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B512 partial sym packed | 0 | 0 | 160 | 0 | 0 | 83 |
| L16/S16/B512 partial sym recycle | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B512 partial torch adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B512 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B512 partial torch packed | 0 | 0 | 160 | 0 | 0 | 100 |
| L16/S16/B512 partial torch recycle | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B512 resident torch fresh | — | — | — | — | — | — |
| L16/S16/B512 resident torch packed | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 partial sym adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B1024 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 partial torch adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S16/B1024 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 resident torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S16/B1024 resident torch packed | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B512 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B512 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B512 partial sym adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S512/B512 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B512 partial torch adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S512/B512 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B512 resident torch fresh | — | — | — | — | — | — |
| L16/S512/B512 resident torch packed | — | — | — | — | — | — |
| L16/S512/B1024 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B1024 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B1024 partial sym adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S512/B1024 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B1024 partial torch adaptive | 0 | 0 | 0 | 160 | 160 | 100 |
| L16/S512/B1024 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B1024 resident torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L16/S512/B1024 resident torch packed | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B512 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B512 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B512 partial sym adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S16/B512 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B512 partial torch adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S16/B512 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B512 resident torch fresh | — | — | — | — | — | — |
| L32/S16/B512 resident torch packed | — | — | — | — | — | — |
| L32/S16/B1024 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B1024 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B1024 partial sym adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S16/B1024 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B1024 partial torch adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S16/B1024 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B1024 resident torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S16/B1024 resident torch packed | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B512 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B512 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B512 partial sym adaptive | 0 | 0 | 0 | 480 | 480 | 0 |
| L32/S512/B512 partial sym fresh | — | — | — | — | — | — |
| L32/S512/B512 partial sym packed | 0 | 0 | 480 | 0 | 0 | 414 |
| L32/S512/B512 partial sym recycle | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S512/B512 partial torch adaptive | 0 | 0 | 0 | 480 | 480 | 0 |
| L32/S512/B512 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B512 partial torch packed | 0 | 0 | 480 | 0 | 0 | 420 |
| L32/S512/B512 partial torch recycle | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S512/B512 resident torch fresh | — | — | — | — | — | — |
| L32/S512/B512 resident torch packed | — | — | — | — | — | — |
| L32/S512/B1024 offload sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B1024 offload torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B1024 partial sym adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S512/B1024 partial sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B1024 partial torch adaptive | 0 | 0 | 0 | 480 | 480 | 420 |
| L32/S512/B1024 partial torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B1024 resident torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| L32/S512/B1024 resident torch packed | 0 | 0 | 0 | 0 | 0 | 0 |

## E5: completed timing and physical memory

| Arm | Success/OOM/admission | Median ms | P95 ms | Success physical MiB |
| --- | ---: | ---: | ---: | ---: |
| llm_decode C1 sym adaptive | 3/0/0 | 23.1018 | 184.8466 | 304 |
| llm_decode C1 sym fresh | 3/0/0 | 19.1307 | 152.9356 | 332 |
| llm_decode C1 sym packed | 3/0/0 | 20.5462 | 164.2928 | 304 |
| llm_decode C1 sym recycle | 3/0/0 | 23.0603 | 184.4269 | 304 |
| llm_decode C1 torch adaptive | 3/0/0 | 21.6408 | 173.3163 | 290 |
| llm_decode C1 torch fresh | 3/0/0 | 19.7336 | 158.2315 | 298 |
| llm_decode C1 torch packed | 3/0/0 | 19.5782 | 156.589 | 290 |
| llm_decode C1 torch recycle | 3/0/0 | 21.6753 | 177.7858 | 290 |
| llm_decode C2 sym adaptive | 3/0/0 | 24.8331 | 201.2135 | 444 |
| llm_decode C2 sym fresh | 3/0/0 | 17.3773 | 140.1258 | 540 |
| llm_decode C2 torch adaptive | 3/0/0 | 21.7543 | 175.502 | 414 |
| llm_decode C2 torch fresh | 3/0/0 | 17.5984 | 150.3654 | 430 |
| llm_decode C4 sym adaptive | 3/0/0 | 30.4317 | 243.9671 | 526 |
| llm_decode C4 sym fresh | 3/0/0 | 19.7156 | 157.8248 | 956 |
| llm_decode C4 torch adaptive | 3/0/0 | 23.8912 | 192.129 | 566 |
| llm_decode C4 torch fresh | 3/0/0 | 20.504 | 164.0667 | 694 |
| moe_sparse C1 sym adaptive | 3/0/0 | 3.8576 | 30.7606 | 184 |
| moe_sparse C1 sym fresh | 3/0/0 | 3.0642 | 24.4506 | 184 |
| moe_sparse C1 sym packed | 3/0/0 | 3.2522 | 25.8649 | 184 |
| moe_sparse C1 sym recycle | 3/0/0 | 3.6495 | 29.237 | 184 |
| moe_sparse C1 torch adaptive | 3/0/0 | 3.2282 | 25.7243 | 170 |
| moe_sparse C1 torch fresh | 3/0/0 | 2.8264 | 22.4618 | 170 |
| moe_sparse C1 torch packed | 3/0/0 | 3.2458 | 25.8798 | 170 |
| moe_sparse C1 torch recycle | 3/0/0 | 3.2477 | 25.8066 | 170 |
| moe_sparse C2 sym adaptive | 3/0/0 | 3.2542 | 26.0309 | 270 |
| moe_sparse C2 sym fresh | 3/0/0 | 2.4786 | 19.7192 | 270 |
| moe_sparse C2 torch adaptive | 3/0/0 | 2.8898 | 23.0065 | 242 |
| moe_sparse C2 torch fresh | 3/0/0 | 2.5402 | 20.1775 | 242 |
| moe_sparse C4 sym adaptive | 3/0/0 | 3.3844 | 27.3442 | 442 |
| moe_sparse C4 sym fresh | 3/0/0 | 2.3106 | 18.48 | 442 |
| moe_sparse C4 torch adaptive | 3/0/0 | 3.0725 | 24.8083 | 386 |
| moe_sparse C4 torch fresh | 3/0/0 | 2.4326 | 19.5311 | 386 |

### Preparation and retirement

| Arm | Plan ms | Snapshot/preparation ms | Owner ms | Workers prepare sum ms | First max ms | Workers retire sum ms | Service p95 ms | Queue p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_decode C1 sym adaptive | 0.0513 | 118.5214 | 6.6594 | 4.5394 | 60.1847 | 2.2919 | 23.227 | 161.7797 |
| llm_decode C1 sym fresh | 0.0524 | 118.5846 | 6.725 | 4.5257 | 57.2609 | 2.4427 | 19.3453 | 133.8484 |
| llm_decode C1 sym packed | 0.0528 | 118.9321 | 6.7748 | 4.5927 | 59.4461 | 2.2549 | 20.6785 | 143.7598 |
| llm_decode C1 sym recycle | 0.0517 | 119.2844 | 6.7619 | 4.5974 | 59.879 | 2.2905 | 24.3443 | 161.41 |
| llm_decode C1 torch adaptive | 0.0517 | 116.9023 | 0.0006 | 4.1865 | 24.3132 | 0.1805 | 23.1005 | 151.5531 |
| llm_decode C1 torch fresh | 0.0525 | 116.923 | 0.0008 | 4.2759 | 24.4299 | 0.1028 | 21.7394 | 138.0836 |
| llm_decode C1 torch packed | 0.052 | 117.1865 | 0.0007 | 4.1647 | 24.094 | 0.1158 | 21.0864 | 137.0121 |
| llm_decode C1 torch recycle | 0.0524 | 116.6519 | 0.0008 | 4.2309 | 24.7693 | 0.2232 | 23.2456 | 155.6355 |
| llm_decode C2 sym adaptive | 0.053 | 117.8991 | 6.747 | 5.6686 | 60.817 | 4.6637 | 52.5155 | 153.3713 |
| llm_decode C2 sym fresh | 0.0519 | 117.9145 | 6.7705 | 5.0016 | 58.2565 | 4.5596 | 37.4259 | 108.0511 |
| llm_decode C2 torch adaptive | 0.0514 | 117.2569 | 0.0008 | 4.3577 | 24.5322 | 0.3035 | 46.6673 | 132.4077 |
| llm_decode C2 torch fresh | 0.0512 | 117.085 | 0.0007 | 4.433 | 23.9114 | 0.1413 | 40.1322 | 114.6333 |
| llm_decode C4 sym adaptive | 0.0568 | 118.1136 | 6.694 | 5.7316 | 74.2352 | 4.54 | 135.5843 | 135.7849 |
| llm_decode C4 sym fresh | 0.0566 | 118.6264 | 10.8035 | 5.8685 | 60.5056 | 8.6892 | 87.055 | 87.7428 |
| llm_decode C4 torch adaptive | 0.0565 | 116.8993 | 0.0008 | 4.5976 | 39.2503 | 0.5304 | 101.6118 | 102.0349 |
| llm_decode C4 torch fresh | 0.057 | 117.1174 | 0.0007 | 4.7747 | 24.6888 | 0.236 | 89.537 | 89.8963 |
| moe_sparse C1 sym adaptive | 0.0596 | 156.5922 | 6.715 | 4.5896 | 25.9247 | 1.4464 | 3.896 | 26.9309 |
| moe_sparse C1 sym fresh | 0.0584 | 156.626 | 6.6115 | 4.5289 | 25.2513 | 1.449 | 3.0951 | 21.3954 |
| moe_sparse C1 sym packed | 0.0588 | 157.7839 | 6.6787 | 4.5312 | 25.9254 | 1.364 | 3.2741 | 22.6308 |
| moe_sparse C1 sym recycle | 0.0591 | 155.6905 | 6.6156 | 4.5654 | 25.8876 | 1.3898 | 3.9035 | 25.6362 |
| moe_sparse C1 torch adaptive | 0.059 | 151.6854 | 0.0006 | 4.1897 | 6.1531 | 0.1336 | 3.2516 | 22.5231 |
| moe_sparse C1 torch fresh | 0.0579 | 152.3854 | 0.0005 | 4.189 | 6.0062 | 0.0521 | 3.0933 | 19.6756 |
| moe_sparse C1 torch packed | 0.0583 | 152.874 | 0.0007 | 4.2148 | 6.3237 | 0.0783 | 3.2977 | 22.6518 |
| moe_sparse C1 torch recycle | 0.0576 | 152.6428 | 0.0006 | 4.1801 | 6.2158 | 0.1118 | 3.2911 | 22.6152 |
| moe_sparse C2 sym adaptive | 0.0591 | 154.8123 | 6.8114 | 5.5271 | 26.0521 | 2.7261 | 6.9413 | 20.3248 |
| moe_sparse C2 sym fresh | 0.0591 | 155.681 | 6.6746 | 4.9395 | 25.3261 | 2.7521 | 5.1744 | 15.2967 |
| moe_sparse C2 torch adaptive | 0.0595 | 151.2448 | 0.001 | 4.6527 | 6.1475 | 0.2882 | 6.1712 | 18.2208 |
| moe_sparse C2 torch fresh | 0.0595 | 155.4102 | 0.0006 | 4.5801 | 5.9959 | 0.0906 | 5.3455 | 15.9487 |
| moe_sparse C4 sym adaptive | 0.0589 | 162.8037 | 6.6522 | 6.0857 | 25.8351 | 4.6305 | 14.2351 | 14.5111 |
| moe_sparse C4 sym fresh | 0.0579 | 156.0894 | 6.6198 | 5.5816 | 25.3157 | 4.5661 | 9.8396 | 10.2079 |
| moe_sparse C4 torch adaptive | 0.0597 | 151.503 | 0.0008 | 4.6864 | 6.1102 | 0.4742 | 13.0526 | 13.2727 |
| moe_sparse C4 torch fresh | 0.0598 | 151.4332 | 0.0006 | 4.6983 | 5.7641 | 0.1553 | 10.1997 | 10.4703 |

### Retained owner and allocator scopes at warm end

| Arm | Torch allocated MiB | Torch reserved MiB | Native device MiB | Native host MiB | Arena allocated MiB | Arena live MiB | Torch pinned active MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_decode C1 sym adaptive | 92.5767 | 206 | 27.5703 | 0.0703 | 72 | 24 | 52 |
| llm_decode C1 sym fresh | 44.5767 | 234 | 27.5703 | 0.0703 | 0 | 0 | 52 |
| llm_decode C1 sym packed | 44.5767 | 206 | 27.5703 | 0.0703 | 24 | 24 | 52 |
| llm_decode C1 sym recycle | 92.5767 | 206 | 27.5703 | 0.0703 | 72 | 24 | 52 |
| llm_decode C1 torch adaptive | 92.5767 | 226 | 0 | 0 | 72 | 0 | 52.4062 |
| llm_decode C1 torch fresh | 20.5767 | 234 | 0 | 0 | 0 | 0 | 52.4062 |
| llm_decode C1 torch packed | 20.5767 | 226 | 0 | 0 | 0 | 0 | 52.4062 |
| llm_decode C1 torch recycle | 92.5767 | 226 | 0 | 0 | 72 | 0 | 52.4062 |
| llm_decode C2 sym adaptive | 172.8579 | 302 | 55.1406 | 0.1406 | 144 | 48 | 52 |
| llm_decode C2 sym fresh | 76.8579 | 398 | 55.1406 | 0.1406 | 0 | 0 | 52 |
| llm_decode C2 torch adaptive | 172.8579 | 342 | 0 | 0 | 144 | 0 | 52.6562 |
| llm_decode C2 torch fresh | 28.8579 | 358 | 0 | 0 | 0 | 0 | 52.6562 |
| llm_decode C4 sym adaptive | 237.4204 | 398 | 32.0469 | 0.1094 | 192 | 96 | 52 |
| llm_decode C4 sym fresh | 141.4204 | 726 | 110.2812 | 0.2812 | 0 | 0 | 52 |
| llm_decode C4 torch adaptive | 237.4204 | 478 | 0 | 0 | 192 | 0 | 53.1562 |
| llm_decode C4 torch fresh | 45.4204 | 606 | 0 | 0 | 0 | 0 | 53.1562 |
| moe_sparse C1 sym adaptive | 40.3145 | 104 | 9.0234 | 0.0234 | 24 | 8 | 64 |
| moe_sparse C1 sym fresh | 24.3145 | 104 | 9.0234 | 0.0234 | 0 | 0 | 64 |
| moe_sparse C1 sym packed | 24.3145 | 104 | 9.0234 | 0.0234 | 8 | 8 | 64 |
| moe_sparse C1 sym recycle | 40.3145 | 104 | 9.0234 | 0.0234 | 24 | 8 | 64 |
| moe_sparse C1 torch adaptive | 40.3145 | 106 | 0 | 0 | 24 | 0 | 64.1875 |
| moe_sparse C1 torch fresh | 16.3145 | 106 | 0 | 0 | 0 | 0 | 64.1875 |
| moe_sparse C1 torch packed | 16.3145 | 106 | 0 | 0 | 0 | 0 | 64.1875 |
| moe_sparse C1 torch recycle | 40.3145 | 106 | 0 | 0 | 24 | 0 | 64.1875 |
| moe_sparse C2 sym adaptive | 72.4395 | 166 | 18.0469 | 0.0469 | 48 | 16 | 64 |
| moe_sparse C2 sym fresh | 40.4395 | 166 | 18.0469 | 0.0469 | 0 | 0 | 64 |
| moe_sparse C2 torch adaptive | 72.4395 | 170 | 0 | 0 | 48 | 0 | 64.1875 |
| moe_sparse C2 torch fresh | 24.4395 | 170 | 0 | 0 | 0 | 0 | 64.1875 |
| moe_sparse C4 sym adaptive | 136.6895 | 290 | 36.0938 | 0.0938 | 96 | 32 | 64 |
| moe_sparse C4 sym fresh | 72.6895 | 290 | 36.0938 | 0.0938 | 0 | 0 | 64 |
| moe_sparse C4 torch adaptive | 136.6895 | 298 | 0 | 0 | 96 | 0 | 64.1875 |
| moe_sparse C4 torch fresh | 40.6895 | 298 | 0 | 0 | 0 | 0 | 64.1875 |

### Warm allocation and reuse counter deltas

| Arm | Native device allocations | Native context creations | Arena allocations | Arena reuses | Arena barriers | Arena refused attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_decode C1 sym adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C1 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_decode C1 sym packed | 0 | 0 | 3840 | 0 | 0 | 0 |
| llm_decode C1 sym recycle | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C1 torch adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C1 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_decode C1 torch packed | 0 | 0 | 3840 | 0 | 0 | 0 |
| llm_decode C1 torch recycle | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C2 sym adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C2 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_decode C2 torch adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C2 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_decode C4 sym adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C4 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| llm_decode C4 torch adaptive | 0 | 0 | 0 | 3840 | 3840 | 0 |
| llm_decode C4 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C1 sym adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C1 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C1 sym packed | 0 | 0 | 960 | 0 | 0 | 0 |
| moe_sparse C1 sym recycle | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C1 torch adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C1 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C1 torch packed | 0 | 0 | 960 | 0 | 0 | 0 |
| moe_sparse C1 torch recycle | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C2 sym adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C2 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C2 torch adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C2 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C4 sym adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C4 sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| moe_sparse C4 torch adaptive | 0 | 0 | 0 | 960 | 960 | 0 |
| moe_sparse C4 torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |

## GROUPS: completed timing and physical memory

| Arm | Success/OOM/admission | Median ms | P95 ms | Success physical MiB |
| --- | ---: | ---: | ---: | ---: |
| large sym adaptive | 3/0/0 | 1.6053 | 1.7134 | 86 |
| large sym fresh | 3/0/0 | 1.4772 | 1.5608 | 176 |
| large sym packed | 3/0/0 | 1.5937 | 1.7277 | 128 |
| large sym recycle | 3/0/0 | 1.6013 | 1.662 | 128 |
| large torch adaptive | 3/0/0 | 1.4699 | 1.5415 | 90 |
| large torch fresh | 3/0/0 | 1.3179 | 1.378 | 142 |
| large torch packed | 3/0/0 | 1.4531 | 1.5751 | 114 |
| large torch recycle | 3/0/0 | 1.4489 | 1.583 | 114 |
| small sym adaptive | 3/0/0 | 0.4738 | 0.5759 | 24 |
| small sym fresh | 3/0/0 | 0.2602 | 0.3427 | 24 |
| small sym packed | 3/0/0 | 0.4629 | 0.5612 | 24 |
| small sym recycle | 3/0/0 | 0.4563 | 0.5611 | 24 |
| small torch adaptive | 3/0/0 | 0.4766 | 0.6099 | 22 |
| small torch fresh | 3/0/0 | 0.3905 | 0.5184 | 22 |
| small torch packed | 3/0/0 | 0.4902 | 0.6067 | 22 |
| small torch recycle | 3/0/0 | 0.4824 | 0.6163 | 22 |

### Preparation and retirement

| Arm | Plan ms | Snapshot/preparation ms | First ms | Retire ms |
| --- | ---: | ---: | ---: | ---: |
| large sym adaptive | 0.1134 | 41.4014 | 19.5235 | 1.3625 |
| large sym fresh | 0.1193 | 41.3857 | 19.4627 | 2.29 |
| large sym packed | 0.1211 | 41.9655 | 19.5699 | 2.2433 |
| large sym recycle | 0.1141 | 41.6281 | 19.5234 | 2.457 |
| large torch adaptive | 0.116 | 33.3785 | 2.6097 | 0.1268 |
| large torch fresh | 0.1159 | 33.3953 | 2.1734 | 0.0718 |
| large torch packed | 0.1217 | 33.3096 | 2.5981 | 0.0953 |
| large torch recycle | 0.1188 | 33.4717 | 2.6272 | 0.1726 |
| small sym adaptive | 0.1143 | 16.0175 | 19.0884 | 0.5246 |
| small sym fresh | 0.1203 | 13.814 | 19.2483 | 0.4679 |
| small sym packed | 0.1117 | 14.9555 | 18.9936 | 0.5083 |
| small sym recycle | 0.1104 | 15.5957 | 20.755 | 0.5131 |
| small torch adaptive | 0.1216 | 6.5112 | 1.7567 | 0.0715 |
| small torch fresh | 0.1166 | 6.1533 | 1.5375 | 0.0215 |
| small torch packed | 0.1149 | 6.3445 | 2.8117 | 0.0416 |
| small torch recycle | 0.117 | 7.2223 | 1.7146 | 0.0722 |

### Retained owner and allocator scopes at warm end

| Arm | Torch allocated MiB | Torch reserved MiB | Native device MiB | Native host MiB | Arena allocated MiB | Arena live MiB | Torch pinned active MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large sym adaptive | 48 | 48 | 13.7773 | 0.0273 | 48 | 24 | 13 |
| large sym fresh | 24 | 120 | 27.5547 | 0.0547 | 0 | 0 | 13 |
| large sym packed | 24 | 72 | 27.5547 | 0.0547 | 24 | 24 | 13 |
| large sym recycle | 72 | 72 | 27.5547 | 0.0547 | 72 | 24 | 13 |
| large torch adaptive | 48 | 70 | 0 | 0 | 48 | 24 | 13.0273 |
| large torch fresh | 24 | 122 | 0 | 0 | 0 | 0 | 13.0273 |
| large torch packed | 24 | 94 | 0 | 0 | 24 | 24 | 13.0273 |
| large torch recycle | 72 | 94 | 0 | 0 | 72 | 24 | 13.0273 |
| small sym adaptive | 0.002 | 2 | 0.0007 | 0.0002 | 0.002 | 0.001 | 0.0005 |
| small sym fresh | 0.002 | 2 | 0.0007 | 0.0002 | 0 | 0 | 0.0005 |
| small sym packed | 0.001 | 2 | 0.0007 | 0.0002 | 0.001 | 0.001 | 0.0005 |
| small sym recycle | 0.0029 | 2 | 0.0007 | 0.0002 | 0.0029 | 0.001 | 0.0005 |
| small torch adaptive | 0.002 | 2 | 0 | 0 | 0.002 | 0.001 | 0.0007 |
| small torch fresh | 0.002 | 2 | 0 | 0 | 0 | 0 | 0.0007 |
| small torch packed | 0.001 | 2 | 0 | 0 | 0.001 | 0.001 | 0.0007 |
| small torch recycle | 0.0029 | 2 | 0 | 0 | 0.0029 | 0.001 | 0.0007 |

### Warm allocation and reuse counter deltas

| Arm | Native device allocations | Native context creations | Arena allocations | Arena reuses | Arena barriers | Arena refused attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large sym adaptive | 0 | 0 | 1 | 99 | 99 | 0 |
| large sym fresh | 8 | 1 | 0 | 0 | 0 | 0 |
| large sym packed | 8 | 1 | 100 | 0 | 0 | 97 |
| large sym recycle | 8 | 1 | 2 | 98 | 98 | 97 |
| large torch adaptive | 0 | 0 | 1 | 99 | 99 | 0 |
| large torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| large torch packed | 0 | 0 | 100 | 0 | 0 | 97 |
| large torch recycle | 0 | 0 | 2 | 98 | 98 | 97 |
| small sym adaptive | 0 | 0 | 1 | 99 | 99 | 98 |
| small sym fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| small sym packed | 0 | 0 | 100 | 0 | 0 | 97 |
| small sym recycle | 0 | 0 | 2 | 98 | 98 | 97 |
| small torch adaptive | 0 | 0 | 1 | 99 | 99 | 98 |
| small torch fresh | 0 | 0 | 0 | 0 | 0 | 0 |
| small torch packed | 0 | 0 | 100 | 0 | 0 | 97 |
| small torch recycle | 0 | 0 | 2 | 98 | 98 | 97 |

Arena refused attempts include speculative prefetch backpressure that subsequently progressed; they are not terminal admission outcomes. Full per-round samples, worker lifecycle timings, allocator peaks, snapshot counters and post-close counters remain in the JSON records and analysis.
