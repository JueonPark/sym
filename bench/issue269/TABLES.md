# Complete qualification tables (#269)

Times are milliseconds; physical peaks are sampled GPU-wide used MiB and include context/reservation. Resident has a separate no-offload deployment contract. See RESULTS.md for scope and interpretation.

## Model results

| Round | Case | Compute | Replay | Runtime | Variant | p50 | p95 | Submit p50 | First | Setup | Compile | Warm peak MiB |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | decode8 | eager | false | torch | materialized | 23.260 | 23.397 | 23.249 | 1049.734 | 2765.176 | 56.028 | 908.4 |
| 1 | decode8 | eager | false | torch | resident | 13.559 | 13.636 | 13.554 | 973.510 | 2536.746 | 54.876 | 838.4 |
| 1 | decode8 | eager | false | sym | materialized | 22.259 | 22.416 | 22.253 | 1071.180 | 2792.316 | 57.049 | 940.4 |
| 1 | decode8 | eager | false | torch | lowered | 22.912 | 23.006 | 22.907 | 1085.437 | 2800.188 | 85.160 | 908.4 |
| 1 | decode8 | eager | false | sym | lowered | 21.871 | 22.214 | 21.865 | 1100.498 | 2820.711 | 86.250 | 940.4 |
| 1 | decode8 | eager | false | torch | native | 22.544 | 22.895 | 22.539 | 1010.348 | 2722.215 | 55.146 | 908.4 |
| 1 | decode8 | eager | false | sym | native | 21.973 | 22.151 | 21.967 | 1014.889 | 2737.557 | 54.653 | 942.4 |
| 1 | decode8 | inductor | false | torch | materialized | 17.488 | 17.603 | 17.483 | 12244.492 | 13933.221 | 10066.211 | 908.4 |
| 1 | decode8 | inductor | false | torch | resident | 8.113 | 8.216 | 8.107 | 12223.296 | 13759.908 | 10056.006 | 838.4 |
| 1 | decode8 | inductor | false | sym | materialized | 17.847 | 18.004 | 17.841 | 12267.069 | 13979.376 | 10057.599 | 920.4 |
| 1 | decode8 | inductor | false | torch | lowered | 17.468 | 17.642 | 17.463 | 12469.805 | 14159.553 | 10265.975 | 908.4 |
| 1 | decode8 | inductor | false | sym | lowered | 17.870 | 18.071 | 17.864 | 12351.335 | 14073.762 | 10128.237 | 920.4 |
| 1 | decode8 | inductor | false | torch | native | 17.479 | 17.694 | 17.474 | 12236.770 | 13923.577 | 10087.280 | 908.4 |
| 1 | decode8 | inductor | false | sym | native | 19.035 | 19.148 | 19.028 | 12387.435 | 14109.165 | 10201.732 | 942.4 |
| 1 | decode64 | eager | false | torch | materialized | 23.851 | 25.230 | 23.846 | 1106.045 | 2840.598 | 59.313 | 912.4 |
| 1 | decode64 | eager | false | torch | resident | 13.730 | 13.803 | 13.724 | 971.455 | 2533.692 | 61.317 | 842.4 |
| 1 | decode64 | eager | false | sym | materialized | 22.783 | 23.051 | 22.777 | 1078.536 | 2811.875 | 56.857 | 944.4 |
| 1 | decode64 | eager | false | torch | lowered | 23.321 | 23.525 | 23.313 | 1084.680 | 2793.735 | 85.021 | 912.4 |
| 1 | decode64 | eager | false | sym | lowered | 22.669 | 23.235 | 22.663 | 1105.977 | 2829.829 | 86.127 | 944.4 |
| 1 | decode64 | eager | false | torch | native | 25.191 | 25.461 | 25.186 | 993.600 | 2716.295 | 54.636 | 912.4 |
| 1 | decode64 | eager | false | sym | native | 22.302 | 22.450 | 22.296 | 1016.779 | 2732.327 | 55.048 | 948.4 |
| 1 | decode64 | inductor | false | torch | materialized | 17.783 | 17.958 | 17.777 | 13021.228 | 14706.280 | 10858.254 | 912.4 |
| 1 | decode64 | inductor | false | torch | resident | 8.209 | 8.296 | 8.204 | 13064.676 | 14606.757 | 10957.481 | 842.4 |
| 1 | decode64 | inductor | false | sym | materialized | 18.030 | 18.382 | 18.024 | 13010.482 | 14710.016 | 10835.937 | 924.4 |
| 1 | decode64 | inductor | false | torch | lowered | 17.704 | 17.892 | 17.700 | 13109.114 | 14798.951 | 10929.980 | 912.4 |
| 1 | decode64 | inductor | false | sym | lowered | 18.062 | 18.401 | 18.056 | 13088.242 | 14795.509 | 10846.584 | 924.4 |
| 1 | decode64 | inductor | false | torch | native | 17.738 | 17.836 | 17.733 | 12837.564 | 14523.733 | 10756.259 | 912.4 |
| 1 | decode64 | inductor | false | sym | native | 18.318 | 18.692 | 18.311 | 12882.033 | 14578.829 | 10762.467 | 928.4 |
| 1 | prefill256 | eager | false | torch | materialized | 5.922 | 6.629 | 5.917 | 388.564 | 2091.556 | 32.037 | 932.4 |
| 1 | prefill256 | eager | false | torch | resident | 3.968 | 4.308 | 3.963 | 376.179 | 1905.273 | 32.066 | 858.4 |
| 1 | prefill256 | eager | false | sym | materialized | 5.647 | 6.173 | 5.640 | 402.702 | 2058.446 | 32.342 | 944.4 |
| 1 | prefill256 | eager | false | torch | lowered | 5.985 | 6.089 | 5.980 | 395.470 | 2050.630 | 38.250 | 932.4 |
| 1 | prefill256 | eager | false | sym | lowered | 5.581 | 5.698 | 5.574 | 408.965 | 2068.789 | 38.537 | 944.4 |
| 1 | prefill256 | eager | false | torch | native | 5.898 | 6.299 | 5.894 | 382.998 | 2026.704 | 32.319 | 932.4 |
| 1 | prefill256 | eager | false | sym | native | 5.680 | 5.766 | 5.674 | 395.920 | 2050.958 | 32.562 | 944.4 |
| 1 | prefill256 | inductor | false | torch | materialized | 4.851 | 5.000 | 4.846 | 5138.306 | 6783.607 | 4511.452 | 932.4 |
| 1 | prefill256 | inductor | false | torch | resident | 2.316 | 2.479 | 2.311 | 5204.594 | 6725.535 | 4569.745 | 858.4 |
| 1 | prefill256 | inductor | false | sym | materialized | 4.894 | 4.976 | 4.888 | 5146.753 | 6798.662 | 4514.776 | 944.4 |
| 1 | prefill256 | inductor | false | torch | lowered | 4.858 | 4.885 | 4.851 | 5204.818 | 6848.001 | 4563.480 | 932.4 |
| 1 | prefill256 | inductor | false | sym | lowered | 4.889 | 5.005 | 4.883 | 5144.248 | 6791.716 | 4511.821 | 944.4 |
| 1 | prefill256 | inductor | false | torch | native | 4.875 | 4.966 | 4.870 | 5111.677 | 6752.270 | 4500.929 | 932.4 |
| 1 | prefill256 | inductor | false | sym | native | 4.889 | 4.990 | 4.883 | 5130.553 | 6800.685 | 4506.310 | 944.4 |
| 1 | decode8 | eager | true | torch | materialized | 18.028 | 18.790 | 18.023 | 1145.688 | 2943.184 | 58.208 | 1454.4 |
| 1 | decode8 | eager | true | sym | materialized | 18.754 | 19.399 | 18.748 | 1149.212 | 2926.841 | 53.275 | 1466.4 |
| 1 | decode8 | eager | true | torch | lowered | 17.707 | 22.112 | 17.702 | 1217.746 | 3037.466 | 90.291 | 1454.4 |
| 1 | decode8 | eager | true | sym | lowered | 18.777 | 19.415 | 18.770 | 1178.733 | 2961.343 | 82.532 | 1486.4 |
| 1 | decode8 | eager | true | torch | native | 17.679 | 18.128 | 17.674 | 1071.134 | 2830.374 | 54.335 | 1454.4 |
| 1 | decode8 | eager | true | sym | native | 19.184 | 19.738 | 19.178 | 1101.664 | 2884.092 | 51.690 | 1468.4 |
| 1 | decode8 | inductor | true | torch | materialized | 17.689 | 18.289 | 17.684 | 12345.323 | 14106.687 | 10071.833 | 1454.4 |
| 1 | decode8 | inductor | true | sym | materialized | 18.819 | 19.918 | 18.812 | 12385.706 | 14170.357 | 10088.151 | 1486.4 |
| 1 | decode8 | inductor | true | torch | lowered | 17.667 | 18.167 | 17.663 | 12341.312 | 14103.785 | 10085.277 | 1454.4 |
| 1 | decode8 | inductor | true | sym | lowered | 19.118 | 19.883 | 19.111 | 12445.297 | 14235.696 | 10136.282 | 1486.4 |
| 1 | decode8 | inductor | true | torch | native | 17.661 | 18.131 | 17.655 | 12271.329 | 14025.822 | 10059.816 | 1454.4 |
| 1 | decode8 | inductor | true | sym | native | 19.652 | 21.366 | 19.646 | 12321.052 | 14127.085 | 10040.196 | 1488.4 |
| 2 | decode8 | eager | false | sym | lowered | 22.022 | 22.236 | 22.015 | 1104.805 | 2822.288 | 85.874 | 940.4 |
| 2 | decode8 | eager | false | torch | lowered | 22.909 | 23.817 | 22.904 | 1079.331 | 2796.425 | 84.701 | 908.4 |
| 2 | decode8 | eager | false | torch | resident | 13.547 | 13.651 | 13.543 | 972.603 | 2542.116 | 54.588 | 838.4 |
| 2 | decode8 | eager | false | sym | native | 22.115 | 22.366 | 22.108 | 1014.609 | 2739.010 | 54.737 | 942.4 |
| 2 | decode8 | eager | false | torch | native | 22.631 | 23.030 | 22.626 | 987.595 | 2701.084 | 54.844 | 908.4 |
| 2 | decode8 | eager | false | sym | materialized | 22.489 | 22.677 | 22.483 | 1069.608 | 2796.335 | 56.740 | 940.4 |
| 2 | decode8 | eager | false | torch | materialized | 23.245 | 23.656 | 23.240 | 1085.167 | 2857.163 | 59.530 | 908.4 |
| 2 | decode8 | inductor | false | sym | lowered | 18.810 | 19.531 | 18.803 | 12340.201 | 14040.620 | 10130.001 | 920.4 |
| 2 | decode8 | inductor | false | torch | lowered | 17.482 | 17.689 | 17.476 | 12293.152 | 14038.955 | 10094.573 | 908.4 |
| 2 | decode8 | inductor | false | torch | resident | 8.292 | 8.363 | 8.287 | 12173.552 | 13712.259 | 10020.036 | 838.4 |
| 2 | decode8 | inductor | false | sym | native | 18.018 | 18.564 | 18.011 | 12287.883 | 13987.748 | 10104.406 | 942.4 |
| 2 | decode8 | inductor | false | torch | native | 19.340 | 20.646 | 19.334 | 12176.126 | 13874.459 | 10028.587 | 908.4 |
| 2 | decode8 | inductor | false | sym | materialized | 17.864 | 18.259 | 17.858 | 12314.757 | 14009.439 | 10085.078 | 940.4 |
| 2 | decode8 | inductor | false | torch | materialized | 17.466 | 17.550 | 17.461 | 12323.478 | 14005.387 | 10119.896 | 908.4 |
| 2 | decode64 | eager | false | sym | lowered | 22.265 | 22.592 | 22.259 | 1102.787 | 2818.482 | 85.269 | 944.4 |
| 2 | decode64 | eager | false | torch | lowered | 23.310 | 23.946 | 23.306 | 1087.807 | 2803.146 | 85.056 | 912.4 |
| 2 | decode64 | eager | false | torch | resident | 13.877 | 13.968 | 13.872 | 971.940 | 2535.472 | 54.763 | 842.4 |
| 2 | decode64 | eager | false | sym | native | 22.419 | 25.922 | 22.413 | 1056.536 | 2788.368 | 55.793 | 948.4 |
| 2 | decode64 | eager | false | torch | native | 22.882 | 24.038 | 22.878 | 988.356 | 2705.260 | 54.581 | 912.4 |
| 2 | decode64 | eager | false | sym | materialized | 22.534 | 22.899 | 22.528 | 1074.905 | 2797.768 | 57.052 | 944.4 |
| 2 | decode64 | eager | false | torch | materialized | 23.581 | 23.732 | 23.577 | 1051.975 | 2766.955 | 55.781 | 912.4 |
| 2 | decode64 | inductor | false | sym | lowered | 18.043 | 18.555 | 18.037 | 13038.699 | 14737.833 | 10835.244 | 924.4 |
| 2 | decode64 | inductor | false | torch | lowered | 17.717 | 17.766 | 17.712 | 13206.638 | 14897.188 | 10951.129 | 912.4 |
| 2 | decode64 | inductor | false | torch | resident | 9.457 | 11.057 | 9.450 | 12922.418 | 14459.964 | 10802.397 | 842.4 |
| 2 | decode64 | inductor | false | sym | native | 18.445 | 18.753 | 18.425 | 12957.393 | 14657.680 | 10809.594 | 928.4 |
| 2 | decode64 | inductor | false | torch | native | 17.745 | 17.830 | 17.741 | 13083.801 | 14783.413 | 10930.649 | 912.4 |
| 2 | decode64 | inductor | false | sym | materialized | 18.049 | 18.398 | 18.042 | 13037.846 | 14742.358 | 10857.653 | 924.4 |
| 2 | decode64 | inductor | false | torch | materialized | 17.714 | 17.865 | 17.709 | 13009.355 | 14694.880 | 10852.646 | 912.4 |
| 2 | prefill256 | eager | false | sym | lowered | 5.593 | 5.770 | 5.587 | 408.208 | 2065.790 | 38.327 | 944.4 |
| 2 | prefill256 | eager | false | torch | lowered | 5.990 | 6.299 | 5.985 | 395.302 | 2057.104 | 38.031 | 932.4 |
| 2 | prefill256 | eager | false | torch | resident | 3.987 | 4.087 | 3.982 | 397.406 | 1937.125 | 34.501 | 858.4 |
| 2 | prefill256 | eager | false | sym | native | 5.604 | 5.720 | 5.598 | 396.038 | 2052.151 | 32.467 | 944.4 |
| 2 | prefill256 | eager | false | torch | native | 6.144 | 6.699 | 6.139 | 384.121 | 2035.373 | 32.259 | 932.4 |
| 2 | prefill256 | eager | false | sym | materialized | 5.692 | 5.795 | 5.686 | 401.644 | 2073.070 | 32.464 | 944.4 |
| 2 | prefill256 | eager | false | torch | materialized | 5.945 | 6.479 | 5.940 | 389.856 | 2037.396 | 32.124 | 932.4 |
| 2 | prefill256 | inductor | false | sym | lowered | 4.883 | 4.978 | 4.877 | 5185.545 | 6833.786 | 4549.345 | 944.4 |
| 2 | prefill256 | inductor | false | torch | lowered | 4.856 | 4.923 | 4.852 | 5172.203 | 6822.362 | 4530.034 | 932.4 |
| 2 | prefill256 | inductor | false | torch | resident | 2.365 | 2.545 | 2.360 | 5181.937 | 6702.434 | 4550.443 | 858.4 |
| 2 | prefill256 | inductor | false | sym | native | 4.885 | 4.980 | 4.879 | 5127.212 | 6841.488 | 4499.525 | 944.4 |
| 2 | prefill256 | inductor | false | torch | native | 4.853 | 4.911 | 4.848 | 5210.705 | 6857.591 | 4571.487 | 932.4 |
| 2 | prefill256 | inductor | false | sym | materialized | 4.882 | 4.996 | 4.876 | 5145.926 | 6806.288 | 4511.714 | 944.4 |
| 2 | prefill256 | inductor | false | torch | materialized | 4.866 | 4.971 | 4.861 | 5139.771 | 6775.426 | 4511.729 | 932.4 |
| 2 | decode8 | eager | true | sym | lowered | 18.776 | 19.441 | 18.770 | 1173.990 | 2957.462 | 82.542 | 1466.4 |
| 2 | decode8 | eager | true | torch | lowered | 17.691 | 18.184 | 17.686 | 1155.291 | 2911.730 | 86.068 | 1454.4 |
| 2 | decode8 | eager | true | sym | native | 19.039 | 19.803 | 19.033 | 1099.411 | 2885.168 | 52.406 | 1488.4 |
| 2 | decode8 | eager | true | torch | native | 17.679 | 18.035 | 17.675 | 1075.829 | 2834.626 | 54.202 | 1454.4 |
| 2 | decode8 | eager | true | sym | materialized | 18.747 | 19.658 | 18.742 | 1155.256 | 2933.274 | 53.397 | 1486.4 |
| 2 | decode8 | eager | true | torch | materialized | 17.722 | 18.018 | 17.718 | 1194.151 | 2951.607 | 57.017 | 1454.4 |
| 2 | decode8 | inductor | true | sym | lowered | 19.119 | 19.469 | 19.112 | 12559.586 | 14350.005 | 10224.388 | 1486.4 |
| 2 | decode8 | inductor | true | torch | lowered | 17.676 | 18.243 | 17.671 | 12545.751 | 14308.768 | 10223.953 | 1454.4 |
| 2 | decode8 | inductor | true | sym | native | 19.027 | 19.838 | 19.021 | 12365.557 | 14170.624 | 10094.013 | 1468.4 |
| 2 | decode8 | inductor | true | torch | native | 17.668 | 18.221 | 17.663 | 12276.329 | 14036.926 | 10034.108 | 1454.4 |
| 2 | decode8 | inductor | true | sym | materialized | 18.792 | 18.968 | 18.786 | 12332.095 | 14113.717 | 10039.885 | 1486.4 |
| 2 | decode8 | inductor | true | torch | materialized | 17.691 | 18.202 | 17.687 | 12455.974 | 14238.764 | 10169.784 | 1454.4 |
| 3 | decode8 | eager | false | torch | native | 22.595 | 22.751 | 22.590 | 987.691 | 2701.078 | 54.865 | 908.4 |
| 3 | decode8 | eager | false | sym | native | 21.918 | 22.163 | 21.912 | 1015.294 | 2736.217 | 55.042 | 942.4 |
| 3 | decode8 | eager | false | torch | materialized | 28.335 | 33.868 | 28.326 | 1052.641 | 2772.940 | 56.227 | 908.4 |
| 3 | decode8 | eager | false | torch | resident | 13.599 | 13.690 | 13.585 | 970.904 | 2530.843 | 54.447 | 838.4 |
| 3 | decode8 | eager | false | sym | materialized | 22.554 | 22.989 | 22.547 | 1077.790 | 2806.123 | 57.008 | 940.4 |
| 3 | decode8 | eager | false | torch | lowered | 22.845 | 22.983 | 22.836 | 1080.400 | 2796.433 | 84.853 | 908.4 |
| 3 | decode8 | eager | false | sym | lowered | 21.961 | 22.177 | 21.955 | 1102.287 | 2830.408 | 82.410 | 940.4 |
| 3 | decode8 | inductor | false | torch | native | 17.468 | 17.543 | 17.463 | 12168.824 | 13859.798 | 10040.754 | 908.4 |
| 3 | decode8 | inductor | false | sym | native | 18.062 | 18.521 | 18.056 | 12288.551 | 14049.118 | 10111.367 | 922.4 |
| 3 | decode8 | inductor | false | torch | materialized | 17.489 | 18.291 | 17.484 | 12513.620 | 14203.264 | 10248.752 | 908.4 |
| 3 | decode8 | inductor | false | torch | resident | 8.218 | 8.312 | 8.213 | 12153.781 | 13689.203 | 10017.923 | 838.4 |
| 3 | decode8 | inductor | false | sym | materialized | 17.846 | 18.002 | 17.840 | 12267.570 | 14014.284 | 10053.258 | 920.4 |
| 3 | decode8 | inductor | false | torch | lowered | 17.525 | 17.681 | 17.521 | 12459.790 | 14155.047 | 10210.005 | 908.4 |
| 3 | decode8 | inductor | false | sym | lowered | 17.999 | 18.379 | 17.993 | 12280.490 | 13982.917 | 10062.571 | 920.4 |
| 3 | decode64 | eager | false | torch | native | 22.985 | 24.986 | 22.980 | 991.994 | 2705.100 | 54.592 | 912.4 |
| 3 | decode64 | eager | false | sym | native | 22.314 | 22.565 | 22.308 | 1022.384 | 2740.972 | 54.638 | 948.4 |
| 3 | decode64 | eager | false | torch | materialized | 23.708 | 23.984 | 23.703 | 1058.915 | 2782.432 | 56.647 | 912.4 |
| 3 | decode64 | eager | false | torch | resident | 13.730 | 13.785 | 13.724 | 975.725 | 2561.840 | 55.054 | 842.4 |
| 3 | decode64 | eager | false | sym | materialized | 22.392 | 22.696 | 22.385 | 1076.419 | 2797.837 | 64.188 | 944.4 |
| 3 | decode64 | eager | false | torch | lowered | 23.479 | 23.613 | 23.473 | 1086.711 | 2797.386 | 84.786 | 912.4 |
| 3 | decode64 | eager | false | sym | lowered | 22.101 | 22.361 | 22.095 | 1113.714 | 2833.951 | 85.728 | 944.4 |
| 3 | decode64 | inductor | false | torch | native | 19.522 | 19.740 | 19.517 | 12949.634 | 14644.878 | 10821.555 | 912.4 |
| 3 | decode64 | inductor | false | sym | native | 18.407 | 18.758 | 18.398 | 12949.688 | 14647.912 | 10812.698 | 928.4 |
| 3 | decode64 | inductor | false | torch | materialized | 17.778 | 17.987 | 17.772 | 12989.150 | 14684.710 | 10826.243 | 912.4 |
| 3 | decode64 | inductor | false | torch | resident | 8.190 | 9.130 | 8.185 | 12916.316 | 14450.955 | 10836.651 | 842.4 |
| 3 | decode64 | inductor | false | sym | materialized | 18.102 | 18.556 | 18.096 | 13077.175 | 14777.319 | 10850.258 | 924.4 |
| 3 | decode64 | inductor | false | torch | lowered | 17.747 | 17.839 | 17.742 | 12998.836 | 14683.630 | 10846.821 | 912.4 |
| 3 | decode64 | inductor | false | sym | lowered | 17.973 | 18.353 | 17.967 | 13002.043 | 14698.878 | 10834.810 | 924.4 |
| 3 | prefill256 | eager | false | torch | native | 5.962 | 6.119 | 5.957 | 383.978 | 2049.967 | 32.315 | 932.4 |
| 3 | prefill256 | eager | false | sym | native | 5.610 | 6.009 | 5.603 | 396.028 | 2045.029 | 32.431 | 944.4 |
| 3 | prefill256 | eager | false | torch | materialized | 5.902 | 6.021 | 5.897 | 388.687 | 2042.835 | 32.168 | 932.4 |
| 3 | prefill256 | eager | false | torch | resident | 4.043 | 4.122 | 4.023 | 377.690 | 1916.265 | 32.286 | 858.4 |
| 3 | prefill256 | eager | false | sym | materialized | 6.105 | 6.307 | 6.098 | 424.692 | 2107.709 | 34.643 | 944.4 |
| 3 | prefill256 | eager | false | torch | lowered | 5.945 | 6.126 | 5.941 | 395.164 | 2037.676 | 37.827 | 932.4 |
| 3 | prefill256 | eager | false | sym | lowered | 5.657 | 5.837 | 5.650 | 409.171 | 2063.292 | 38.439 | 944.4 |
| 3 | prefill256 | inductor | false | torch | native | 4.871 | 4.918 | 4.866 | 5184.845 | 6840.595 | 4571.445 | 932.4 |
| 3 | prefill256 | inductor | false | sym | native | 4.893 | 4.994 | 4.887 | 5123.606 | 6774.991 | 4499.301 | 944.4 |
| 3 | prefill256 | inductor | false | torch | materialized | 4.848 | 4.890 | 4.843 | 5142.050 | 6780.778 | 4518.759 | 932.4 |
| 3 | prefill256 | inductor | false | torch | resident | 2.391 | 2.614 | 2.385 | 5108.210 | 6639.398 | 4494.236 | 858.4 |
| 3 | prefill256 | inductor | false | sym | materialized | 4.874 | 4.989 | 4.868 | 5130.831 | 6783.876 | 4499.075 | 944.4 |
| 3 | prefill256 | inductor | false | torch | lowered | 4.869 | 4.970 | 4.865 | 5164.457 | 6821.881 | 4539.170 | 932.4 |
| 3 | prefill256 | inductor | false | sym | lowered | 4.905 | 5.075 | 4.899 | 5144.238 | 6797.835 | 4514.885 | 944.4 |
| 3 | decode8 | eager | true | torch | native | 17.683 | 18.095 | 17.678 | 1070.966 | 2859.779 | 54.152 | 1454.4 |
| 3 | decode8 | eager | true | sym | native | 19.176 | 19.799 | 19.170 | 1100.351 | 2882.304 | 51.994 | 1468.4 |
| 3 | decode8 | eager | true | torch | materialized | 17.689 | 18.028 | 17.684 | 1122.313 | 2880.595 | 55.806 | 1454.4 |
| 3 | decode8 | eager | true | sym | materialized | 18.752 | 19.541 | 18.746 | 1146.112 | 2924.300 | 53.766 | 1466.4 |
| 3 | decode8 | eager | true | torch | lowered | 18.422 | 18.801 | 18.417 | 1173.311 | 2958.898 | 93.188 | 1454.4 |
| 3 | decode8 | eager | true | sym | lowered | 18.729 | 19.233 | 18.723 | 1176.239 | 2973.362 | 82.962 | 1486.4 |
| 3 | decode8 | inductor | true | torch | native | 17.724 | 18.164 | 17.709 | 12258.644 | 14013.858 | 10046.063 | 1454.4 |
| 3 | decode8 | inductor | true | sym | native | 19.197 | 19.951 | 19.190 | 12510.602 | 14294.097 | 10226.987 | 1468.4 |
| 3 | decode8 | inductor | true | torch | materialized | 17.666 | 18.188 | 17.661 | 12437.245 | 14256.688 | 10114.034 | 1454.4 |
| 3 | decode8 | inductor | true | sym | materialized | 18.728 | 19.723 | 18.722 | 12534.911 | 14313.473 | 10190.202 | 1486.4 |
| 3 | decode8 | inductor | true | torch | lowered | 17.715 | 18.329 | 17.710 | 12410.033 | 14171.864 | 10122.645 | 1454.4 |
| 3 | decode8 | inductor | true | sym | lowered | 18.757 | 19.752 | 18.751 | 12483.220 | 14267.400 | 10170.919 | 1486.4 |

## Candidate versus materialized controls

Ratio = control p50 / lowered p50. Break-even uses preparation + first + warmup and a positive measured warm advantage; it is an estimate, not a measured multi-request amortization experiment.

| Round | Case | Compute | Replay | Runtime | Control | Ratio | Break-even requests |
|---|---|---|---|---|---|---:|---:|
| 1 | decode8 | eager | false | torch | native | 0.984 | none |
| 1 | decode8 | eager | false | torch | materialized | 1.015 | 101 |
| 1 | decode8 | eager | false | sym | native | 1.005 | 814 |
| 1 | decode8 | eager | false | sym | materialized | 1.018 | 74 |
| 1 | decode8 | inductor | false | torch | native | 1.001 | 21235 |
| 1 | decode8 | inductor | false | torch | materialized | 1.001 | 11381 |
| 1 | decode8 | inductor | false | sym | native | 1.065 | 0 |
| 1 | decode8 | inductor | false | sym | materialized | 0.999 | none |
| 1 | decode64 | eager | false | torch | native | 1.080 | 42 |
| 1 | decode64 | eager | false | torch | materialized | 1.023 | 0 |
| 1 | decode64 | eager | false | sym | native | 0.984 | none |
| 1 | decode64 | eager | false | sym | materialized | 1.005 | 159 |
| 1 | decode64 | inductor | false | torch | native | 1.002 | 8241 |
| 1 | decode64 | inductor | false | torch | materialized | 1.004 | 1187 |
| 1 | decode64 | inductor | false | sym | native | 1.014 | 846 |
| 1 | decode64 | inductor | false | sym | materialized | 0.998 | none |
| 1 | prefill256 | eager | false | torch | native | 0.985 | none |
| 1 | prefill256 | eager | false | torch | materialized | 0.989 | none |
| 1 | prefill256 | eager | false | sym | native | 1.018 | 180 |
| 1 | prefill256 | eager | false | sym | materialized | 1.012 | 157 |
| 1 | prefill256 | inductor | false | torch | native | 1.003 | 5693 |
| 1 | prefill256 | inductor | false | torch | materialized | 0.998 | none |
| 1 | prefill256 | inductor | false | sym | native | 1.000 | 0 |
| 1 | prefill256 | inductor | false | sym | materialized | 1.001 | 0 |
| 1 | decode8 | eager | true | torch | native | 0.998 | none |
| 1 | decode8 | eager | true | torch | materialized | 1.018 | 294 |
| 1 | decode8 | eager | true | sym | native | 1.022 | 190 |
| 1 | decode8 | eager | true | sym | materialized | 0.999 | none |
| 1 | decode8 | inductor | true | torch | native | 1.000 | none |
| 1 | decode8 | inductor | true | torch | materialized | 1.001 | 0 |
| 1 | decode8 | inductor | true | sym | native | 1.028 | 204 |
| 1 | decode8 | inductor | true | sym | materialized | 0.984 | none |
| 2 | decode8 | eager | false | sym | native | 1.004 | 896 |
| 2 | decode8 | eager | false | sym | materialized | 1.021 | 56 |
| 2 | decode8 | eager | false | torch | native | 0.988 | none |
| 2 | decode8 | eager | false | torch | materialized | 1.015 | 0 |
| 2 | decode8 | inductor | false | sym | native | 0.958 | none |
| 2 | decode8 | inductor | false | sym | materialized | 0.950 | none |
| 2 | decode8 | inductor | false | torch | native | 1.106 | 89 |
| 2 | decode8 | inductor | false | torch | materialized | 0.999 | none |
| 2 | decode64 | eager | false | sym | native | 1.007 | 196 |
| 2 | decode64 | eager | false | sym | materialized | 1.012 | 77 |
| 2 | decode64 | eager | false | torch | native | 0.982 | none |
| 2 | decode64 | eager | false | torch | materialized | 1.012 | 134 |
| 2 | decode64 | inductor | false | sym | native | 1.022 | 200 |
| 2 | decode64 | inductor | false | sym | materialized | 1.000 | 0 |
| 2 | decode64 | inductor | false | torch | native | 1.002 | 3974 |
| 2 | decode64 | inductor | false | torch | materialized | 1.000 | none |
| 2 | prefill256 | eager | false | sym | native | 1.002 | 1164 |
| 2 | prefill256 | eager | false | sym | materialized | 1.018 | 0 |
| 2 | prefill256 | eager | false | torch | native | 1.026 | 142 |
| 2 | prefill256 | eager | false | torch | materialized | 0.992 | none |
| 2 | prefill256 | inductor | false | sym | native | 1.000 | 0 |
| 2 | prefill256 | inductor | false | sym | materialized | 1.000 | none |
| 2 | prefill256 | inductor | false | torch | native | 0.999 | none |
| 2 | prefill256 | inductor | false | torch | materialized | 1.002 | 5151 |
| 2 | decode8 | eager | true | sym | native | 1.014 | 275 |
| 2 | decode8 | eager | true | sym | materialized | 0.998 | none |
| 2 | decode8 | eager | true | torch | native | 0.999 | none |
| 2 | decode8 | eager | true | torch | materialized | 1.002 | 0 |
| 2 | decode8 | inductor | true | sym | native | 0.995 | none |
| 2 | decode8 | inductor | true | sym | materialized | 0.983 | none |
| 2 | decode8 | inductor | true | torch | native | 1.000 | none |
| 2 | decode8 | inductor | true | torch | materialized | 1.001 | 4523 |
| 3 | decode8 | eager | false | torch | native | 0.989 | none |
| 3 | decode8 | eager | false | torch | materialized | 1.240 | 5 |
| 3 | decode8 | eager | false | sym | native | 0.998 | none |
| 3 | decode8 | eager | false | sym | materialized | 1.027 | 41 |
| 3 | decode8 | inductor | false | torch | native | 0.997 | none |
| 3 | decode8 | inductor | false | torch | materialized | 0.998 | none |
| 3 | decode8 | inductor | false | sym | native | 1.003 | 0 |
| 3 | decode8 | inductor | false | sym | materialized | 0.992 | none |
| 3 | decode64 | eager | false | torch | native | 0.979 | none |
| 3 | decode64 | eager | false | torch | materialized | 1.010 | 66 |
| 3 | decode64 | eager | false | sym | native | 1.010 | 436 |
| 3 | decode64 | eager | false | sym | materialized | 1.013 | 125 |
| 3 | decode64 | inductor | false | torch | native | 1.100 | 22 |
| 3 | decode64 | inductor | false | torch | materialized | 1.002 | 0 |
| 3 | decode64 | inductor | false | sym | native | 1.024 | 118 |
| 3 | decode64 | inductor | false | sym | materialized | 1.007 | 0 |
| 3 | prefill256 | eager | false | torch | native | 1.003 | 0 |
| 3 | prefill256 | eager | false | torch | materialized | 0.993 | none |
| 3 | prefill256 | eager | false | sym | native | 0.992 | none |
| 3 | prefill256 | eager | false | sym | materialized | 1.079 | 0 |
| 3 | prefill256 | inductor | false | torch | native | 1.000 | 0 |
| 3 | prefill256 | inductor | false | torch | materialized | 0.996 | none |
| 3 | prefill256 | inductor | false | sym | native | 0.998 | none |
| 3 | prefill256 | inductor | false | sym | materialized | 0.994 | none |
| 3 | decode8 | eager | true | torch | native | 0.960 | none |
| 3 | decode8 | eager | true | torch | materialized | 0.960 | none |
| 3 | decode8 | eager | true | sym | native | 1.024 | 204 |
| 3 | decode8 | eager | true | sym | materialized | 1.001 | 2125 |
| 3 | decode8 | inductor | true | torch | native | 1.001 | 16151 |
| 3 | decode8 | inductor | true | torch | materialized | 0.997 | none |
| 3 | decode8 | inductor | true | sym | native | 1.023 | 0 |
| 3 | decode8 | inductor | true | sym | materialized | 0.998 | none |

## Parent target

A pass needs at least 5% lower p50 than both matched eager and Inductor Torch controls in every round.

| Case | Compute | Replay | Variant | Round passes | Qualified |
|---|---|---|---|---|---|
| decode8 | eager | False | native | [False, False, False] | False |
| decode8 | eager | False | materialized | [False, False, False] | False |
| decode8 | eager | False | lowered | [False, False, False] | False |
| decode8 | eager | True | native | [False, False, False] | False |
| decode8 | eager | True | materialized | [False, False, False] | False |
| decode8 | eager | True | lowered | [False, False, False] | False |
| decode8 | inductor | False | native | [False, True, False] | False |
| decode8 | inductor | False | materialized | [False, False, False] | False |
| decode8 | inductor | False | lowered | [False, False, False] | False |
| decode8 | inductor | True | native | [False, False, False] | False |
| decode8 | inductor | True | materialized | [False, False, False] | False |
| decode8 | inductor | True | lowered | [False, False, False] | False |
| decode64 | eager | False | native | [False, False, False] | False |
| decode64 | eager | False | materialized | [False, False, False] | False |
| decode64 | eager | False | lowered | [False, False, False] | False |
| decode64 | inductor | False | native | [False, False, True] | False |
| decode64 | inductor | False | materialized | [False, False, False] | False |
| decode64 | inductor | False | lowered | [False, False, False] | False |
| prefill256 | eager | False | native | [False, False, False] | False |
| prefill256 | eager | False | materialized | [False, False, False] | False |
| prefill256 | eager | False | lowered | [False, False, False] | False |
| prefill256 | inductor | False | native | [False, False, False] | False |
| prefill256 | inductor | False | materialized | [False, False, False] | False |
| prefill256 | inductor | False | lowered | [False, False, False] | False |

## KV supplements

Each successful row has 100 completed requests plus separate 100-call consumer-only and restore intervals. Isolated intervals are never subtracted from model time. OOM rows retain their actual phase; all use four changing input sets and bitwise cache checks.

| Round | Compute | Shape | Variant | Outcome | p50 | p95 | Consumer p50 | Restore p50 | Setup | Warm peak MiB |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | eager | canonical | materialized | success | 0.342 | 0.456 | 0.192 | 0.157 | 2454.352 | 644.4 |
| 1 | eager | canonical | lowered | success | 0.305 | 0.350 | 0.157 | 0.156 | 77.060 | 644.4 |
| 1 | eager | canonical | native | success | 0.302 | 0.328 | 0.140 | 0.167 | 61.815 | 644.4 |
| 1 | eager | held_small | materialized | success | 0.337 | 0.350 | 0.191 | 0.157 | 73.564 | 644.4 |
| 1 | eager | held_small | lowered | success | 0.305 | 0.417 | 0.156 | 0.156 | 75.262 | 644.4 |
| 1 | eager | held_small | native | success | 0.301 | 0.428 | 0.141 | 0.167 | 60.618 | 644.4 |
| 1 | eager | held_long | materialized | success | 1.705 | 1.732 | 0.377 | 1.578 | 661.036 | 720.4 |
| 1 | eager | held_long | lowered | success | 1.652 | 1.669 | 0.329 | 1.576 | 431.977 | 700.4 |
| 1 | eager | held_long | native | success | 1.596 | 1.608 | 0.285 | 1.571 | 404.130 | 720.4 |
| 1 | eager | mixed | materialized | success | 0.585 | 0.669 | 0.277 | 0.348 | 117.359 | 668.4 |
| 1 | eager | mixed | lowered | success | 0.562 | 0.669 | 0.263 | 0.340 | 108.558 | 668.4 |
| 1 | eager | mixed | native | success | 0.510 | 0.605 | 0.208 | 0.293 | 92.939 | 668.4 |
| 1 | eager | pressure | materialized | success | 3.134 | 3.256 | 0.545 | 2.912 | 1215.560 | 11260.4 |
| 1 | eager | pressure | lowered | success | 3.041 | 3.080 | 0.458 | 2.912 | 789.177 | 11242.4 |
| 1 | eager | pressure | native | oom | warm | — | — | — | — | — |
| 1 | inductor | canonical | materialized | success | 0.300 | 0.317 | 0.151 | 0.159 | 4052.449 | 664.4 |
| 1 | inductor | canonical | lowered | success | 0.304 | 0.317 | 0.152 | 0.159 | 385.889 | 664.4 |
| 1 | inductor | canonical | native | success | 0.316 | 0.332 | 0.150 | 0.171 | 367.401 | 664.4 |
| 1 | inductor | held_small | materialized | success | 0.300 | 0.411 | 0.150 | 0.159 | 423.565 | 664.4 |
| 1 | inductor | held_small | lowered | success | 0.302 | 0.392 | 0.152 | 0.159 | 378.125 | 664.4 |
| 1 | inductor | held_small | native | success | 0.314 | 0.388 | 0.151 | 0.169 | 372.377 | 664.4 |
| 1 | inductor | held_long | materialized | success | 1.609 | 1.626 | 0.297 | 1.586 | 1001.674 | 700.4 |
| 1 | inductor | held_long | lowered | success | 1.613 | 1.636 | 0.302 | 1.590 | 740.227 | 700.4 |
| 1 | inductor | held_long | native | success | 1.592 | 1.613 | 0.297 | 1.566 | 718.845 | 720.4 |
| 1 | inductor | mixed | materialized | success | 0.547 | 0.655 | 0.248 | 0.352 | 667.570 | 668.4 |
| 1 | inductor | mixed | lowered | success | 0.559 | 0.645 | 0.291 | 0.372 | 124.529 | 668.4 |
| 1 | inductor | mixed | native | success | 0.587 | 0.634 | 0.246 | 0.302 | 426.442 | 668.4 |
| 1 | inductor | pressure | materialized | success | 2.968 | 3.001 | 0.380 | 2.922 | 1508.705 | 11244.4 |
| 1 | inductor | pressure | lowered | success | 2.959 | 2.984 | 0.386 | 2.919 | 1132.615 | 11244.4 |
| 1 | inductor | pressure | native | oom | warm | — | — | — | — | — |
| 2 | eager | canonical | lowered | success | 0.306 | 0.317 | 0.157 | 0.158 | 2440.142 | 622.4 |
| 2 | eager | canonical | native | success | 0.305 | 0.318 | 0.141 | 0.170 | 62.688 | 622.4 |
| 2 | eager | canonical | materialized | success | 0.344 | 0.469 | 0.192 | 0.160 | 93.408 | 644.4 |
| 2 | eager | held_small | lowered | success | 0.304 | 0.419 | 0.156 | 0.158 | 76.106 | 644.4 |
| 2 | eager | held_small | native | success | 0.306 | 0.428 | 0.141 | 0.169 | 61.059 | 644.4 |
| 2 | eager | held_small | materialized | success | 0.343 | 0.376 | 0.192 | 0.159 | 73.782 | 644.4 |
| 2 | eager | held_long | lowered | success | 1.661 | 1.683 | 0.339 | 1.594 | 687.381 | 700.4 |
| 2 | eager | held_long | native | success | 1.599 | 1.628 | 0.303 | 1.583 | 422.697 | 720.4 |
| 2 | eager | held_long | materialized | success | 1.693 | 1.719 | 0.397 | 1.590 | 417.105 | 720.4 |
| 2 | eager | mixed | lowered | success | 0.590 | 0.760 | 0.290 | 0.355 | 119.864 | 668.4 |
| 2 | eager | mixed | native | success | 0.554 | 0.624 | 0.237 | 0.304 | 97.916 | 668.4 |
| 2 | eager | mixed | materialized | success | 0.578 | 0.684 | 0.461 | 0.362 | 108.032 | 668.4 |
| 2 | eager | pressure | lowered | success | 3.049 | 3.103 | 0.454 | 2.910 | 1198.098 | 11242.4 |
| 2 | eager | pressure | native | oom | warm | — | — | — | — | — |
| 2 | eager | pressure | materialized | success | 3.126 | 3.238 | 0.528 | 2.904 | 834.869 | 11260.4 |
| 2 | inductor | canonical | lowered | success | 0.305 | 0.318 | 0.151 | 0.160 | 4045.364 | 664.4 |
| 2 | inductor | canonical | native | success | 0.315 | 0.330 | 0.149 | 0.171 | 369.585 | 664.4 |
| 2 | inductor | canonical | materialized | success | 0.304 | 0.319 | 0.151 | 0.160 | 387.874 | 664.4 |
| 2 | inductor | held_small | lowered | success | 0.301 | 0.409 | 0.149 | 0.160 | 419.930 | 664.4 |
| 2 | inductor | held_small | native | success | 0.312 | 0.421 | 0.150 | 0.169 | 361.750 | 664.4 |
| 2 | inductor | held_small | materialized | success | 0.302 | 0.408 | 0.149 | 0.160 | 396.039 | 664.4 |
| 2 | inductor | held_long | lowered | success | 1.606 | 1.636 | 0.288 | 1.589 | 1006.658 | 700.4 |
| 2 | inductor | held_long | native | success | 1.594 | 1.616 | 0.296 | 1.557 | 707.669 | 720.4 |
| 2 | inductor | held_long | materialized | success | 1.604 | 1.635 | 0.292 | 1.578 | 741.951 | 700.4 |
| 2 | inductor | mixed | lowered | success | 0.561 | 0.683 | 0.258 | 0.361 | 673.474 | 668.4 |
| 2 | inductor | mixed | native | success | 0.534 | 0.625 | 0.246 | 0.305 | 395.436 | 668.4 |
| 2 | inductor | mixed | materialized | success | 0.547 | 0.728 | 0.280 | 0.370 | 129.457 | 668.4 |
| 2 | inductor | pressure | lowered | success | 2.963 | 3.012 | 0.377 | 2.912 | 1534.410 | 11244.4 |
| 2 | inductor | pressure | native | oom | warm | — | — | — | — | — |
| 2 | inductor | pressure | materialized | success | 2.950 | 2.984 | 0.382 | 2.907 | 1111.892 | 11244.4 |
| 3 | eager | canonical | native | success | 0.304 | 0.330 | 0.142 | 0.168 | 2419.511 | 622.4 |
| 3 | eager | canonical | materialized | success | 0.346 | 0.470 | 0.194 | 0.159 | 98.566 | 644.4 |
| 3 | eager | canonical | lowered | success | 0.307 | 0.370 | 0.157 | 0.157 | 76.325 | 644.4 |
| 3 | eager | held_small | native | success | 0.302 | 0.400 | 0.141 | 0.168 | 60.833 | 644.4 |
| 3 | eager | held_small | materialized | success | 0.343 | 0.454 | 0.192 | 0.158 | 73.734 | 644.4 |
| 3 | eager | held_small | lowered | success | 0.306 | 0.395 | 0.157 | 0.158 | 75.434 | 644.4 |
| 3 | eager | held_long | native | success | 1.609 | 1.627 | 0.277 | 1.557 | 672.165 | 720.4 |
| 3 | eager | held_long | materialized | success | 1.694 | 1.731 | 0.374 | 1.582 | 415.667 | 720.4 |
| 3 | eager | held_long | lowered | success | 1.644 | 1.665 | 0.334 | 1.570 | 425.318 | 700.4 |
| 3 | eager | mixed | native | success | 0.479 | 0.581 | 0.201 | 0.282 | 103.238 | 668.4 |
| 3 | eager | mixed | materialized | success | 0.566 | 0.676 | 0.248 | 0.330 | 104.357 | 668.4 |
| 3 | eager | mixed | lowered | success | 0.566 | 0.664 | 0.262 | 0.345 | 107.138 | 668.4 |
| 3 | eager | pressure | native | oom | warm | — | — | — | — | — |
| 3 | eager | pressure | materialized | success | 3.120 | 3.233 | 0.537 | 2.908 | 917.079 | 11260.4 |
| 3 | eager | pressure | lowered | success | 3.040 | 3.074 | 0.452 | 2.932 | 774.288 | 11242.4 |
| 3 | inductor | canonical | native | success | 0.315 | 0.328 | 0.151 | 0.170 | 4018.989 | 664.4 |
| 3 | inductor | canonical | materialized | success | 0.301 | 0.313 | 0.150 | 0.158 | 404.321 | 664.4 |
| 3 | inductor | canonical | lowered | success | 0.301 | 0.312 | 0.150 | 0.158 | 382.652 | 664.4 |
| 3 | inductor | held_small | native | success | 0.313 | 0.408 | 0.150 | 0.168 | 392.506 | 664.4 |
| 3 | inductor | held_small | materialized | success | 0.303 | 0.384 | 0.152 | 0.159 | 390.950 | 664.4 |
| 3 | inductor | held_small | lowered | success | 0.298 | 0.411 | 0.150 | 0.158 | 390.483 | 664.4 |
| 3 | inductor | held_long | native | success | 1.612 | 1.640 | 0.300 | 1.573 | 971.045 | 720.4 |
| 3 | inductor | held_long | materialized | success | 1.609 | 1.635 | 0.302 | 1.593 | 724.892 | 700.4 |
| 3 | inductor | held_long | lowered | success | 1.616 | 1.643 | 0.293 | 1.592 | 726.861 | 700.4 |
| 3 | inductor | mixed | native | success | 0.556 | 0.633 | 0.243 | 0.303 | 423.180 | 668.4 |
| 3 | inductor | mixed | materialized | success | 0.561 | 0.647 | 0.236 | 0.340 | 651.089 | 668.4 |
| 3 | inductor | mixed | lowered | success | 0.554 | 0.650 | 0.255 | 0.352 | 121.519 | 668.4 |
| 3 | inductor | pressure | native | oom | warm | — | — | — | — | — |
| 3 | inductor | pressure | materialized | success | 2.959 | 3.043 | 0.381 | 2.911 | 1133.570 | 11244.4 |
| 3 | inductor | pressure | lowered | success | 2.954 | 2.990 | 0.381 | 2.914 | 1108.367 | 11244.4 |
