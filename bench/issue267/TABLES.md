# Complete issue 267 results

Median of three process medians/p95s, in milliseconds. All cases and controls retained.

| Suite | Case | Workers | Runtime | Arm | Mode | Median ms | P95 ms |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | manual_fused | completed_consumer | 2.025576 | 2.123668 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | sym_folded | completed_consumer | 2.498483 | 2.536993 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | sym_unfused | completed_consumer | 12.756923 | 13.279848 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | torch_eager | completed_consumer | 1.674013 | 2.825535 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | torch_inductor | completed_consumer | 1.393675 | 1.743189 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | manual_fused | completed_consumer | 2.320163 | 2.567995 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | sym_folded | completed_consumer | 1.636619 | 1.664646 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | sym_unfused | completed_consumer | 12.716098 | 12.780743 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | torch_eager | completed_consumer | 1.783628 | 3.974863 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | torch_inductor | completed_consumer | 1.389636 | 1.447474 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | manual_fused | completed_consumer | 0.940604 | 1.129748 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | sym_folded | completed_consumer | 0.726163 | 1.399801 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | sym_unfused | completed_consumer | 3.229586 | 4.802640 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | torch_eager | completed_consumer | 1.568583 | 1.696088 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | torch_inductor | completed_consumer | 0.703440 | 0.900943 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | manual_fused | completed_consumer | 0.939002 | 1.293398 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | sym_folded | completed_consumer | 0.780518 | 1.058387 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | sym_unfused | completed_consumer | 3.200800 | 3.912650 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | torch_eager | completed_consumer | 2.066248 | 3.621597 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | torch_inductor | completed_consumer | 0.691898 | 0.948398 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | manual_fused | completed_consumer | 1.407917 | 1.432668 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | sym_folded | completed_consumer | 2.275040 | 2.288699 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | sym_unfused | completed_consumer | 2.695183 | 2.747444 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | torch_eager | completed_consumer | 0.999911 | 1.070544 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | torch_inductor | completed_consumer | 1.281086 | 1.299198 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | manual_fused | completed_consumer | 1.361959 | 1.397651 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | sym_folded | completed_consumer | 0.688030 | 0.700099 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | sym_unfused | completed_consumer | 2.639800 | 2.724305 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | torch_eager | completed_consumer | 1.011119 | 1.045114 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | torch_inductor | completed_consumer | 1.288847 | 1.328726 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | manual_fused | completed_consumer | 0.786787 | 1.042996 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | sym_folded | completed_consumer | 1.927822 | 2.036271 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | sym_unfused | completed_consumer | 2.729229 | 2.814617 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | torch_eager | completed_consumer | 1.096333 | 1.221135 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | torch_inductor | completed_consumer | 0.782260 | 0.799911 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | manual_fused | completed_consumer | 0.789336 | 1.071425 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | sym_folded | completed_consumer | 0.597217 | 0.713822 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | sym_unfused | completed_consumer | 2.850994 | 3.570638 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | torch_eager | completed_consumer | 1.095109 | 1.233294 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | torch_inductor | completed_consumer | 0.764061 | 0.902311 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | completed_h2d | 0.198207 | 0.209137 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | producer | 0.166723 | 0.171691 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.155512 | 0.177768 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | producer | 0.122904 | 0.136602 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | completed_h2d | 0.091014 | 0.096081 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | producer | 0.058647 | 0.065872 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.147676 | 0.154129 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | producer | 0.106963 | 0.117598 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | completed_h2d | 0.173690 | 0.183746 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | producer | 0.141009 | 0.145872 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.145137 | 0.153761 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | producer | 0.111824 | 0.119888 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | completed_h2d | 0.091755 | 0.098131 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | producer | 0.059699 | 0.066379 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.149871 | 0.161053 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | producer | 0.107164 | 0.116482 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | completed_h2d | 0.150373 | 0.160382 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | producer | 0.135593 | 0.145627 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.160627 | 0.176725 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | producer | 0.138676 | 0.148066 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | completed_h2d | 0.106667 | 0.123776 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | producer | 0.072494 | 0.090802 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.204120 | 0.217176 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | producer | 0.154044 | 0.168997 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | completed_h2d | 0.145568 | 0.160987 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | producer | 0.127679 | 0.143023 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.151194 | 0.169227 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | producer | 0.130523 | 0.135138 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | completed_h2d | 0.106814 | 0.130628 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | producer | 0.075370 | 0.096505 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.199972 | 0.212900 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | producer | 0.155761 | 0.168707 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_folded | completed_h2d | 0.392825 | 0.410222 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_folded | producer | 0.333286 | 0.344790 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_unfused | completed_h2d | 1.888176 | 1.967353 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_unfused | producer | 1.879860 | 2.090722 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_eager | completed_h2d | 0.260958 | 0.285691 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_eager | producer | 0.210412 | 0.231032 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.425154 | 0.445142 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_inductor | producer | 0.352562 | 0.378968 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_folded | completed_h2d | 0.223010 | 0.237350 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_folded | producer | 0.162965 | 0.169834 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_unfused | completed_h2d | 1.857896 | 1.866511 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_unfused | producer | 1.799732 | 1.821826 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_eager | completed_h2d | 0.261043 | 0.281605 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_eager | producer | 0.203087 | 0.232216 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.423594 | 0.446785 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_inductor | producer | 0.347158 | 0.373251 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_folded | completed_h2d | 0.206304 | 0.214118 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_folded | producer | 0.237235 | 0.245811 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.669701 | 0.690097 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_unfused | producer | 0.979149 | 1.022382 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_eager | completed_h2d | 0.446156 | 0.475861 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_eager | producer | 0.400608 | 0.447246 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.387254 | 0.415975 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_inductor | producer | 0.333274 | 0.372277 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_folded | completed_h2d | 0.164062 | 0.177304 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_folded | producer | 0.183855 | 0.192972 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.937519 | 1.002936 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_unfused | producer | 0.982859 | 1.302207 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_eager | completed_h2d | 0.447733 | 0.486339 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_eager | producer | 0.392864 | 0.444281 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.388215 | 0.412174 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_inductor | producer | 0.304734 | 0.328536 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | manual_fused | completed_h2d | 0.241575 | 0.253201 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | manual_fused | producer | 0.198899 | 0.207154 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_folded | completed_h2d | 0.311287 | 0.326352 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_folded | producer | 0.269493 | 0.278704 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_unfused | completed_h2d | 1.808118 | 1.878901 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_unfused | producer | 1.809866 | 1.858736 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_eager | completed_h2d | 0.227489 | 0.245296 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_eager | producer | 0.184857 | 0.207355 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.406168 | 0.419671 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_inductor | producer | 0.337804 | 0.362753 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | manual_fused | completed_h2d | 0.262360 | 0.273128 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | manual_fused | producer | 0.221052 | 0.230023 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_folded | completed_h2d | 0.189847 | 0.202432 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_folded | producer | 0.148764 | 0.159695 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_unfused | completed_h2d | 1.801610 | 1.813571 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_unfused | producer | 1.757281 | 1.772918 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_eager | completed_h2d | 0.226061 | 0.256545 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_eager | producer | 0.186105 | 0.203226 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.395730 | 0.415220 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_inductor | producer | 0.330275 | 0.356214 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | manual_fused | completed_h2d | 0.193742 | 0.210562 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | manual_fused | producer | 0.172939 | 0.180419 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_folded | completed_h2d | 0.209294 | 0.212569 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_folded | producer | 0.264730 | 0.346619 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.634167 | 0.947031 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_unfused | producer | 0.927115 | 1.128683 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_eager | completed_h2d | 0.389968 | 0.431820 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_eager | producer | 0.377610 | 0.392176 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.308823 | 0.335243 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_inductor | producer | 0.252179 | 0.295790 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | manual_fused | completed_h2d | 0.212502 | 0.220061 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | manual_fused | producer | 0.172739 | 0.178123 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_folded | completed_h2d | 0.160525 | 0.172980 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_folded | producer | 0.190373 | 0.195888 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.649940 | 0.661994 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_unfused | producer | 0.626170 | 1.064054 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_eager | completed_h2d | 0.414821 | 0.444035 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_eager | producer | 0.367946 | 0.401483 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.311704 | 0.327421 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_inductor | producer | 0.257180 | 0.285558 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | manual_fused | completed_h2d | 0.115490 | 0.123539 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | manual_fused | producer | 0.081881 | 0.086095 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | completed_h2d | 0.182212 | 0.192932 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | producer | 0.150172 | 0.177297 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.220235 | 0.244867 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | producer | 0.192822 | 0.222854 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | completed_h2d | 0.112624 | 0.121277 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | producer | 0.078905 | 0.084249 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.186496 | 0.199400 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | producer | 0.141986 | 0.152861 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | manual_fused | completed_h2d | 0.112861 | 0.122172 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | manual_fused | producer | 0.079386 | 0.084629 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | completed_h2d | 0.065180 | 0.068918 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | producer | 0.032747 | 0.034497 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.218080 | 0.245688 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | producer | 0.193655 | 0.224500 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | completed_h2d | 0.114403 | 0.125265 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | producer | 0.079496 | 0.087963 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.190764 | 0.206706 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | producer | 0.140969 | 0.159684 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | manual_fused | completed_h2d | 0.113897 | 0.125992 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | manual_fused | producer | 0.075459 | 0.092332 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | completed_h2d | 0.203520 | 0.222040 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | producer | 0.208746 | 0.225255 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.224703 | 0.251933 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | producer | 0.225635 | 0.227840 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | completed_h2d | 0.128069 | 0.149378 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | producer | 0.093919 | 0.113502 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.228384 | 0.246799 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | producer | 0.194198 | 0.210353 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | manual_fused | completed_h2d | 0.111333 | 0.137650 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | manual_fused | producer | 0.077487 | 0.100015 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | completed_h2d | 0.104575 | 0.112659 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | producer | 0.072208 | 0.080081 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.264614 | 0.321506 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | producer | 0.212055 | 0.256885 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | completed_h2d | 0.125735 | 0.147001 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | producer | 0.091386 | 0.118049 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.223864 | 0.234876 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | producer | 0.193576 | 0.210601 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | completed_h2d | 0.095747 | 0.105352 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | producer | 0.063321 | 0.071825 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | completed_h2d | 1.077065 | 1.124001 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | producer | 1.052724 | 1.091906 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | completed_h2d | 0.178856 | 0.189389 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | producer | 0.143118 | 0.150661 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.215566 | 0.226811 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | producer | 0.172399 | 0.181556 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | completed_h2d | 0.097065 | 0.104908 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | producer | 0.064252 | 0.066825 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.970380 | 1.003757 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | producer | 0.928836 | 0.952781 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | completed_h2d | 0.179603 | 0.187333 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | producer | 0.142337 | 0.152683 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.216328 | 0.222995 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | producer | 0.172169 | 0.181058 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | completed_h2d | 0.112409 | 0.124981 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | producer | 0.100812 | 0.111249 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.490010 | 0.504365 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | producer | 0.653655 | 0.681125 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | completed_h2d | 0.252567 | 0.284690 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | producer | 0.212806 | 0.249701 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.242252 | 0.266285 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | producer | 0.190824 | 0.211536 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | completed_h2d | 0.114540 | 0.125402 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | producer | 0.096290 | 0.108534 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.481422 | 0.503166 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | producer | 0.560124 | 0.660346 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | completed_h2d | 0.257321 | 0.306758 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | producer | 0.220586 | 0.265150 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.240616 | 0.255775 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | producer | 0.196931 | 0.212072 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_folded | completed_h2d | 0.041550 | 0.042842 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_folded | producer | 0.011231 | 0.012660 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_unfused | completed_h2d | 0.039751 | 0.042287 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_unfused | producer | 0.009724 | 0.009999 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_eager | completed_h2d | 0.062093 | 0.066179 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_eager | producer | 0.031631 | 0.033953 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_inductor | completed_h2d | 0.127643 | 0.135448 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_inductor | producer | 0.089797 | 0.097238 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_folded | completed_h2d | 0.036109 | 0.040138 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_folded | producer | 0.005822 | 0.007089 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_unfused | completed_h2d | 0.039906 | 0.040642 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_unfused | producer | 0.009663 | 0.009984 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_eager | completed_h2d | 0.062609 | 0.064633 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_eager | producer | 0.031976 | 0.034111 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_inductor | completed_h2d | 0.127533 | 0.135861 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_inductor | producer | 0.089666 | 0.096887 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_folded | completed_h2d | 0.042186 | 0.043663 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_folded | producer | 0.011517 | 0.012818 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_unfused | completed_h2d | 0.040963 | 0.046879 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_unfused | producer | 0.010450 | 0.010687 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_eager | completed_h2d | 0.062163 | 0.063466 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_eager | producer | 0.031745 | 0.033505 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_inductor | completed_h2d | 0.128312 | 0.136509 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_inductor | producer | 0.090057 | 0.098173 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_folded | completed_h2d | 0.036385 | 0.039238 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_folded | producer | 0.006042 | 0.007339 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_unfused | completed_h2d | 0.040477 | 0.041515 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_unfused | producer | 0.010195 | 0.010469 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_eager | completed_h2d | 0.062053 | 0.064280 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_eager | producer | 0.031731 | 0.034628 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_inductor | completed_h2d | 0.127177 | 0.135890 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_inductor | producer | 0.089571 | 0.096111 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_folded | completed_h2d | 1.784428 | 1.856822 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_folded | producer | 1.781588 | 1.845979 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_unfused | completed_h2d | 0.772070 | 0.829888 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_unfused | producer | 0.705834 | 0.768755 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_eager | completed_h2d | 0.273292 | 0.297463 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_eager | producer | 0.215535 | 0.227765 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_inductor | completed_h2d | 0.491803 | 0.505710 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_inductor | producer | 0.407747 | 0.426025 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_folded | completed_h2d | 0.178179 | 0.192009 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_folded | producer | 0.109704 | 0.118257 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_unfused | completed_h2d | 0.773219 | 0.824705 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_unfused | producer | 0.716617 | 0.769875 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_eager | completed_h2d | 0.269603 | 0.294959 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_eager | producer | 0.200111 | 0.222250 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_inductor | completed_h2d | 0.488982 | 0.502559 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_inductor | producer | 0.402591 | 0.426866 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_folded | completed_h2d | 0.623271 | 0.642139 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_folded | producer | 0.691547 | 0.701580 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_unfused | completed_h2d | 0.975973 | 1.022721 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_unfused | producer | 0.959211 | 1.160025 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_eager | completed_h2d | 0.335900 | 0.367745 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_eager | producer | 0.285645 | 0.300269 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_inductor | completed_h2d | 0.354351 | 0.374713 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_inductor | producer | 0.306360 | 0.346497 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_folded | completed_h2d | 0.153923 | 0.167077 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_folded | producer | 0.103188 | 0.135171 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_unfused | completed_h2d | 1.021249 | 1.160548 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_unfused | producer | 1.034825 | 1.145728 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_eager | completed_h2d | 0.339666 | 0.364867 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_eager | producer | 0.280270 | 0.308910 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_inductor | completed_h2d | 0.344676 | 0.365872 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_inductor | producer | 0.284771 | 0.314881 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | manual_fused | completed_h2d | 0.048477 | 0.057680 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | manual_fused | producer | 0.015254 | 0.017919 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_folded | completed_h2d | 0.119213 | 0.129343 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_folded | producer | 0.086344 | 0.094613 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.724506 | 0.775855 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_unfused | producer | 0.690151 | 0.777036 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_eager | completed_h2d | 0.501763 | 0.514350 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_eager | producer | 0.466770 | 0.477673 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.184437 | 0.209517 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_inductor | producer | 0.135268 | 0.150339 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | manual_fused | completed_h2d | 0.049063 | 0.060661 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | manual_fused | producer | 0.015630 | 0.020344 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_folded | completed_h2d | 0.072854 | 0.081610 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_folded | producer | 0.040592 | 0.044444 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.708729 | 0.797835 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_unfused | producer | 0.667080 | 0.824865 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_eager | completed_h2d | 0.503737 | 0.515745 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_eager | producer | 0.466295 | 0.478496 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.187363 | 0.220036 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_inductor | producer | 0.137367 | 0.155495 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | manual_fused | completed_h2d | 0.052050 | 0.059620 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | manual_fused | producer | 0.020965 | 0.031367 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_folded | completed_h2d | 0.119583 | 0.136230 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_folded | producer | 0.090007 | 0.109408 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.737772 | 0.863341 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_unfused | producer | 0.714621 | 0.830015 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_eager | completed_h2d | 0.522603 | 0.623816 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_eager | producer | 0.486467 | 0.572560 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.216703 | 0.239222 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_inductor | producer | 0.160886 | 0.186962 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | manual_fused | completed_h2d | 0.051068 | 0.062016 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | manual_fused | producer | 0.019533 | 0.029323 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_folded | completed_h2d | 0.093925 | 0.106549 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_folded | producer | 0.058332 | 0.067199 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.738081 | 0.948896 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_unfused | producer | 0.687174 | 0.790996 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_eager | completed_h2d | 0.516987 | 0.585268 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_eager | producer | 0.479621 | 0.568664 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.218215 | 0.239368 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_inductor | producer | 0.161778 | 0.185761 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.050110 | 0.052932 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.018811 | 0.019584 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.059669 | 0.063325 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.028745 | 0.031445 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.135152 | 0.146589 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.108411 | 0.115815 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.075489 | 0.081779 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.044319 | 0.048141 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.144271 | 0.153707 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.105166 | 0.113619 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.050487 | 0.057489 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.019552 | 0.022914 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.047260 | 0.051765 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.016762 | 0.017970 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.134156 | 0.143685 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.103543 | 0.109012 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.075645 | 0.084218 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.044224 | 0.047268 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.144967 | 0.154431 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.106092 | 0.114606 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.079377 | 0.090771 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.039249 | 0.047119 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.066166 | 0.072735 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.038849 | 0.041745 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.136304 | 0.154034 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.113712 | 0.120141 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.087036 | 0.094641 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.054244 | 0.059760 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.211524 | 0.220209 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.168095 | 0.178930 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.080523 | 0.087652 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.043167 | 0.047244 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.055521 | 0.062823 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.023395 | 0.026629 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.143073 | 0.156539 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.120831 | 0.138623 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.085328 | 0.088358 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.052305 | 0.055219 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.211217 | 0.220220 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.165370 | 0.179729 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.214340 | 0.226230 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.175376 | 0.184639 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.271364 | 0.290029 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_folded | producer | 0.226392 | 0.240641 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 1.592090 | 1.660504 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 1.626286 | 1.698669 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.209254 | 0.225694 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.163046 | 0.188716 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.339519 | 0.355275 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.283227 | 0.302831 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.239467 | 0.255173 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.207847 | 0.220552 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.169820 | 0.185278 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.127378 | 0.133870 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 1.583468 | 1.602033 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 1.542154 | 1.587590 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.201138 | 0.216203 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.161618 | 0.181472 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.326877 | 0.345225 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.268572 | 0.294623 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.165015 | 0.175641 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.145768 | 0.209540 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.238989 | 0.261155 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.270164 | 0.308985 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.547111 | 0.561941 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 0.699705 | 1.082339 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.310673 | 0.336112 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_eager | producer | 0.270295 | 0.354805 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.301250 | 0.323716 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.244186 | 0.309073 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.171919 | 0.180547 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.147035 | 0.153831 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.169959 | 0.187691 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.220946 | 0.233331 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.548332 | 0.569961 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 0.519216 | 0.909837 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.308408 | 0.346782 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_eager | producer | 0.278475 | 0.304301 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.298853 | 0.334445 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.247621 | 0.304335 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 4.512666 | 4.721580 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 4.290560 | 4.495731 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 4.585938 | 4.638690 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 4.363455 | 4.415199 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 25.514900 | 25.711650 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 25.443696 | 26.106910 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 3.271310 | 3.437714 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 3.070579 | 3.246068 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 2.331291 | 2.348364 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 2.100856 | 2.119085 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 3.653029 | 3.757853 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 3.441100 | 3.548230 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 3.147023 | 3.260427 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 2.957260 | 3.058447 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 25.517453 | 25.796659 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 25.243772 | 25.359664 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 3.278689 | 3.484781 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 3.068821 | 3.258152 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 2.331983 | 2.377425 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 2.103215 | 2.122853 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 1.616691 | 1.635905 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 1.382969 | 1.411967 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 1.810520 | 1.837605 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 1.480330 | 1.610256 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 6.102372 | 6.235872 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 5.890189 | 6.001126 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 2.871424 | 3.162885 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 2.643574 | 2.917726 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.933322 | 0.947945 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 0.704715 | 0.725174 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 1.727282 | 1.804476 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 1.503261 | 1.520167 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 1.329566 | 1.408949 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 1.073504 | 1.176794 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 6.125463 | 6.226166 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 5.886627 | 5.989239 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 2.906147 | 3.071724 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 2.710540 | 2.910034 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.940725 | 0.966426 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 0.705176 | 0.737362 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.059543 | 0.067079 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.027573 | 0.031962 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.084366 | 0.089222 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.052691 | 0.053966 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.091134 | 0.102041 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.065745 | 0.067591 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.089872 | 0.097783 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.056649 | 0.059973 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.150020 | 0.159703 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.110155 | 0.117310 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.058627 | 0.066134 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.026586 | 0.027563 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.047214 | 0.052988 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.015495 | 0.016863 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.092402 | 0.105984 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.063932 | 0.071327 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.089876 | 0.099780 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.056944 | 0.063888 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.149971 | 0.160121 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.110301 | 0.116691 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.085468 | 0.095314 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.045211 | 0.055350 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.090482 | 0.102309 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.057530 | 0.066911 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.101860 | 0.111855 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.070760 | 0.078152 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.096288 | 0.111046 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.066497 | 0.074083 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.198554 | 0.206869 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.152791 | 0.163987 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.081421 | 0.089016 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.044260 | 0.060487 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.055776 | 0.058052 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.021330 | 0.022951 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.099635 | 0.119758 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.080639 | 0.092053 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.103107 | 0.104611 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.066997 | 0.073513 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.198357 | 0.210030 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.151499 | 0.163768 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.353407 | 0.363955 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.293555 | 0.301009 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.580286 | 0.587949 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_folded | producer | 0.526444 | 0.534183 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.681912 | 0.774720 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 0.691673 | 0.721116 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.258767 | 0.278835 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.195873 | 0.219302 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.430578 | 0.448925 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.335477 | 0.362556 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.341753 | 0.353059 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.284393 | 0.293293 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.159399 | 0.168298 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.104710 | 0.113855 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.665115 | 0.765293 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 0.652786 | 0.729604 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.260167 | 0.286706 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.195226 | 0.222591 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.436098 | 0.456887 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.351168 | 0.376061 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.216268 | 0.240999 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.168921 | 0.176995 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.522775 | 0.545149 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.718741 | 0.771667 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.890840 | 1.023766 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 0.874424 | 0.992658 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.415303 | 0.441872 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_eager | producer | 0.370146 | 0.395845 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.330300 | 0.354525 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.288469 | 0.318699 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.219157 | 0.224817 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.170015 | 0.176965 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.171562 | 0.201287 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.179723 | 0.209815 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.953013 | 1.103989 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 1.005149 | 1.132083 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.398250 | 0.423501 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_eager | producer | 0.380005 | 0.394513 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.333070 | 0.354313 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.286407 | 0.304602 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 5.304312 | 5.391891 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 4.894527 | 4.949085 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 8.865973 | 8.923599 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 8.448455 | 8.516273 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 10.426673 | 12.021576 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 10.079137 | 11.821458 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 3.774186 | 4.056685 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 3.348372 | 3.653514 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 3.785596 | 3.863160 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 3.358314 | 3.452363 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 5.123905 | 5.245207 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 4.705214 | 4.808012 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 2.358027 | 2.567503 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 1.938821 | 2.145575 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 10.234993 | 11.863121 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 10.104173 | 11.584599 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 3.834966 | 4.126614 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 3.330229 | 3.697670 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 3.775919 | 3.902936 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 3.360946 | 3.466609 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 2.042929 | 2.131165 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 2.140485 | 2.534122 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 7.129487 | 7.455542 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 6.672702 | 7.104772 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 9.582646 | 10.074737 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 9.543411 | 11.009355 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 3.103406 | 3.376138 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 2.662380 | 2.878887 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 1.902515 | 2.031515 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 1.485981 | 1.608505 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 2.460894 | 2.588146 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 2.098528 | 2.604657 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 1.809249 | 2.404552 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 1.323825 | 1.749370 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 9.883610 | 10.531430 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 9.617166 | 11.712852 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 3.248694 | 3.517031 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 2.843267 | 3.084991 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 1.874270 | 2.049719 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 1.514884 | 1.636655 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.107371 | 0.116476 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.070856 | 0.077517 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.341778 | 0.350215 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.304955 | 0.314577 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.325997 | 0.361808 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.302453 | 0.317349 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.120025 | 0.127436 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.084875 | 0.091930 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.199742 | 0.223102 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.140759 | 0.162393 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.108185 | 0.117075 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.071207 | 0.078444 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.064828 | 0.071401 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.029942 | 0.034384 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.340967 | 0.358271 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.308620 | 0.343269 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.119925 | 0.133329 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.086550 | 0.100005 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.209683 | 0.227321 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.138170 | 0.170596 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.109018 | 0.121991 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.092977 | 0.102012 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.193870 | 0.279541 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.223407 | 0.308473 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.523424 | 0.538352 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.493267 | 0.504554 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.241961 | 0.276010 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.214770 | 0.268632 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.205192 | 0.233479 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.153994 | 0.173871 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.108763 | 0.123782 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.078951 | 0.111977 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.084811 | 0.099168 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.053347 | 0.058904 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.404440 | 0.437059 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.376769 | 0.397914 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.264093 | 0.294858 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.237156 | 0.274558 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.202997 | 0.217014 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.158828 | 0.176738 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 1.184438 | 1.189907 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.960324 | 0.963361 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 4.947404 | 4.989159 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_folded | producer | 4.722351 | 4.803989 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 5.018163 | 5.296977 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 4.745505 | 5.126262 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 1.247818 | 1.463962 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.974098 | 1.151860 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.841414 | 0.874381 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.613893 | 0.668700 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 1.178930 | 1.184515 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.955065 | 0.962246 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.624186 | 0.653358 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.401112 | 0.420246 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 4.855913 | 5.062083 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 4.649404 | 5.156404 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 1.194991 | 1.374358 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.977483 | 1.139648 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.838634 | 0.871046 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.616093 | 0.642528 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.534350 | 0.548130 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.382808 | 0.651335 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 1.166640 | 1.181113 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.941992 | 1.591938 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 5.035254 | 5.460459 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 4.862449 | 5.251118 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.853835 | 0.898931 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_eager | producer | 0.632027 | 0.705411 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.700268 | 0.717738 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.473536 | 0.499543 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.533591 | 0.549213 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.503945 | 0.567478 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.460967 | 0.473072 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.291647 | 0.324904 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 5.116298 | 5.445678 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 4.971758 | 5.470762 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.858133 | 1.038710 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_eager | producer | 0.628687 | 0.794465 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.692468 | 0.710371 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.466948 | 0.489199 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 17.457443 | 17.502711 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 14.791764 | 14.831063 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 79.064550 | 79.317538 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 76.415157 | 76.557438 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 104.392167 | 107.104645 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 102.492998 | 104.239309 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 91.877275 | 96.757973 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 89.120949 | 93.804335 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 9.289836 | 9.550736 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 6.610242 | 6.696930 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 17.461673 | 17.593349 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 14.806784 | 14.849107 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 9.088981 | 9.119090 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 6.425437 | 6.481421 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 102.269907 | 104.005944 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 99.693286 | 103.361440 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 91.898109 | 97.218766 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 89.082269 | 94.317621 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 9.266837 | 9.597927 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 6.603786 | 7.015602 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 6.700089 | 6.772665 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 4.042857 | 4.123959 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 16.745185 | 16.862606 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 14.090615 | 14.155904 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 21.803068 | 22.277345 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 19.180204 | 19.617880 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 28.461323 | 29.395813 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 26.140341 | 26.902635 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 5.870869 | 6.002163 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 3.191531 | 3.379487 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 6.684033 | 6.805408 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 4.047424 | 4.144970 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 5.589551 | 5.699390 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 2.922443 | 3.041670 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 21.777974 | 22.187102 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 19.132472 | 19.640827 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 29.060561 | 29.943671 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 26.321928 | 27.188130 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 5.855557 | 5.984037 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 3.192770 | 3.332830 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.041379 | 0.042397 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.009323 | 0.010194 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.040898 | 0.042059 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.008917 | 0.010460 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.062428 | 0.071233 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.035599 | 0.041217 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.083925 | 0.090067 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.051432 | 0.054016 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.131831 | 0.140268 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.092291 | 0.100239 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.041264 | 0.046069 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.009282 | 0.010149 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.037637 | 0.046295 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.006032 | 0.007387 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.061603 | 0.067554 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.032482 | 0.039033 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.084600 | 0.090466 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.052310 | 0.054703 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.133010 | 0.142586 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.094390 | 0.105017 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.046168 | 0.048954 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.010911 | 0.011756 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.045973 | 0.048461 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.011898 | 0.013655 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.073957 | 0.079743 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.038609 | 0.042747 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.097111 | 0.103375 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.061692 | 0.064735 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.187598 | 0.197909 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.143052 | 0.155072 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.045487 | 0.052119 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.012453 | 0.013552 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.041965 | 0.047065 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.009643 | 0.011580 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.072864 | 0.078813 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.040888 | 0.047049 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.099510 | 0.105495 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.062609 | 0.067916 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.187222 | 0.196383 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.143499 | 0.153766 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.085805 | 0.089537 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.038553 | 0.046610 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.113528 | 0.122659 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_folded | producer | 0.066102 | 0.072046 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.434663 | 0.503127 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 0.388070 | 0.441001 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.270945 | 0.284593 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.220343 | 0.238063 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.249839 | 0.265412 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.174524 | 0.215941 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.089191 | 0.096381 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.038438 | 0.043953 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.070028 | 0.085041 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.022038 | 0.030857 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.426445 | 0.507212 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 0.417552 | 0.497065 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.278288 | 0.298253 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.229820 | 0.260475 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.272230 | 0.302781 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.179138 | 0.229510 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.105822 | 0.131611 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.067238 | 0.096941 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.119793 | 0.143237 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.097075 | 0.161763 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.581908 | 0.695505 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 0.555295 | 0.817223 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.396566 | 0.446703 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_eager | producer | 0.362216 | 0.429573 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.274529 | 0.337518 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.209340 | 0.303004 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.097607 | 0.121845 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.082693 | 0.102164 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.102035 | 0.115458 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.070945 | 0.085563 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.591789 | 0.701500 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 0.657760 | 0.723244 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.407199 | 0.498565 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_eager | producer | 0.359901 | 0.453607 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.272931 | 0.299096 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.202576 | 0.283365 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.868429 | 1.060935 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 0.465032 | 0.668206 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 1.619608 | 1.692021 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 1.226533 | 1.287208 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 10.276037 | 10.860647 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 9.890165 | 10.376328 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 4.596434 | 4.833112 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 4.152216 | 4.458467 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 1.072638 | 1.296462 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 0.671361 | 0.892502 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.862304 | 1.076649 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 0.467152 | 0.660975 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.815694 | 1.056881 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 0.420714 | 0.650540 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 9.230450 | 9.613658 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 8.849693 | 9.062374 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 4.509378 | 4.754364 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 4.060014 | 4.427312 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 1.073003 | 1.296426 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 0.669049 | 0.894401 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.743765 | 0.768239 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 0.380518 | 0.428620 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.838205 | 0.872171 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 0.452067 | 0.596790 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 5.569661 | 5.835401 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 5.130292 | 7.022169 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 4.196254 | 4.396094 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 3.969101 | 4.138218 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.993005 | 1.030252 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 0.608712 | 0.654251 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.773292 | 0.818871 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 0.385725 | 0.430464 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.749365 | 0.806164 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 0.374720 | 0.409861 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 7.167447 | 7.315643 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 5.313583 | 5.506814 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 4.188557 | 4.534387 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 4.046900 | 4.319760 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.995720 | 1.084471 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 0.602340 | 0.670152 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.040933 | 0.046770 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.009372 | 0.010392 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.043408 | 0.050940 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.011928 | 0.013396 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.087267 | 0.097214 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.059955 | 0.073569 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.117235 | 0.127282 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.085064 | 0.091355 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.131682 | 0.142999 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.092652 | 0.101022 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.041173 | 0.049459 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.009393 | 0.009920 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.038138 | 0.040121 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.006848 | 0.008301 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.086815 | 0.096979 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.063551 | 0.069903 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.117585 | 0.127461 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.085399 | 0.094632 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.132472 | 0.140937 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.092963 | 0.101633 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.045823 | 0.051518 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.011086 | 0.011706 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.048387 | 0.056156 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.014753 | 0.016540 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.102316 | 0.121008 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.072514 | 0.081848 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.134262 | 0.142691 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.098969 | 0.108791 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.187422 | 0.196686 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.141060 | 0.153593 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.045692 | 0.047750 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.012569 | 0.013170 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.042771 | 0.045311 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.010265 | 0.011288 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.098904 | 0.110552 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.070474 | 0.081380 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.136364 | 0.148661 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.100647 | 0.106860 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.187602 | 0.195992 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.141459 | 0.152578 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.088520 | 0.095342 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.040468 | 0.044829 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.163449 | 0.173900 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_folded | producer | 0.115018 | 0.120148 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.835836 | 1.054049 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 0.776026 | 0.856213 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.453158 | 0.467834 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.405199 | 0.427736 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.272346 | 0.284469 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.208455 | 0.228755 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.087452 | 0.093463 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.040081 | 0.042468 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.080614 | 0.087715 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.034195 | 0.039410 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.864763 | 1.014314 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 0.841023 | 1.011628 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.459449 | 0.470862 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.404724 | 0.426410 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.269234 | 0.280296 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.201861 | 0.221437 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.113616 | 0.126206 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.068997 | 0.077736 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.134016 | 0.167024 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.133134 | 0.148287 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 1.256483 | 1.413844 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 1.237132 | 1.389677 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.765041 | 0.825418 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_eager | producer | 0.661547 | 0.710766 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.274128 | 0.317064 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.220571 | 0.257618 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.105746 | 0.114246 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.068075 | 0.086069 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.101258 | 0.107334 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.073766 | 0.080259 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 1.259399 | 1.441111 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 1.372275 | 1.501968 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.786992 | 0.851976 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_eager | producer | 0.767008 | 0.816801 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.274669 | 0.310528 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.211973 | 0.251095 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.868029 | 1.103678 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 0.482156 | 0.701837 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 2.378900 | 2.480821 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 1.985745 | 2.094674 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 19.504669 | 20.336491 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 19.164698 | 20.150611 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 7.858930 | 8.285595 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 7.690362 | 8.070064 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 1.147899 | 1.370371 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 0.751545 | 0.969987 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.877703 | 1.109523 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 0.476276 | 0.712998 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.894315 | 1.112035 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 0.503561 | 0.719992 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 20.531491 | 20.835778 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 20.187108 | 20.432046 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 8.131203 | 8.432994 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 7.675311 | 8.130684 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 1.154372 | 1.370140 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 0.752724 | 0.969004 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.750186 | 0.794929 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 0.383788 | 0.481850 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.902893 | 0.956167 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 0.521796 | 1.017465 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 11.025967 | 13.348119 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 10.785293 | 13.822243 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 7.898547 | 8.368563 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 7.681525 | 7.971375 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.986017 | 1.055141 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 0.607165 | 0.677955 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.745702 | 0.800298 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 0.388641 | 0.459015 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.743794 | 0.786580 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 0.391707 | 0.466746 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 10.528540 | 11.201657 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 10.192918 | 10.956739 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 8.022274 | 8.569278 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 7.633438 | 8.078127 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.977857 | 1.039901 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 0.599095 | 0.662998 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.041189 | 0.043424 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | manual_fused | producer | 0.009353 | 0.013750 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.048633 | 0.054946 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_folded | producer | 0.017273 | 0.018656 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 0.133576 | 0.160500 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_unfused | producer | 0.112481 | 0.122369 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.181754 | 0.190954 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_eager | producer | 0.149838 | 0.156152 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.134817 | 0.142960 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_inductor | producer | 0.094176 | 0.097954 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.040798 | 0.049052 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | manual_fused | producer | 0.009348 | 0.010320 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.040237 | 0.046921 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_folded | producer | 0.009123 | 0.010356 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 0.152154 | 0.164913 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_unfused | producer | 0.122964 | 0.136528 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.182509 | 0.192536 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_eager | producer | 0.149904 | 0.159839 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.134896 | 0.145041 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_inductor | producer | 0.095922 | 0.102677 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.042836 | 0.046454 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | manual_fused | producer | 0.010424 | 0.011317 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.053287 | 0.063051 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_folded | producer | 0.019662 | 0.021501 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 0.160892 | 0.181214 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_unfused | producer | 0.126712 | 0.144570 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 0.183871 | 0.209778 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_eager | producer | 0.170165 | 0.179043 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.183374 | 0.192484 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_inductor | producer | 0.139191 | 0.147079 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.045457 | 0.052734 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | manual_fused | producer | 0.012554 | 0.013101 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.045562 | 0.046915 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_folded | producer | 0.011907 | 0.013229 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 0.160020 | 0.177155 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_unfused | producer | 0.129447 | 0.143789 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 0.210115 | 0.222901 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_eager | producer | 0.173527 | 0.185369 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.188674 | 0.197746 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_inductor | producer | 0.142442 | 0.151922 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.091215 | 0.100858 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | manual_fused | producer | 0.043609 | 0.047719 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 0.247127 | 0.261427 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_folded | producer | 0.197053 | 0.203940 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 1.644579 | 1.833666 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_unfused | producer | 1.700688 | 1.896361 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 0.815750 | 0.846880 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_eager | producer | 0.767558 | 0.799039 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 0.321606 | 0.333840 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_inductor | producer | 0.260984 | 0.280182 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.090056 | 0.104487 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | manual_fused | producer | 0.043999 | 0.047716 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 0.117731 | 0.128534 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_folded | producer | 0.073125 | 0.078696 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 1.693427 | 1.911621 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_unfused | producer | 1.703837 | 1.926557 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 0.812202 | 0.848384 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_eager | producer | 0.777532 | 0.819889 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 0.320462 | 0.336271 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_inductor | producer | 0.266296 | 0.279408 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.104474 | 0.121760 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | manual_fused | producer | 0.067880 | 0.074841 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 0.152175 | 0.163889 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_folded | producer | 0.122694 | 0.171031 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 2.550187 | 2.888351 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_unfused | producer | 2.614219 | 2.850183 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 1.375296 | 1.476728 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_eager | producer | 1.200727 | 1.475459 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.299833 | 0.326346 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_inductor | producer | 0.226987 | 0.277883 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.107695 | 0.132837 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | manual_fused | producer | 0.069006 | 0.077748 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.121748 | 0.152422 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_folded | producer | 0.078570 | 0.082749 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 2.355712 | 2.589570 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_unfused | producer | 2.444381 | 2.638709 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 1.410846 | 1.563114 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_eager | producer | 1.308897 | 1.467225 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.311944 | 0.366332 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_inductor | producer | 0.233079 | 0.270394 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | manual_fused | completed_h2d | 0.872107 | 1.106235 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | manual_fused | producer | 0.481472 | 0.708196 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_folded | completed_h2d | 3.435257 | 3.464706 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_folded | producer | 3.041640 | 3.077838 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_unfused | completed_h2d | 41.974451 | 42.824240 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_unfused | producer | 41.538748 | 42.328379 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_eager | completed_h2d | 15.506811 | 15.904062 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_eager | producer | 15.024740 | 15.500908 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_inductor | completed_h2d | 1.577695 | 1.636626 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_inductor | producer | 1.169599 | 1.221400 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | manual_fused | completed_h2d | 0.873680 | 1.109298 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | manual_fused | producer | 0.481530 | 0.707360 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_folded | completed_h2d | 1.418932 | 1.449830 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_folded | producer | 1.026227 | 1.072303 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_unfused | completed_h2d | 41.520380 | 41.874624 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_unfused | producer | 40.448833 | 41.466719 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_eager | completed_h2d | 15.371063 | 15.790170 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_eager | producer | 15.176277 | 15.482685 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_inductor | completed_h2d | 1.578832 | 1.614588 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_inductor | producer | 1.170348 | 1.219217 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | manual_fused | completed_h2d | 0.744482 | 0.828990 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | manual_fused | producer | 0.357722 | 0.434334 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_folded | completed_h2d | 1.142238 | 1.172181 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_folded | producer | 0.758258 | 1.380761 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_unfused | completed_h2d | 21.112118 | 21.952139 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_unfused | producer | 20.750514 | 21.547437 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_eager | completed_h2d | 15.253747 | 15.887481 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_eager | producer | 15.015764 | 15.565180 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_inductor | completed_h2d | 0.986577 | 1.064264 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_inductor | producer | 0.604279 | 0.692915 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | manual_fused | completed_h2d | 0.749496 | 0.840264 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | manual_fused | producer | 0.362201 | 0.450408 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_folded | completed_h2d | 0.748343 | 0.804464 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_folded | producer | 0.403274 | 0.541009 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_unfused | completed_h2d | 20.416214 | 20.683770 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_unfused | producer | 20.297378 | 20.865930 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_eager | completed_h2d | 15.304744 | 15.770928 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_eager | producer | 15.048877 | 15.565500 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_inductor | completed_h2d | 0.982976 | 1.045146 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_inductor | producer | 0.599684 | 0.663190 |

## Setup, first completed call and amortized totals

Total(N) = shared allocation + arm-specific compile/bind/setup/storage + first completed call + (N−1)×warm median. Warmup runs used to estimate steady state are reported separately in raw data and excluded from this derived total. Medians are computed from complete per-process sums. All times are ms.

| Suite | Case | Workers | Runtime | Arm | Preparation | First completed | Total(1) | Total(100) | Total(1000) |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | manual_fused | 206.599 | 2.515 | 214.576 | 415.108 | 2238.126 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | sym_folded | 211.168 | 2.646 | 213.814 | 461.382 | 2712.000 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | sym_unfused | 221.359 | 15.857 | 237.216 | 1504.841 | 12986.072 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | torch_eager | 206.597 | 2.853 | 214.912 | 482.612 | 2016.329 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | baseline | torch_inductor | 212.498 | 7094.300 | 7309.706 | 7447.412 | 8699.293 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | manual_fused | 206.610 | 2.768 | 209.173 | 456.090 | 2526.303 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | sym_folded | 210.859 | 1.677 | 212.949 | 377.073 | 1869.107 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | sym_unfused | 220.559 | 15.565 | 236.124 | 1494.851 | 12939.186 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | torch_eager | 206.608 | 1.842 | 208.704 | 385.283 | 1990.548 |
| consumer | dequant_transpose_n1024_m512_o8 | 1 | candidate | torch_inductor | 212.565 | 7091.897 | 7304.868 | 7444.538 | 8695.048 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | manual_fused | 185.413 | 1.963 | 191.363 | 304.420 | 1262.682 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | sym_folded | 189.736 | 1.509 | 190.997 | 262.887 | 916.434 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | sym_unfused | 199.375 | 7.967 | 207.342 | 527.071 | 3433.698 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | torch_eager | 185.133 | 2.975 | 192.579 | 381.218 | 1893.680 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | baseline | torch_inductor | 190.607 | 7109.229 | 7298.062 | 7367.702 | 8000.798 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | manual_fused | 183.955 | 1.355 | 188.148 | 278.026 | 1123.129 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | sym_folded | 188.323 | 1.286 | 189.609 | 267.028 | 970.834 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | sym_unfused | 198.182 | 7.346 | 205.302 | 523.287 | 3404.021 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | torch_eager | 183.684 | 3.233 | 189.758 | 390.007 | 2249.630 |
| consumer | dequant_transpose_n1024_m512_o8 | 8 | candidate | torch_inductor | 189.161 | 7118.813 | 7307.796 | 7376.144 | 7997.484 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | manual_fused | 2.522 | 1.506 | 4.132 | 143.284 | 1410.409 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | sym_folded | 7.078 | 2.371 | 9.479 | 235.677 | 2284.001 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | sym_unfused | 24.765 | 2.778 | 27.539 | 294.355 | 2720.019 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | torch_eager | 2.521 | 1.130 | 3.644 | 104.412 | 1004.272 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | baseline | torch_inductor | 3.297 | 1450.608 | 1453.906 | 1580.733 | 2732.827 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | manual_fused | 2.534 | 1.482 | 4.016 | 139.433 | 1364.436 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | sym_folded | 7.003 | 0.776 | 7.779 | 75.789 | 695.017 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | sym_unfused | 24.525 | 2.700 | 27.219 | 288.112 | 2663.932 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | torch_eager | 2.533 | 1.083 | 3.624 | 103.725 | 1013.732 |
| consumer | padded_roundtrip_n1024_m256_o8 | 1 | candidate | torch_inductor | 3.252 | 1444.235 | 1447.433 | 1575.029 | 2739.047 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | manual_fused | 3.024 | 0.862 | 3.854 | 81.746 | 789.854 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | sym_folded | 7.521 | 2.308 | 10.831 | 201.686 | 1936.725 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | sym_unfused | 24.802 | 2.973 | 28.043 | 299.214 | 2754.185 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | torch_eager | 2.728 | 1.184 | 3.912 | 113.425 | 1099.009 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | baseline | torch_inductor | 3.375 | 1459.981 | 1463.392 | 1540.576 | 2246.663 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | manual_fused | 2.925 | 0.830 | 3.822 | 85.694 | 796.096 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | sym_folded | 7.393 | 0.833 | 8.258 | 67.803 | 604.878 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | sym_unfused | 26.090 | 2.851 | 28.923 | 311.171 | 2877.065 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | torch_eager | 2.642 | 1.243 | 3.885 | 112.130 | 1097.728 |
| consumer | padded_roundtrip_n1024_m256_o8 | 8 | candidate | torch_inductor | 3.372 | 1456.378 | 1459.718 | 1534.943 | 2229.545 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | 4.273 | 0.191 | 4.474 | 24.164 | 202.510 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | 13.316 | 0.189 | 13.504 | 28.934 | 169.200 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | 0.065 | 0.120 | 0.196 | 9.207 | 91.119 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | 0.525 | 1014.571 | 1015.115 | 1029.890 | 1164.200 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | 4.275 | 0.180 | 4.459 | 21.654 | 177.976 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | 13.227 | 0.174 | 13.402 | 27.771 | 158.395 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | 0.063 | 0.119 | 0.184 | 9.260 | 91.828 |
| heldout | bf16_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | 0.530 | 1017.163 | 1017.701 | 1032.538 | 1167.421 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | 4.607 | 0.172 | 4.777 | 19.658 | 155.119 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | 13.535 | 0.202 | 13.738 | 29.713 | 174.204 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | 0.069 | 0.138 | 0.206 | 10.771 | 106.767 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | 0.532 | 1018.338 | 1018.885 | 1039.093 | 1220.312 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | 4.814 | 0.169 | 4.979 | 19.484 | 150.496 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | 14.122 | 0.189 | 14.299 | 29.492 | 165.649 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | 0.066 | 0.135 | 0.201 | 10.776 | 106.908 |
| heldout | bf16_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | 0.556 | 1017.910 | 1018.466 | 1037.783 | 1220.483 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_folded | 4.475 | 0.396 | 4.947 | 43.837 | 397.469 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | sym_unfused | 17.659 | 1.994 | 19.664 | 206.593 | 1905.951 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_eager | 0.064 | 0.298 | 0.362 | 26.313 | 261.175 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | baseline | torch_inductor | 0.539 | 1010.470 | 1011.009 | 1053.099 | 1435.738 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_folded | 4.433 | 0.232 | 4.733 | 26.799 | 227.445 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | sym_unfused | 17.585 | 1.892 | 19.477 | 203.618 | 1875.725 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_eager | 0.067 | 0.321 | 0.388 | 26.233 | 261.171 |
| heldout | dequant_roundtrip_n769_m97_o8 | 1 | candidate | torch_inductor | 0.536 | 1013.143 | 1013.679 | 1055.244 | 1438.971 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_folded | 4.698 | 0.223 | 4.914 | 25.305 | 210.978 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | sym_unfused | 18.283 | 1.096 | 19.379 | 85.566 | 688.297 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_eager | 0.068 | 0.510 | 0.580 | 44.843 | 446.481 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | baseline | torch_inductor | 0.866 | 1041.911 | 1042.859 | 1081.470 | 1430.830 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_folded | 4.833 | 0.175 | 5.017 | 21.307 | 168.963 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | sym_unfused | 18.653 | 1.348 | 19.963 | 113.157 | 956.924 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_eager | 0.069 | 0.486 | 0.553 | 44.878 | 447.838 |
| heldout | dequant_roundtrip_n769_m97_o8 | 8 | candidate | torch_inductor | 0.692 | 1032.804 | 1033.355 | 1071.788 | 1421.181 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | manual_fused | 0.202 | 0.260 | 0.503 | 24.419 | 241.836 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_folded | 4.557 | 0.333 | 4.925 | 35.788 | 315.947 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | sym_unfused | 14.187 | 1.842 | 16.017 | 194.903 | 1821.996 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_eager | 0.201 | 0.259 | 0.456 | 22.978 | 227.718 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | baseline | torch_inductor | 0.922 | 1185.071 | 1185.952 | 1223.514 | 1589.065 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | manual_fused | 0.196 | 0.275 | 0.471 | 26.444 | 262.569 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_folded | 4.628 | 0.194 | 4.880 | 23.678 | 194.584 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | sym_unfused | 13.848 | 1.846 | 15.672 | 194.649 | 1816.741 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_eager | 0.193 | 0.250 | 0.444 | 22.825 | 226.279 |
| heldout | dequant_transpose_n769_m97_o8 | 1 | candidate | torch_inductor | 0.878 | 1184.953 | 1185.891 | 1225.482 | 1581.225 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | manual_fused | 0.373 | 0.256 | 0.632 | 19.733 | 194.100 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_folded | 4.727 | 0.310 | 5.052 | 25.862 | 214.227 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | sym_unfused | 14.090 | 1.565 | 15.654 | 79.076 | 649.827 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_eager | 0.101 | 0.462 | 0.563 | 39.127 | 390.099 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | baseline | torch_inductor | 0.602 | 1187.089 | 1187.698 | 1218.271 | 1496.212 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | manual_fused | 0.373 | 0.275 | 0.639 | 21.677 | 212.929 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_folded | 4.765 | 0.247 | 5.027 | 20.919 | 165.392 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | sym_unfused | 14.243 | 1.440 | 15.682 | 79.888 | 665.921 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_eager | 0.102 | 0.464 | 0.566 | 41.634 | 414.973 |
| heldout | dequant_transpose_n769_m97_o8 | 8 | candidate | torch_inductor | 0.598 | 1189.369 | 1189.967 | 1220.618 | 1499.918 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | manual_fused | 0.065 | 0.136 | 0.201 | 11.634 | 115.575 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | 4.513 | 0.203 | 4.759 | 22.820 | 186.810 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | 21.344 | 0.264 | 21.603 | 43.705 | 241.878 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | 0.064 | 0.143 | 0.210 | 11.399 | 112.760 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | 0.541 | 1434.668 | 1435.279 | 1453.746 | 1621.627 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | manual_fused | 0.069 | 0.133 | 0.210 | 11.481 | 113.055 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | 4.543 | 0.073 | 4.670 | 11.123 | 69.784 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | 21.291 | 0.248 | 21.598 | 43.343 | 239.459 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | 0.067 | 0.149 | 0.216 | 11.586 | 114.548 |
| heldout | padded_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | 0.582 | 1435.329 | 1435.911 | 1454.796 | 1626.484 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | manual_fused | 0.351 | 0.153 | 0.504 | 11.899 | 114.406 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | 4.804 | 0.232 | 5.032 | 25.370 | 208.547 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | 22.671 | 0.271 | 22.934 | 44.951 | 247.178 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | 0.081 | 0.163 | 0.248 | 12.900 | 128.163 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | 0.648 | 1451.856 | 1452.499 | 1475.151 | 1680.509 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | manual_fused | 0.461 | 0.132 | 0.645 | 11.769 | 111.968 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | 4.808 | 0.129 | 5.035 | 15.388 | 109.505 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | 23.187 | 0.264 | 23.468 | 49.884 | 288.874 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | 0.079 | 0.200 | 0.275 | 12.723 | 125.884 |
| heldout | padded_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | 0.652 | 1461.063 | 1461.715 | 1484.410 | 1689.051 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_folded | 4.424 | 0.119 | 4.543 | 14.022 | 100.194 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | sym_unfused | 13.556 | 1.061 | 14.741 | 121.371 | 1090.729 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_eager | 0.070 | 0.249 | 0.319 | 18.022 | 178.943 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | baseline | torch_inductor | 0.536 | 1088.112 | 1088.648 | 1109.979 | 1304.275 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_folded | 4.467 | 0.103 | 4.633 | 14.249 | 101.628 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | sym_unfused | 13.553 | 0.977 | 14.529 | 110.587 | 983.929 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_eager | 0.066 | 0.248 | 0.314 | 18.197 | 179.690 |
| heldout | quant_roundtrip_n257_m67_o8 | 1 | candidate | torch_inductor | 0.533 | 1089.458 | 1089.991 | 1111.768 | 1309.290 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_folded | 4.949 | 0.124 | 5.073 | 16.202 | 117.371 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | sym_unfused | 14.724 | 0.710 | 15.426 | 63.937 | 504.946 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_eager | 0.071 | 0.300 | 0.371 | 25.384 | 252.694 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | baseline | torch_inductor | 0.668 | 1099.556 | 1100.223 | 1124.679 | 1346.130 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_folded | 4.811 | 0.127 | 5.017 | 16.358 | 119.444 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | sym_unfused | 15.638 | 0.744 | 16.560 | 62.766 | 497.038 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_eager | 0.070 | 0.314 | 0.384 | 25.895 | 257.484 |
| heldout | quant_roundtrip_n257_m67_o8 | 8 | candidate | torch_inductor | 0.584 | 1100.884 | 1101.422 | 1125.208 | 1341.445 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_folded | 4.450 | 0.050 | 4.500 | 8.614 | 46.008 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | sym_unfused | 17.721 | 0.068 | 17.788 | 21.744 | 57.526 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_eager | 0.114 | 0.087 | 0.201 | 6.347 | 62.240 |
| heldout | rank3_chain_n1_m67_o3 | 1 | baseline | torch_inductor | 0.872 | 945.979 | 946.851 | 959.442 | 1073.901 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_folded | 4.455 | 0.044 | 4.566 | 8.141 | 40.639 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | sym_unfused | 17.703 | 0.062 | 17.765 | 21.690 | 57.603 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_eager | 0.114 | 0.087 | 0.201 | 6.399 | 62.747 |
| heldout | rank3_chain_n1_m67_o3 | 1 | candidate | torch_inductor | 0.892 | 948.563 | 949.585 | 962.186 | 1076.740 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_folded | 4.803 | 0.051 | 4.961 | 9.138 | 47.105 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | sym_unfused | 19.845 | 0.069 | 19.922 | 23.982 | 60.570 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_eager | 0.113 | 0.086 | 0.201 | 6.355 | 62.303 |
| heldout | rank3_chain_n1_m67_o3 | 8 | baseline | torch_inductor | 0.590 | 941.768 | 942.361 | 954.869 | 1068.585 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_folded | 4.804 | 0.044 | 4.847 | 8.438 | 41.178 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | sym_unfused | 19.755 | 0.065 | 19.819 | 23.832 | 60.316 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_eager | 0.108 | 0.088 | 0.205 | 6.349 | 62.197 |
| heldout | rank3_chain_n1_m67_o3 | 8 | candidate | torch_inductor | 0.589 | 939.206 | 939.795 | 952.385 | 1066.845 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_folded | 4.503 | 1.773 | 6.282 | 182.940 | 1788.925 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | sym_unfused | 17.930 | 0.809 | 18.739 | 95.372 | 790.037 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_eager | 0.096 | 0.278 | 0.376 | 27.441 | 273.376 |
| heldout | rank3_chain_n257_m67_o5 | 1 | baseline | torch_inductor | 0.560 | 959.586 | 960.146 | 1009.111 | 1451.236 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_folded | 4.435 | 0.202 | 4.668 | 22.263 | 182.707 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | sym_unfused | 17.670 | 0.826 | 18.496 | 95.045 | 790.942 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_eager | 0.096 | 0.297 | 0.395 | 27.085 | 269.728 |
| heldout | rank3_chain_n257_m67_o5 | 1 | candidate | torch_inductor | 0.561 | 956.098 | 956.657 | 1005.473 | 1444.302 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_folded | 4.830 | 1.128 | 5.938 | 67.635 | 628.579 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | sym_unfused | 19.530 | 0.942 | 20.472 | 117.093 | 995.469 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_eager | 0.194 | 0.454 | 0.648 | 33.830 | 336.140 |
| heldout | rank3_chain_n257_m67_o5 | 8 | baseline | torch_inductor | 0.789 | 963.199 | 964.073 | 999.373 | 1318.288 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_folded | 4.797 | 0.245 | 5.059 | 20.520 | 158.784 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | sym_unfused | 18.447 | 0.976 | 19.423 | 120.527 | 1039.650 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_eager | 0.189 | 0.425 | 0.612 | 34.233 | 339.933 |
| heldout | rank3_chain_n257_m67_o5 | 8 | candidate | torch_inductor | 0.775 | 968.256 | 969.147 | 1002.950 | 1315.709 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | manual_fused | 205.985 | 0.075 | 206.054 | 210.853 | 254.482 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_folded | 211.152 | 0.126 | 211.276 | 222.962 | 329.198 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | sym_unfused | 322.092 | 1.549 | 323.914 | 396.405 | 1052.670 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_eager | 205.983 | 0.525 | 206.608 | 256.732 | 707.829 |
| heldout | roundtrip7_n257_m67_o8 | 1 | baseline | torch_inductor | 211.903 | 7000.034 | 7212.419 | 7230.752 | 7397.407 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | manual_fused | 206.036 | 0.073 | 206.409 | 211.230 | 255.744 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_folded | 211.231 | 0.080 | 211.310 | 218.515 | 284.016 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | sym_unfused | 322.150 | 1.526 | 324.009 | 394.555 | 1032.029 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_eager | 206.034 | 0.517 | 206.548 | 256.418 | 709.781 |
| heldout | roundtrip7_n257_m67_o8 | 1 | candidate | torch_inductor | 211.967 | 7008.935 | 7218.758 | 7237.567 | 7408.555 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | manual_fused | 191.688 | 0.079 | 191.765 | 196.795 | 242.521 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_folded | 196.743 | 0.178 | 196.921 | 209.524 | 317.374 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | sym_unfused | 307.449 | 1.712 | 309.404 | 383.427 | 1046.877 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_eager | 191.422 | 0.550 | 192.130 | 243.525 | 710.754 |
| heldout | roundtrip7_n257_m67_o8 | 8 | baseline | torch_inductor | 196.858 | 7052.047 | 7248.905 | 7270.501 | 7466.832 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | manual_fused | 184.003 | 0.082 | 184.450 | 189.462 | 235.097 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_folded | 189.196 | 0.146 | 189.676 | 198.975 | 283.647 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | sym_unfused | 302.824 | 1.660 | 304.705 | 377.775 | 1042.048 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_eager | 183.734 | 0.635 | 184.369 | 235.390 | 700.678 |
| heldout | roundtrip7_n257_m67_o8 | 8 | candidate | torch_inductor | 189.205 | 7038.968 | 7231.589 | 7253.192 | 7449.586 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | manual_fused | 0.178 | 0.071 | 0.312 | 5.274 | 50.309 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_folded | 4.517 | 0.075 | 4.592 | 10.504 | 64.255 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | sym_unfused | 14.115 | 0.162 | 14.277 | 27.827 | 149.752 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_eager | 0.177 | 0.101 | 0.341 | 7.794 | 75.669 |
| main | dequant_transpose_n33_m128_o8 | 1 | baseline | torch_inductor | 1.092 | 1045.301 | 1046.270 | 1060.553 | 1190.397 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | manual_fused | 0.176 | 0.070 | 0.246 | 5.240 | 50.671 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_folded | 4.571 | 0.053 | 4.624 | 9.324 | 52.053 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | sym_unfused | 14.127 | 0.169 | 14.291 | 27.570 | 148.230 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_eager | 0.174 | 0.086 | 0.265 | 7.754 | 75.835 |
| main | dequant_transpose_n33_m128_o8 | 1 | candidate | torch_inductor | 1.079 | 1038.686 | 1039.739 | 1054.091 | 1184.561 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | manual_fused | 0.499 | 0.097 | 0.609 | 8.753 | 80.193 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_folded | 4.891 | 0.076 | 4.964 | 11.518 | 71.104 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | sym_unfused | 14.510 | 0.200 | 14.699 | 28.430 | 151.104 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_eager | 0.242 | 0.115 | 0.357 | 8.994 | 87.443 |
| main | dequant_transpose_n33_m128_o8 | 8 | baseline | torch_inductor | 1.289 | 1044.230 | 1045.518 | 1066.586 | 1258.112 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | manual_fused | 0.487 | 0.102 | 0.622 | 8.782 | 81.341 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_folded | 4.781 | 0.065 | 5.005 | 10.480 | 60.488 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | sym_unfused | 14.311 | 0.200 | 14.498 | 28.662 | 157.428 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_eager | 0.230 | 0.107 | 0.337 | 8.894 | 85.485 |
| main | dequant_transpose_n33_m128_o8 | 8 | candidate | torch_inductor | 1.132 | 1035.694 | 1036.826 | 1057.606 | 1246.511 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | manual_fused | 0.106 | 0.223 | 0.333 | 21.553 | 214.459 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_folded | 4.431 | 0.277 | 4.722 | 31.716 | 275.944 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | sym_unfused | 13.461 | 1.633 | 15.095 | 172.737 | 1605.617 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_eager | 0.104 | 0.358 | 0.456 | 21.072 | 209.401 |
| main | dequant_transpose_n512_m128_o8 | 1 | baseline | torch_inductor | 0.576 | 937.363 | 937.939 | 971.552 | 1277.119 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | manual_fused | 0.098 | 0.243 | 0.341 | 24.045 | 239.565 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_folded | 4.429 | 0.173 | 4.602 | 21.415 | 174.252 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | sym_unfused | 13.461 | 1.663 | 15.129 | 171.922 | 1597.319 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_eager | 0.096 | 0.235 | 0.332 | 20.284 | 201.309 |
| main | dequant_transpose_n512_m128_o8 | 1 | candidate | torch_inductor | 0.555 | 940.453 | 941.001 | 973.331 | 1267.238 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | manual_fused | 0.383 | 0.175 | 0.573 | 16.909 | 165.422 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_folded | 4.734 | 0.257 | 5.037 | 28.575 | 243.666 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | sym_unfused | 14.174 | 1.231 | 15.405 | 69.802 | 561.644 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_eager | 0.112 | 0.537 | 0.649 | 31.335 | 310.940 |
| main | dequant_transpose_n512_m128_o8 | 8 | baseline | torch_inductor | 0.587 | 943.890 | 944.478 | 974.772 | 1245.427 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | manual_fused | 0.371 | 0.199 | 0.570 | 17.644 | 172.370 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_folded | 4.707 | 0.191 | 4.902 | 21.792 | 174.755 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | sym_unfused | 13.705 | 1.099 | 14.804 | 68.990 | 562.464 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_eager | 0.113 | 0.473 | 0.588 | 31.197 | 308.763 |
| main | dequant_transpose_n512_m128_o8 | 8 | candidate | torch_inductor | 0.586 | 942.807 | 943.389 | 971.759 | 1240.727 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | manual_fused | 0.077 | 4.799 | 4.877 | 451.631 | 4513.031 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_folded | 4.426 | 4.593 | 9.019 | 463.181 | 4589.847 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | sym_unfused | 13.674 | 25.758 | 40.026 | 2568.872 | 25532.282 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_eager | 0.075 | 3.674 | 3.750 | 327.610 | 3271.789 |
| main | dequant_transpose_n8192_m128_o8 | 1 | baseline | torch_inductor | 0.552 | 940.787 | 941.365 | 1171.159 | 3272.578 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | manual_fused | 0.079 | 3.691 | 3.777 | 365.427 | 3653.153 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_folded | 4.462 | 3.174 | 7.636 | 319.191 | 3151.511 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | sym_unfused | 13.520 | 26.326 | 39.788 | 2566.016 | 25531.723 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_eager | 0.077 | 3.344 | 3.422 | 328.012 | 3278.831 |
| main | dequant_transpose_n8192_m128_o8 | 1 | candidate | torch_inductor | 0.540 | 940.566 | 941.106 | 1171.972 | 3270.756 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | manual_fused | 0.361 | 1.705 | 2.046 | 162.099 | 1617.120 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_folded | 4.696 | 1.803 | 6.485 | 185.893 | 1815.361 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | sym_unfused | 13.745 | 10.900 | 24.777 | 627.685 | 6121.194 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_eager | 0.090 | 3.010 | 3.100 | 288.971 | 2873.253 |
| main | dequant_transpose_n8192_m128_o8 | 8 | baseline | torch_inductor | 0.560 | 936.849 | 937.413 | 1029.811 | 1869.801 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | manual_fused | 0.358 | 1.769 | 2.127 | 173.128 | 1727.682 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_folded | 4.676 | 1.547 | 6.223 | 137.701 | 1334.311 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | sym_unfused | 13.658 | 10.691 | 24.648 | 630.244 | 6146.491 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_eager | 0.082 | 3.168 | 3.249 | 291.109 | 2906.650 |
| main | dequant_transpose_n8192_m128_o8 | 8 | candidate | torch_inductor | 0.551 | 944.504 | 945.055 | 1038.187 | 1884.832 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | manual_fused | 0.093 | 0.080 | 0.174 | 6.068 | 59.737 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_folded | 4.554 | 0.094 | 4.837 | 13.189 | 89.275 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | sym_unfused | 22.677 | 0.130 | 22.810 | 32.019 | 114.385 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_eager | 0.091 | 0.115 | 0.195 | 9.161 | 89.977 |
| main | padded_roundtrip_n33_m128_o8 | 1 | baseline | torch_inductor | 0.560 | 1444.698 | 1445.288 | 1460.140 | 1595.157 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | manual_fused | 0.076 | 0.080 | 0.156 | 5.985 | 58.917 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_folded | 4.523 | 0.054 | 4.734 | 9.380 | 52.015 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | sym_unfused | 22.020 | 0.116 | 22.135 | 31.283 | 114.445 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_eager | 0.074 | 0.107 | 0.198 | 9.111 | 90.155 |
| main | padded_roundtrip_n33_m128_o8 | 1 | candidate | torch_inductor | 0.593 | 1422.248 | 1422.840 | 1437.671 | 1572.500 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | manual_fused | 0.374 | 0.120 | 0.608 | 9.193 | 86.114 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_folded | 4.839 | 0.101 | 4.943 | 14.035 | 95.469 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | sym_unfused | 22.153 | 0.149 | 22.303 | 32.387 | 124.061 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_eager | 0.083 | 0.126 | 0.210 | 9.903 | 96.562 |
| main | padded_roundtrip_n33_m128_o8 | 8 | baseline | torch_inductor | 0.564 | 1427.932 | 1428.477 | 1448.133 | 1626.832 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | manual_fused | 0.357 | 0.106 | 0.463 | 8.803 | 81.892 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_folded | 4.826 | 0.064 | 4.960 | 10.482 | 60.680 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | sym_unfused | 22.334 | 0.136 | 22.462 | 32.326 | 121.997 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_eager | 0.085 | 0.127 | 0.218 | 10.426 | 103.222 |
| main | padded_roundtrip_n33_m128_o8 | 8 | candidate | torch_inductor | 0.584 | 1438.000 | 1438.548 | 1458.185 | 1636.706 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | manual_fused | 0.098 | 0.358 | 0.458 | 35.445 | 353.511 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_folded | 4.477 | 0.585 | 5.088 | 62.536 | 584.794 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | sym_unfused | 21.527 | 0.706 | 22.242 | 89.650 | 703.371 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_eager | 0.097 | 0.290 | 0.387 | 26.005 | 258.895 |
| main | padded_roundtrip_n512_m128_o8 | 1 | baseline | torch_inductor | 0.560 | 1326.396 | 1326.962 | 1370.214 | 1763.414 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | manual_fused | 0.109 | 0.370 | 0.479 | 34.350 | 341.928 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_folded | 4.553 | 0.162 | 4.763 | 20.607 | 164.144 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | sym_unfused | 21.597 | 0.683 | 22.280 | 88.145 | 686.749 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_eager | 0.107 | 0.300 | 0.407 | 26.169 | 260.319 |
| main | padded_roundtrip_n512_m128_o8 | 1 | candidate | torch_inductor | 0.585 | 1301.536 | 1302.101 | 1344.775 | 1737.135 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | manual_fused | 0.373 | 0.221 | 0.658 | 22.069 | 216.710 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_folded | 4.901 | 0.510 | 5.399 | 57.026 | 527.523 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | sym_unfused | 22.372 | 0.786 | 23.158 | 111.351 | 913.106 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_eager | 0.102 | 0.461 | 0.570 | 41.678 | 415.451 |
| main | padded_roundtrip_n512_m128_o8 | 8 | baseline | torch_inductor | 0.581 | 1309.841 | 1310.412 | 1342.259 | 1644.641 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | manual_fused | 0.384 | 0.224 | 0.665 | 22.368 | 219.610 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_folded | 4.950 | 0.162 | 5.111 | 21.909 | 176.314 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | sym_unfused | 22.193 | 0.980 | 23.187 | 117.720 | 975.431 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_eager | 0.104 | 0.467 | 0.574 | 40.000 | 398.425 |
| main | padded_roundtrip_n512_m128_o8 | 8 | candidate | torch_inductor | 0.577 | 1315.348 | 1315.921 | 1350.438 | 1642.616 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | manual_fused | 8.421 | 5.277 | 13.697 | 538.824 | 5312.704 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_folded | 12.795 | 8.911 | 21.706 | 899.780 | 8879.156 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | sym_unfused | 30.173 | 10.442 | 40.615 | 1072.855 | 10456.862 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_eager | 8.419 | 3.617 | 12.151 | 385.677 | 3782.445 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | baseline | torch_inductor | 8.901 | 1327.712 | 1336.613 | 1704.857 | 5091.919 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | manual_fused | 8.389 | 5.160 | 13.549 | 520.816 | 5132.331 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_folded | 12.898 | 2.384 | 15.369 | 248.813 | 2371.037 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | sym_unfused | 30.650 | 11.726 | 41.938 | 1054.728 | 10266.222 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_eager | 8.388 | 3.761 | 12.228 | 392.671 | 3844.141 |
| main | padded_roundtrip_n8192_m128_o8 | 1 | candidate | torch_inductor | 8.854 | 1302.903 | 1311.756 | 1684.038 | 5080.253 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | manual_fused | 8.731 | 2.607 | 11.363 | 213.484 | 2052.120 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_folded | 13.149 | 7.119 | 20.289 | 725.610 | 7142.148 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | sym_unfused | 30.500 | 9.916 | 40.406 | 988.804 | 9613.185 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_eager | 8.471 | 3.287 | 11.661 | 319.569 | 3112.634 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | baseline | torch_inductor | 8.944 | 1309.084 | 1317.931 | 1506.094 | 3218.358 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | manual_fused | 8.705 | 2.586 | 11.210 | 254.828 | 2469.632 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_folded | 13.366 | 1.796 | 15.161 | 194.881 | 1823.205 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | sym_unfused | 30.269 | 10.238 | 40.507 | 1019.689 | 9914.938 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_eager | 8.424 | 3.228 | 11.770 | 333.273 | 3257.097 |
| main | padded_roundtrip_n8192_m128_o8 | 8 | candidate | torch_inductor | 8.894 | 1313.864 | 1322.758 | 1513.788 | 3191.317 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | manual_fused | 0.097 | 0.125 | 0.222 | 10.932 | 107.565 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_folded | 4.453 | 0.348 | 4.822 | 38.638 | 346.238 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | sym_unfused | 18.474 | 0.404 | 18.909 | 50.717 | 344.098 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_eager | 0.095 | 0.184 | 0.280 | 12.128 | 120.151 |
| main | rank3_chain_n33_m128_o8 | 1 | baseline | torch_inductor | 1.071 | 1045.890 | 1047.030 | 1066.804 | 1246.572 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | manual_fused | 0.083 | 0.122 | 0.206 | 10.957 | 108.283 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_folded | 4.466 | 0.076 | 4.541 | 11.105 | 70.008 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | sym_unfused | 18.170 | 0.386 | 18.653 | 52.514 | 359.385 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_eager | 0.079 | 0.199 | 0.277 | 12.528 | 120.512 |
| main | rank3_chain_n33_m128_o8 | 1 | candidate | torch_inductor | 0.955 | 1043.148 | 1044.019 | 1063.220 | 1248.545 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | manual_fused | 0.355 | 0.154 | 0.561 | 11.384 | 109.500 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_folded | 4.707 | 0.217 | 4.926 | 24.119 | 198.602 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | sym_unfused | 18.780 | 0.461 | 19.241 | 71.043 | 542.230 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_eager | 0.086 | 0.270 | 0.355 | 24.309 | 242.075 |
| main | rank3_chain_n33_m128_o8 | 8 | baseline | torch_inductor | 0.630 | 1064.980 | 1065.553 | 1085.867 | 1261.492 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | manual_fused | 0.378 | 0.152 | 0.600 | 11.367 | 109.254 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_folded | 4.679 | 0.105 | 4.801 | 13.177 | 89.507 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | sym_unfused | 18.796 | 0.440 | 19.207 | 59.318 | 423.243 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_eager | 0.085 | 0.314 | 0.402 | 26.547 | 264.231 |
| main | rank3_chain_n33_m128_o8 | 8 | candidate | torch_inductor | 0.701 | 1043.997 | 1044.812 | 1064.908 | 1247.606 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | manual_fused | 2.623 | 1.197 | 3.820 | 121.078 | 1187.073 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_folded | 7.026 | 4.982 | 12.008 | 501.904 | 4954.573 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | sym_unfused | 20.458 | 4.642 | 25.097 | 521.895 | 5038.241 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_eager | 2.621 | 1.155 | 3.776 | 127.461 | 1250.497 |
| main | rank3_chain_n512_m128_o8 | 1 | baseline | torch_inductor | 3.091 | 876.273 | 879.365 | 962.776 | 1713.074 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | manual_fused | 2.653 | 1.196 | 3.844 | 120.591 | 1181.573 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_folded | 6.988 | 0.630 | 7.621 | 69.207 | 630.975 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | sym_unfused | 20.107 | 5.813 | 26.447 | 507.182 | 4877.504 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_eager | 2.651 | 1.178 | 3.851 | 122.155 | 1197.647 |
| main | rank3_chain_n512_m128_o8 | 1 | candidate | torch_inductor | 3.117 | 868.040 | 871.208 | 954.233 | 1714.748 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | manual_fused | 2.833 | 1.007 | 3.839 | 56.741 | 537.656 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_folded | 7.173 | 2.998 | 10.171 | 125.693 | 1175.569 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | sym_unfused | 20.473 | 5.357 | 25.831 | 524.321 | 5056.049 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_eager | 2.560 | 0.873 | 3.509 | 88.020 | 856.490 |
| main | rank3_chain_n512_m128_o8 | 8 | baseline | torch_inductor | 3.032 | 901.751 | 904.776 | 974.103 | 1604.344 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | manual_fused | 3.006 | 1.016 | 4.021 | 56.847 | 537.078 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_folded | 7.352 | 0.688 | 8.081 | 53.716 | 468.587 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | sym_unfused | 20.585 | 5.425 | 26.009 | 532.523 | 5137.191 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_eager | 2.707 | 0.979 | 3.583 | 88.394 | 860.714 |
| main | rank3_chain_n512_m128_o8 | 8 | candidate | torch_inductor | 3.233 | 881.824 | 885.057 | 953.611 | 1577.068 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | manual_fused | 31.949 | 17.462 | 49.411 | 1777.698 | 17489.397 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_folded | 36.404 | 79.070 | 115.495 | 7942.315 | 79100.410 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | sym_unfused | 49.982 | 126.802 | 177.003 | 10516.089 | 104469.039 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_eager | 31.947 | 82.408 | 114.558 | 9206.642 | 91896.190 |
| main | rank3_chain_n8192_m128_o8 | 1 | baseline | torch_inductor | 32.564 | 913.196 | 945.759 | 1865.453 | 10226.305 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | manual_fused | 31.616 | 17.479 | 49.159 | 1778.433 | 17493.951 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_folded | 35.915 | 9.126 | 45.041 | 944.850 | 9124.933 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | sym_unfused | 49.411 | 129.912 | 179.190 | 10309.328 | 102352.244 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_eager | 31.614 | 83.351 | 114.981 | 9213.529 | 91921.827 |
| main | rank3_chain_n8192_m128_o8 | 1 | candidate | torch_inductor | 32.180 | 908.285 | 941.135 | 1858.552 | 10198.705 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | manual_fused | 31.787 | 7.336 | 39.124 | 702.777 | 6732.199 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_folded | 36.008 | 16.916 | 53.047 | 1710.820 | 16781.486 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | sym_unfused | 49.668 | 30.664 | 80.153 | 2241.769 | 21864.530 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_eager | 31.446 | 34.064 | 65.460 | 2879.692 | 28494.883 |
| main | rank3_chain_n8192_m128_o8 | 8 | baseline | torch_inductor | 31.981 | 910.440 | 942.406 | 1523.622 | 6807.404 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | manual_fused | 32.099 | 7.108 | 39.516 | 700.631 | 6716.261 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_folded | 36.509 | 5.876 | 42.730 | 596.059 | 5626.325 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | sym_unfused | 49.850 | 24.339 | 74.189 | 2232.338 | 21832.515 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_eager | 31.829 | 34.726 | 66.555 | 2943.551 | 29098.055 |
| main | rank3_chain_n8192_m128_o8 | 8 | candidate | torch_inductor | 32.327 | 909.550 | 941.837 | 1521.279 | 6790.919 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | manual_fused | 204.562 | 0.064 | 204.624 | 208.723 | 245.986 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_folded | 208.853 | 0.049 | 208.900 | 212.948 | 249.756 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | sym_unfused | 222.736 | 0.135 | 223.049 | 229.229 | 285.414 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_eager | 204.560 | 0.105 | 204.776 | 213.099 | 288.766 |
| main | roundtrip1_n33_m128_o8 | 1 | baseline | torch_inductor | 210.459 | 6822.525 | 7032.984 | 7046.098 | 7165.312 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | manual_fused | 204.512 | 0.066 | 204.877 | 208.955 | 246.029 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_folded | 208.760 | 0.044 | 208.804 | 212.548 | 246.588 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | sym_unfused | 222.672 | 0.131 | 222.803 | 229.110 | 284.837 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_eager | 204.509 | 0.097 | 204.604 | 212.940 | 288.730 |
| main | roundtrip1_n33_m128_o8 | 1 | candidate | torch_inductor | 210.368 | 6817.215 | 7025.859 | 7039.054 | 7159.013 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | manual_fused | 188.783 | 0.074 | 189.226 | 193.440 | 232.353 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_folded | 193.224 | 0.052 | 193.276 | 197.499 | 236.350 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | sym_unfused | 207.397 | 0.138 | 207.535 | 214.038 | 281.182 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_eager | 188.491 | 0.104 | 188.595 | 197.475 | 285.062 |
| main | roundtrip1_n33_m128_o8 | 8 | baseline | torch_inductor | 193.972 | 6863.436 | 7055.918 | 7074.490 | 7243.328 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | manual_fused | 190.264 | 0.075 | 190.338 | 194.785 | 235.214 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_folded | 194.501 | 0.051 | 194.960 | 199.079 | 236.532 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | sym_unfused | 208.088 | 0.136 | 208.224 | 215.194 | 278.563 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_eager | 189.984 | 0.120 | 190.104 | 199.610 | 286.026 |
| main | roundtrip1_n33_m128_o8 | 8 | candidate | torch_inductor | 195.442 | 6856.692 | 7053.139 | 7071.593 | 7239.353 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | manual_fused | 0.090 | 0.102 | 0.189 | 8.673 | 85.898 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_folded | 4.590 | 0.118 | 4.746 | 16.022 | 118.197 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | sym_unfused | 18.117 | 0.707 | 18.680 | 61.942 | 453.139 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_eager | 0.088 | 0.295 | 0.383 | 27.206 | 271.057 |
| main | roundtrip1_n512_m128_o8 | 1 | baseline | torch_inductor | 0.580 | 852.748 | 853.328 | 878.062 | 1102.918 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | manual_fused | 0.087 | 0.096 | 0.183 | 9.005 | 89.277 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_folded | 4.406 | 0.077 | 4.500 | 11.481 | 74.507 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | sym_unfused | 17.484 | 0.669 | 18.162 | 60.273 | 444.074 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_eager | 0.085 | 0.292 | 0.377 | 27.928 | 278.386 |
| main | roundtrip1_n512_m128_o8 | 1 | candidate | torch_inductor | 0.576 | 856.094 | 856.681 | 881.681 | 1126.688 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | manual_fused | 0.382 | 0.204 | 0.582 | 11.059 | 106.299 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_folded | 4.613 | 0.313 | 4.926 | 16.726 | 124.540 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | sym_unfused | 18.109 | 0.877 | 18.953 | 76.155 | 599.871 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_eager | 0.103 | 0.443 | 0.553 | 39.871 | 396.709 |
| main | roundtrip1_n512_m128_o8 | 8 | baseline | torch_inductor | 0.610 | 864.037 | 864.657 | 891.836 | 1139.182 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | manual_fused | 0.370 | 0.203 | 0.570 | 10.264 | 98.110 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_folded | 4.716 | 0.284 | 4.981 | 15.101 | 106.932 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | sym_unfused | 18.143 | 0.874 | 18.962 | 77.999 | 610.610 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_eager | 0.098 | 0.602 | 0.701 | 41.013 | 407.493 |
| main | roundtrip1_n512_m128_o8 | 8 | candidate | torch_inductor | 0.598 | 863.784 | 864.373 | 891.393 | 1137.031 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | manual_fused | 4.773 | 0.862 | 5.628 | 91.723 | 873.540 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_folded | 9.166 | 1.652 | 10.827 | 171.493 | 1629.140 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | sym_unfused | 22.531 | 13.202 | 36.336 | 1054.084 | 10302.518 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_eager | 4.771 | 4.616 | 9.581 | 464.434 | 4601.225 |
| main | roundtrip1_n8192_m128_o8 | 1 | baseline | torch_inductor | 5.506 | 867.964 | 873.617 | 979.108 | 1943.068 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | manual_fused | 5.111 | 0.856 | 6.054 | 91.321 | 867.394 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_folded | 9.395 | 0.818 | 10.197 | 91.395 | 825.520 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | sym_unfused | 23.029 | 12.448 | 35.383 | 949.159 | 9256.564 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_eager | 5.107 | 4.673 | 9.732 | 456.161 | 4514.601 |
| main | roundtrip1_n8192_m128_o8 | 1 | candidate | torch_inductor | 5.769 | 872.413 | 878.280 | 984.489 | 1950.023 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | manual_fused | 5.301 | 0.957 | 6.210 | 79.897 | 749.322 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_folded | 9.843 | 1.362 | 11.205 | 94.187 | 848.871 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | sym_unfused | 23.189 | 6.979 | 29.982 | 581.249 | 5593.943 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_eager | 5.023 | 4.281 | 9.304 | 424.368 | 4200.996 |
| main | roundtrip1_n8192_m128_o8 | 8 | baseline | torch_inductor | 5.719 | 878.778 | 884.497 | 982.804 | 1876.509 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | manual_fused | 5.401 | 0.951 | 6.352 | 82.794 | 778.757 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_folded | 9.588 | 0.859 | 10.514 | 84.701 | 759.130 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | sym_unfused | 23.056 | 6.998 | 30.053 | 739.631 | 7190.333 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_eager | 5.110 | 4.356 | 9.746 | 424.428 | 4194.130 |
| main | roundtrip1_n8192_m128_o8 | 8 | candidate | torch_inductor | 5.794 | 882.005 | 888.161 | 986.737 | 1882.885 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | manual_fused | 0.087 | 0.068 | 0.157 | 4.280 | 41.223 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_folded | 4.697 | 0.050 | 4.747 | 9.054 | 48.207 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | sym_unfused | 35.202 | 0.119 | 35.320 | 43.875 | 122.500 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_eager | 0.086 | 0.152 | 0.238 | 11.818 | 117.329 |
| main | roundtrip2_n33_m128_o8 | 1 | baseline | torch_inductor | 0.975 | 906.055 | 907.078 | 920.106 | 1038.544 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | manual_fused | 0.088 | 0.060 | 0.147 | 4.223 | 41.307 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_folded | 4.506 | 0.044 | 4.602 | 8.417 | 42.842 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | sym_unfused | 35.337 | 0.121 | 35.472 | 44.425 | 125.819 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_eager | 0.086 | 0.318 | 0.400 | 12.023 | 117.692 |
| main | roundtrip2_n33_m128_o8 | 1 | candidate | torch_inductor | 0.977 | 894.653 | 895.630 | 908.828 | 1028.811 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | manual_fused | 0.372 | 0.067 | 0.436 | 4.973 | 46.213 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_folded | 4.796 | 0.059 | 4.881 | 9.716 | 53.353 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | sym_unfused | 36.317 | 0.143 | 36.461 | 46.517 | 137.933 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_eager | 0.091 | 0.156 | 0.250 | 13.531 | 134.367 |
| main | roundtrip2_n33_m128_o8 | 8 | baseline | torch_inductor | 0.936 | 906.816 | 907.752 | 926.579 | 1094.820 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | manual_fused | 0.362 | 0.063 | 0.434 | 4.958 | 46.081 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_folded | 4.788 | 0.057 | 4.929 | 9.070 | 47.563 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | sym_unfused | 35.029 | 0.117 | 35.146 | 44.973 | 135.311 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_eager | 0.090 | 0.145 | 0.235 | 13.901 | 136.629 |
| main | roundtrip2_n33_m128_o8 | 8 | candidate | torch_inductor | 0.934 | 902.738 | 903.677 | 922.250 | 1091.092 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | manual_fused | 0.065 | 0.102 | 0.163 | 8.942 | 88.657 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_folded | 4.471 | 0.164 | 4.703 | 20.816 | 167.921 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | sym_unfused | 34.650 | 0.928 | 35.517 | 118.264 | 870.517 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_eager | 0.060 | 0.465 | 0.526 | 45.442 | 453.284 |
| main | roundtrip2_n512_m128_o8 | 1 | baseline | torch_inductor | 0.533 | 879.453 | 879.983 | 906.669 | 1151.780 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | manual_fused | 0.063 | 0.095 | 0.160 | 8.865 | 87.585 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_folded | 4.457 | 0.089 | 4.610 | 12.505 | 85.152 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | sym_unfused | 33.796 | 0.970 | 34.834 | 120.637 | 898.923 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_eager | 0.061 | 0.574 | 0.635 | 46.121 | 459.625 |
| main | roundtrip2_n512_m128_o8 | 1 | candidate | torch_inductor | 0.544 | 886.382 | 886.926 | 913.575 | 1156.025 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | manual_fused | 0.341 | 0.115 | 0.442 | 11.686 | 113.941 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_folded | 4.772 | 0.199 | 5.046 | 18.314 | 138.928 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | sym_unfused | 34.565 | 1.209 | 35.643 | 160.035 | 1290.870 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_eager | 0.067 | 0.713 | 0.780 | 76.543 | 765.079 |
| main | roundtrip2_n512_m128_o8 | 8 | baseline | torch_inductor | 0.548 | 891.590 | 892.138 | 919.299 | 1166.015 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | manual_fused | 0.334 | 0.113 | 0.438 | 10.907 | 106.079 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_folded | 4.799 | 0.150 | 4.949 | 15.062 | 106.194 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | sym_unfused | 34.333 | 1.197 | 35.530 | 159.651 | 1293.110 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_eager | 0.066 | 0.869 | 0.937 | 78.853 | 787.146 |
| main | roundtrip2_n512_m128_o8 | 8 | candidate | torch_inductor | 0.552 | 892.978 | 893.620 | 919.618 | 1171.862 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | manual_fused | 0.106 | 0.893 | 0.998 | 86.958 | 868.151 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_folded | 4.549 | 2.419 | 6.906 | 242.417 | 2383.427 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | sym_unfused | 34.171 | 29.402 | 63.286 | 1994.696 | 19548.898 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_eager | 0.103 | 7.779 | 7.883 | 785.917 | 7858.954 |
| main | roundtrip2_n8192_m128_o8 | 1 | baseline | torch_inductor | 0.637 | 891.465 | 892.102 | 1004.770 | 2037.254 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | manual_fused | 0.106 | 0.869 | 0.978 | 87.861 | 877.793 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_folded | 4.488 | 0.912 | 5.471 | 93.921 | 898.804 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | sym_unfused | 33.791 | 29.587 | 64.332 | 2096.949 | 20575.291 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_eager | 0.102 | 8.122 | 8.220 | 813.065 | 8130.982 |
| main | roundtrip2_n8192_m128_o8 | 1 | candidate | torch_inductor | 0.571 | 889.825 | 890.388 | 1004.753 | 2041.930 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | manual_fused | 0.384 | 0.908 | 1.295 | 75.511 | 750.679 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_folded | 4.801 | 1.813 | 6.613 | 95.686 | 908.289 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | sym_unfused | 34.570 | 11.852 | 46.684 | 1138.255 | 11061.625 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_eager | 0.107 | 7.971 | 8.078 | 793.145 | 7901.837 |
| main | roundtrip2_n8192_m128_o8 | 8 | baseline | torch_inductor | 0.787 | 902.617 | 903.333 | 1000.949 | 1888.364 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | manual_fused | 0.381 | 0.893 | 1.294 | 75.118 | 746.250 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_folded | 4.752 | 0.963 | 5.715 | 79.351 | 748.765 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | sym_unfused | 34.426 | 9.884 | 45.220 | 1089.600 | 10565.286 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_eager | 0.104 | 8.553 | 8.678 | 804.736 | 8024.783 |
| main | roundtrip2_n8192_m128_o8 | 8 | candidate | torch_inductor | 0.806 | 898.655 | 899.563 | 996.371 | 1877.998 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | manual_fused | 0.077 | 0.062 | 0.137 | 4.216 | 41.303 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_folded | 4.820 | 0.055 | 4.920 | 9.735 | 53.504 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | sym_unfused | 67.469 | 0.174 | 67.643 | 81.372 | 202.411 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_eager | 0.076 | 0.189 | 0.265 | 18.258 | 181.837 |
| main | roundtrip4_n33_m128_o8 | 1 | baseline | torch_inductor | 0.931 | 950.978 | 951.880 | 965.257 | 1086.861 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | manual_fused | 0.077 | 0.059 | 0.136 | 4.193 | 41.073 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_folded | 4.773 | 0.046 | 4.959 | 8.899 | 45.194 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | sym_unfused | 69.530 | 0.183 | 69.741 | 84.805 | 221.744 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_eager | 0.075 | 0.201 | 0.280 | 18.348 | 182.624 |
| main | roundtrip4_n33_m128_o8 | 1 | candidate | torch_inductor | 0.907 | 953.603 | 954.486 | 967.902 | 1089.865 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | manual_fused | 0.350 | 0.067 | 0.415 | 4.647 | 43.200 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_folded | 5.154 | 0.062 | 5.213 | 10.479 | 58.329 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | sym_unfused | 67.995 | 0.212 | 68.207 | 85.192 | 230.295 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_eager | 0.077 | 0.230 | 0.315 | 18.519 | 184.003 |
| main | roundtrip4_n33_m128_o8 | 8 | baseline | torch_inductor | 0.569 | 949.720 | 950.384 | 968.538 | 1133.575 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | manual_fused | 0.345 | 0.066 | 0.421 | 4.935 | 45.995 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_folded | 5.098 | 0.054 | 5.260 | 9.771 | 50.777 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | sym_unfused | 67.844 | 0.196 | 68.040 | 83.932 | 228.400 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_eager | 0.075 | 0.224 | 0.299 | 21.267 | 210.474 |
| main | roundtrip4_n33_m128_o8 | 8 | candidate | torch_inductor | 0.563 | 949.315 | 949.936 | 968.615 | 1138.422 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | manual_fused | 0.066 | 0.095 | 0.168 | 9.191 | 91.284 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_folded | 4.754 | 0.269 | 5.033 | 29.499 | 251.914 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | sym_unfused | 67.089 | 1.720 | 68.970 | 231.927 | 1712.048 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_eager | 0.064 | 0.835 | 0.899 | 81.692 | 815.867 |
| main | roundtrip4_n512_m128_o8 | 1 | baseline | torch_inductor | 0.533 | 938.513 | 939.034 | 970.873 | 1260.318 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | manual_fused | 0.067 | 0.097 | 0.167 | 9.109 | 90.133 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_folded | 4.736 | 0.138 | 4.882 | 16.583 | 122.541 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | sym_unfused | 67.212 | 1.780 | 68.972 | 236.529 | 1760.613 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_eager | 0.065 | 0.842 | 0.907 | 81.305 | 812.287 |
| main | roundtrip4_n512_m128_o8 | 1 | candidate | torch_inductor | 0.540 | 941.677 | 942.217 | 973.943 | 1262.359 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | manual_fused | 0.349 | 0.115 | 0.466 | 10.809 | 104.836 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_folded | 5.007 | 0.246 | 5.266 | 20.331 | 157.289 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | sym_unfused | 67.637 | 2.087 | 69.688 | 322.156 | 2617.325 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_eager | 0.067 | 1.182 | 1.249 | 137.377 | 1375.144 |
| main | roundtrip4_n512_m128_o8 | 8 | baseline | torch_inductor | 0.548 | 947.360 | 947.909 | 979.552 | 1252.845 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | manual_fused | 0.331 | 0.111 | 0.452 | 11.101 | 108.027 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_folded | 4.996 | 0.176 | 5.231 | 17.284 | 126.857 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | sym_unfused | 67.354 | 2.100 | 69.454 | 302.669 | 2422.810 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_eager | 0.072 | 1.487 | 1.562 | 141.236 | 1410.997 |
| main | roundtrip4_n512_m128_o8 | 8 | candidate | torch_inductor | 0.554 | 944.583 | 945.118 | 976.000 | 1254.317 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | manual_fused | 0.107 | 0.880 | 0.984 | 87.322 | 872.219 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_folded | 4.840 | 3.436 | 8.311 | 348.402 | 3440.133 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | sym_unfused | 67.124 | 56.775 | 124.030 | 4279.637 | 42056.642 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_eager | 0.105 | 15.684 | 15.790 | 1553.578 | 15512.081 |
| main | roundtrip4_n8192_m128_o8 | 1 | baseline | torch_inductor | 0.621 | 942.448 | 943.095 | 1099.078 | 2519.004 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | manual_fused | 0.106 | 0.932 | 1.038 | 87.602 | 873.914 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_folded | 5.208 | 1.443 | 6.653 | 147.243 | 1424.696 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | sym_unfused | 69.415 | 41.846 | 113.242 | 4220.175 | 41588.516 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_eager | 0.104 | 15.596 | 15.699 | 1537.246 | 15371.204 |
| main | roundtrip4_n8192_m128_o8 | 1 | candidate | torch_inductor | 0.570 | 944.746 | 945.369 | 1101.674 | 2522.623 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | manual_fused | 0.367 | 0.865 | 1.232 | 74.994 | 745.027 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_folded | 5.173 | 1.867 | 6.983 | 120.009 | 1148.024 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | sym_unfused | 67.955 | 24.912 | 92.279 | 2182.379 | 21183.284 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_eager | 0.098 | 15.273 | 15.383 | 1526.855 | 15255.227 |
| main | roundtrip4_n8192_m128_o8 | 8 | baseline | torch_inductor | 0.576 | 969.912 | 970.488 | 1068.672 | 1959.876 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | manual_fused | 0.357 | 0.952 | 1.314 | 75.515 | 750.062 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_folded | 5.057 | 1.115 | 6.169 | 80.250 | 753.758 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | sym_unfused | 67.328 | 21.127 | 90.189 | 2112.207 | 20486.800 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_eager | 0.091 | 15.695 | 15.798 | 1530.968 | 15305.238 |
| main | roundtrip4_n8192_m128_o8 | 8 | candidate | torch_inductor | 0.616 | 964.989 | 965.560 | 1063.725 | 1946.910 |
