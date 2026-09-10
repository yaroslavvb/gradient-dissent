# Half-drop training: exhaustive four-branch eligibility results

Complete 48-run panel: 16 independently trained eligibility subsets × seeds 201–203. Every test evaluation keeps all six affines at gain one. Eligibility means a 50% minibatch drop chance during training, not deleting a layer during inference.

Validation-CE-selected checkpoints are primary; epoch 100 is secondary. Values are seed means with 95% t intervals (n=3, df=2), without multiple-comparison correction.

Validation-selected static prefix order: 4 → 1 → 3 → 2. This order was frozen before test analysis and reused for both endpoints. It is not a temporal training schedule.

| Endpoint | Eligible branches | Mean full-test accuracy % [CI] | Accuracy delta vs none pp [CI] | Mean CE [CI] | Subset-mean accuracy range % |
| --- | ---: | --- | --- | --- | --- |
| selected | 0 | 97.990 [96.936, 99.044] | 0.000 [0.000, 0.000] | 0.0748 [0.0673, 0.0824] | 97.990–97.990 |
| selected | 1 | 97.857 [97.704, 98.009] | -0.133 [-1.053, 0.786] | 0.0739 [0.0622, 0.0856] | 97.710–98.040 |
| selected | 2 | 97.933 [97.627, 98.239] | -0.057 [-0.880, 0.765] | 0.0741 [0.0651, 0.0830] | 97.710–98.103 |
| selected | 3 | 97.857 [97.751, 97.962] | -0.133 [-1.095, 0.828] | 0.0730 [0.0656, 0.0804] | 97.777–98.040 |
| selected | 4 | 98.043 [97.179, 98.908] | 0.053 [-0.197, 0.303] | 0.0748 [0.0645, 0.0851] | 98.043–98.043 |
| final | 0 | 98.440 [98.171, 98.709] | 0.000 [0.000, 0.000] | 0.0926 [0.0842, 0.1009] | 98.440–98.440 |
| final | 1 | 98.491 [98.422, 98.560] | 0.051 [-0.155, 0.256] | 0.1031 [0.0990, 0.1071] | 98.470–98.540 |
| final | 2 | 98.496 [98.412, 98.580] | 0.056 [-0.225, 0.337] | 0.1132 [0.1079, 0.1184] | 98.393–98.577 |
| final | 3 | 98.533 [98.323, 98.744] | 0.093 [-0.225, 0.411] | 0.1232 [0.1173, 0.1291] | 98.457–98.573 |
| final | 4 | 98.510 [98.420, 98.600] | 0.070 [-0.288, 0.428] | 0.1372 [0.1270, 0.1475] | 98.510–98.510 |

Each k averages subsets within each seed before forming an interval. The range describes the means of different subsets and is not a confidence interval.

| Endpoint | Eligibility mask (shallow→deep) | Accuracy % [CI] | CE [CI] | Harm % of all images [CI] | Repair % of all images [CI] | Training seconds [CI] |
| --- | --- | --- | --- | --- | --- | --- |
| selected | 0000 | 97.990 [96.936, 99.044] | 0.0748 [0.0673, 0.0824] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 66.34 [65.74, 66.94] |
| selected | 1000 | 97.727 [97.363, 98.090] | 0.0745 [0.0645, 0.0844] | 0.873 [0.191, 1.556] | 0.610 [-0.105, 1.325] | 54.88 [54.55, 55.21] |
| selected | 0100 | 98.040 [97.013, 99.067] | 0.0732 [0.0640, 0.0823] | 0.490 [0.017, 0.963] | 0.540 [-0.058, 1.138] | 59.61 [59.12, 60.10] |
| selected | 1100 | 97.710 [97.277, 98.143] | 0.0742 [0.0590, 0.0895] | 0.927 [0.241, 1.612] | 0.647 [-0.168, 1.462] | 47.65 [47.36, 47.93] |
| selected | 0010 | 97.710 [97.194, 98.226] | 0.0736 [0.0534, 0.0938] | 0.810 [-0.180, 1.800] | 0.530 [-0.051, 1.111] | 61.96 [61.57, 62.36] |
| selected | 1010 | 98.103 [97.082, 99.124] | 0.0732 [0.0702, 0.0762] | 0.560 [-0.151, 1.271] | 0.673 [-0.262, 1.608] | 50.37 [50.15, 50.59] |
| selected | 0110 | 98.053 [97.259, 98.848] | 0.0756 [0.0698, 0.0814] | 0.607 [0.080, 1.134] | 0.670 [-0.239, 1.579] | 55.42 [54.96, 55.88] |
| selected | 1110 | 97.827 [97.244, 98.409] | 0.0721 [0.0561, 0.0881] | 0.947 [0.223, 1.671] | 0.783 [-0.127, 1.694] | 43.48 [43.31, 43.66] |
| selected | 0001 | 97.950 [97.519, 98.381] | 0.0743 [0.0652, 0.0834] | 0.617 [0.537, 0.697] | 0.577 [-0.055, 1.208] | 63.99 [63.59, 64.40] |
| selected | 1001 | 97.967 [97.571, 98.363] | 0.0719 [0.0581, 0.0857] | 0.687 [0.550, 0.823] | 0.663 [-0.111, 1.438] | 52.58 [52.57, 52.59] |
| selected | 0101 | 97.860 [96.989, 98.731] | 0.0747 [0.0570, 0.0924] | 0.857 [-0.031, 1.744] | 0.727 [-0.255, 1.708] | 57.48 [57.05, 57.91] |
| selected | 1101 | 97.777 [97.377, 98.176] | 0.0732 [0.0652, 0.0813] | 0.883 [0.307, 1.459] | 0.670 [-0.108, 1.448] | 45.42 [45.16, 45.68] |
| selected | 0011 | 97.903 [97.138, 98.669] | 0.0747 [0.0656, 0.0838] | 0.637 [0.491, 0.782] | 0.550 [0.078, 1.022] | 59.83 [59.39, 60.27] |
| selected | 1011 | 98.040 [97.354, 98.726] | 0.0704 [0.0661, 0.0747] | 0.630 [0.269, 0.991] | 0.680 [-0.058, 1.418] | 48.10 [48.04, 48.16] |
| selected | 0111 | 97.783 [97.148, 98.419] | 0.0763 [0.0676, 0.0850] | 0.803 [0.421, 1.185] | 0.597 [-0.047, 1.241] | 53.12 [52.80, 53.44] |
| selected | 1111 | 98.043 [97.179, 98.908] | 0.0748 [0.0645, 0.0851] | 0.697 [0.143, 1.250] | 0.750 [-0.011, 1.511] | 41.22 [41.04, 41.40] |
| final | 0000 | 98.440 [98.171, 98.709] | 0.0926 [0.0842, 0.1009] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 66.34 [65.74, 66.94] |
| final | 1000 | 98.540 [98.490, 98.590] | 0.0976 [0.0922, 0.1031] | 0.220 [0.170, 0.270] | 0.320 [0.083, 0.557] | 54.88 [54.55, 55.21] |
| final | 0100 | 98.483 [98.314, 98.652] | 0.1045 [0.0978, 0.1111] | 0.210 [0.144, 0.276] | 0.253 [0.065, 0.441] | 59.61 [59.12, 60.10] |
| final | 1100 | 98.530 [98.336, 98.724] | 0.1102 [0.1004, 0.1200] | 0.237 [0.112, 0.362] | 0.327 [0.167, 0.486] | 47.65 [47.36, 47.93] |
| final | 0010 | 98.470 [98.356, 98.584] | 0.1043 [0.1001, 0.1084] | 0.200 [0.150, 0.250] | 0.230 [0.051, 0.409] | 61.96 [61.57, 62.36] |
| final | 1010 | 98.577 [98.376, 98.777] | 0.1066 [0.0959, 0.1173] | 0.217 [0.116, 0.317] | 0.353 [-0.051, 0.758] | 50.37 [50.15, 50.59] |
| final | 0110 | 98.497 [98.417, 98.577] | 0.1150 [0.0994, 0.1307] | 0.233 [0.059, 0.408] | 0.290 [0.031, 0.549] | 55.42 [54.96, 55.88] |
| final | 1110 | 98.540 [98.366, 98.714] | 0.1167 [0.1082, 0.1252] | 0.297 [0.209, 0.384] | 0.397 [0.109, 0.685] | 43.48 [43.31, 43.66] |
| final | 0001 | 98.470 [98.380, 98.560] | 0.1059 [0.1017, 0.1100] | 0.177 [0.125, 0.228] | 0.207 [0.057, 0.356] | 63.99 [63.59, 64.40] |
| final | 1001 | 98.537 [98.371, 98.702] | 0.1100 [0.0992, 0.1207] | 0.237 [0.091, 0.382] | 0.333 [0.056, 0.610] | 52.58 [52.57, 52.59] |
| final | 0101 | 98.443 [98.229, 98.658] | 0.1188 [0.0985, 0.1391] | 0.237 [0.136, 0.337] | 0.240 [0.085, 0.395] | 57.48 [57.05, 57.91] |
| final | 1101 | 98.563 [98.247, 98.880] | 0.1198 [0.1005, 0.1392] | 0.260 [0.194, 0.326] | 0.383 [0.153, 0.614] | 45.42 [45.16, 45.68] |
| final | 0011 | 98.393 [98.257, 98.530] | 0.1184 [0.1093, 0.1274] | 0.273 [0.151, 0.396] | 0.227 [-0.022, 0.475] | 59.83 [59.39, 60.27] |
| final | 1011 | 98.573 [98.337, 98.809] | 0.1269 [0.1198, 0.1340] | 0.223 [0.049, 0.398] | 0.357 [0.091, 0.622] | 48.10 [48.04, 48.16] |
| final | 0111 | 98.457 [98.222, 98.691] | 0.1294 [0.1180, 0.1409] | 0.280 [0.166, 0.394] | 0.297 [-0.034, 0.627] | 53.12 [52.80, 53.44] |
| final | 1111 | 98.510 [98.420, 98.600] | 0.1372 [0.1270, 0.1475] | 0.343 [0.178, 0.509] | 0.413 [0.220, 0.606] | 41.22 [41.04, 41.40] |

Harm and repair compare each run with mask 0 under the same seed and endpoint. Accuracy change equals repair minus harm. Conditional harm and all ten class-specific metrics, all 32 directed subset edges per endpoint, the 24 path scores, and every seed value are preserved in analysis.json. Paired harm/repair flags for every official test image are retained in paired-errors.npz.

Training time refers to the complete 100 epochs even when a selected checkpoint is displayed. Source hardware is recorded per run; these values are not energy, job latency or invoice cost.

Gallery files separate the fixed 100 label-selected original images from outcome-selected first two harmed/repaired examples per digit. Empty categories remain empty. No new test result selects a preferred subset or changes the frozen order.

Limits: fixed learning rate 0.01, one architecture and reused MNIST test set; three new study seeds do not restore an untouched dataset. Any apparent best subset is conditional on this recipe and does not establish an optimally tuned or test-independent winner.
