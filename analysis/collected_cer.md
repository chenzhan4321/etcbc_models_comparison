# Collected CER table (auto, 口径对齐 cluster_F)

| Model × dataset | seeds | CER (5-seed mean) | 95% bootstrap CI | seed SD |
|---|---|---|---|---|
| encoder-decoder S2 | 4 | **44.138%** | [44.035, 44.240] | 48.683 pp |
| encoder-only S2 | 5 | **3.876%** | [3.794, 3.963] | 0.032 pp |
| encoder-only S4 | 5 | **3.528%** | [3.450, 3.610] | 0.046 pp |
| MDLM steps2 S2 | 5 | **3.637%** | [3.557, 3.724] | 0.054 pp |
| MDLM steps2 S4 | 5 | **3.420%** | [3.337, 3.502] | 0.086 pp |
| MDLM steps3 S4 | 5 | **3.383%** | [3.302, 3.467] | 0.029 pp |
| BiLSTM-CRF S2 | — | _(待 restore/未完成)_ | — | — |
| BiLSTM-CRF S4 | — | _(待 restore/未完成)_ | — | — |
| Encoder+CRF S2 | — | _(待 restore/未完成)_ | — | — |
| Encoder+CRF S4 | — | _(待 restore/未完成)_ | — | — |
| BERT S2 | — | _(待 restore/未完成)_ | — | — |
| BERT S4 | — | _(待 restore/未完成)_ | — | — |
| enc-dec tuned S2 | — | _(待 restore/未完成)_ | — | — |

## 配对差(A−B,负=A 更优)
| 对比 | 均值差 | 95% CI | 排除0 |
|---|---|---|---|
| MDLM steps3 S4 − encoder-only S4 | -0.145 pp | [-0.180, -0.111] | 是 |
| MDLM steps2 S4 − encoder-only S4 | -0.109 pp | [-0.143, -0.073] | 是 |

_caveat:测试行为滑窗、非独立,行级 bootstrap CI 偏窄,按下界看。_