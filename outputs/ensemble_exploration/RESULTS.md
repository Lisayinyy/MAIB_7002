# Exploratory model combinations

| Method                          |   CV_MAE | selected_by_validation   |      MAE |      MSE |     RMSE |   WAPE_pct |
|:--------------------------------|---------:|:-------------------------|---------:|---------:|---------:|-----------:|
| Random Forest                   | 0.303716 | False                    | 0.292949 | 0.214081 | 0.462689 |  30.243625 |
| HistGBR                         | 0.305520 | False                    | 0.289176 | 0.206346 | 0.454254 |  29.854081 |
| CatBoost                        | 0.303786 | False                    | 0.287797 | 0.204218 | 0.451905 |  29.711751 |
| RF + CatBoost (50/50)           | 0.301405 | True                     | 0.287288 | 0.205098 | 0.452877 |  29.659168 |
| RF + HistGBR + CatBoost (equal) | 0.301546 | False                    | 0.286387 | 0.203038 | 0.450598 |  29.566128 |
| All five models (equal)         | 0.302323 | False                    | 0.288863 | 0.208964 | 0.457126 |  29.821759 |

Selected by mean validation MAE: **RF + CatBoost (50/50)**.

The original main experiment and its validation-selected Random Forest are unchanged. This is a post-hoc extension after the final week had already been viewed. Base-model parameters and ensemble weights reuse the same CV folds, so CV gains include selection optimism. Weights were frozen before applying them to final-week predictions. No forecast gain establishes a causal pricing gain.
