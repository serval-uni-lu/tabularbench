| architecture   | training    | augmentation   |       ID |   ADV+CTR |      auc |   accuracy |   precision |   recall |      mcc |
|:---------------|:------------|:---------------|---------:|----------:|---------:|-----------:|------------:|---------:|---------:|
| STG            | adversarial | None           | 0.625608 |  0.492707 | 0.864524 |   0.875359 |   0.381463  | 0.626603 | 0.424439 |
| STG            | standard    | None           | 0.776337 |  0.688169 | 0.865793 |   0.782346 |   0.260495  | 0.775641 | 0.36081  |
| TabNet         | adversarial | None           | 0.983793 |  0.874878 | 0.834819 |   0.104083 |   0.0898186 | 0.983974 | 0.002955 |
| TabNet         | standard    | None           | 0.797407 |  0.138736 | 0.870354 |   0.776883 |   0.258585  | 0.796474 | 0.365166 |
| TabTransformer | adversarial | None           | 0.773096 |  0.696272 | 0.869378 |   0.794422 |   0.272316  | 0.772436 | 0.373328 |
| TabTransformer | standard    | None           | 0.755267 |  0.591572 | 0.873801 |   0.809661 |   0.286845  | 0.754808 | 0.383481 |
| RLN            | adversarial | None           | 0.779579 |  0.687844 | 0.866706 |   0.788959 |   0.267621  | 0.778846 | 0.37007  |
| RLN            | standard    | None           | 0.774716 |  0.676823 | 0.869486 |   0.795572 |   0.27381   | 0.774038 | 0.375525 |
| VIME           | adversarial | None           | 0.721232 |  0.592545 | 0.858083 |   0.817424 |   0.290803  | 0.719551 | 0.375877 |
| VIME           | standard    | None           | 0.722853 |  0.594165 | 0.865195 |   0.822887 |   0.298408  | 0.721154 | 0.384242 |
