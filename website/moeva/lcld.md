| architecture   | training    | augmentation   |    ID |   ADV+CTR |      auc |   accuracy |   precision |   recall |      mcc |
|:---------------|:------------|:---------------|------:|----------:|---------:|-----------:|------------:|---------:|---------:|
| STG            | adversarial | None           | 0.156 |    0.1552 | 0.678753 |   0.788237 |    0.431905 | 0.17184  | 0.169947 |
| STG            | standard    | None           | 0.664 |    0.5542 | 0.708675 |   0.645624 |    0.316666 | 0.659934 | 0.245234 |
| TabNet         | adversarial | None           | 0     |    0.001  | 0.655993 |   0.799122 |    0        | 0        | 0        |
| TabNet         | standard    | None           | 0.674 |    0.0082 | 0.722427 |   0.656    |    0.32615  | 0.668318 | 0.261517 |
| TabTransformer | adversarial | None           | 0.739 |    0.714  | 0.711116 |   0.590108 |    0.293306 | 0.738255 | 0.233295 |
| TabTransformer | standard    | None           | 0.695 |    0.1072 | 0.7172   |   0.633159 |    0.314217 | 0.698667 | 0.254153 |
| RLN            | adversarial | None           | 0.695 |    0.6486 | 0.715767 |   0.627545 |    0.309323 | 0.692801 | 0.244785 |
| RLN            | standard    | None           | 0.683 |    0.008  | 0.718666 |   0.641318 |    0.317792 | 0.685063 | 0.25502  |
| VIME           | adversarial | None           | 0.655 |    0.1814 | 0.712662 |   0.651316 |    0.320534 | 0.657081 | 0.249882 |
| VIME           | standard    | None           | 0.67  |    0.2406 | 0.714154 |   0.644575 |    0.317863 | 0.671332 | 0.250644 |
