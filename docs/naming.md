# Naming

## Datasets

| Paper   | API          |
| ------- | ------------ |
| CTU     | ctu_13_neris |
| LCLD    | lcld_v2_iid  |
| MALWARE | malware      |
| URL     | url          |
| WIDS    | wids         |

## Models

The name of the models is in form `{model_name}_{dataloader_name}`.

| Paper                   | API            |
| ----------------------- | -------------- |
| Tabnet                  | tabnet         |
| TabTransformer / TabTr. | tabtransformer |
| RLN                     | torchrln       |
| STG                     | stg            |
| VIME                    | vime           |

## Dataloaders

Dataloaders corresponds to training method.

| Data augmentation | Adversarial Training | API Name       |
| ----------------- | -------------------- | -------------- |
| None              | No                   | default        |
| None              | Yes                  | madry          |
| Cutmix            | No                   | cutmix         |
| Cutmix            | Yes                  | cutmix_madry   |
| CTGAN             | No                   | ctgan          |
| CTGAN             | Yes                  | ctgan_madry    |
| GOGGLE            | No                   | goggle         |
| GOGGLE            | Yes                  | goggle_madry   |
| TableGAN          | No                   | tablegan       |
| TableGAN          | Yes                  | tablegan_madry |
| TVAE              | No                   | tvae           |
| TVAE              | Yes                  | tvae_madry     |
| WGAN              | No                   | wgan           |
| WGAN              | Yes                  | wgan_madry     |
