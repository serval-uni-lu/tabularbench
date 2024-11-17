from typing import List, Union

import numpy as np
import pandas as pd


def run():
    print("Updating tables")

    for path in [
        "./malware.md",
        "./url.md",
        "./lcld.md",
        "./wids.md",
        "./ctu.md",
        "./moeva/ctu.md",
        "./moeva/url.md",
        "./moeva/lcld.md",
        "./moeva/wids.md",
    ]:
        process(path)

    # process("./malware.md")


def update_names(to_update: Union[List[str], pd.Series]):
    names = {
        "architecture": "Architecture",
        "training": "Training",
        "augmentation": "Augmentation",
        "ID": "ID",
        "ADV+CTR": "ADV+CTR",
        "auc": "AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "mcc": "MCC",
        "ctgan": "CT-GAN",
        "tablegan": "TableGAN",
        "tvae": "TVAE",
        "none": "None",
        "standard": "Std",
        "adversarial": "Adv",
    }

    def map_name(name):
        return names.get(name, name)

    if isinstance(to_update, pd.Series):
        return to_update.map(map_name)
    else:
        return [map_name(name) for name in to_update]


def update_names_everywhere(df: pd.DataFrame):
    df.columns = update_names(df.columns)
    for col in df.columns:
        df[col] = update_names(df[col])

    return df


def process(path: str):
    df = (
        pd.read_csv(path, sep="|", header=0, skipinitialspace=True)
        .dropna(axis=1, how="all")
        .iloc[1:]
    )
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    df = update_names_everywhere(df)
    # print(df)

    percentage_cols = [
        col
        for col in [
            "ID",
            "ADV+CTR",
            "ADV",
            "Accuracy",
            "Precision",
            "Recall",
        ]
        if col in df.columns
    ]
    raw_data = [col for col in ["AUC", "MCC"] if col in df.columns]

    for col in percentage_cols + raw_data:
        df[col] = df[col].astype(float)

    df["to_rank"] = df["Accuracy"].replace(np.nan, 0) + df["ADV+CTR"].replace(
        np.nan, 0
    )
    print(df["to_rank"])
    df = df.sort_values(by="to_rank", ascending=False)

    df = df.drop(columns=["to_rank"], inplace=False)
    df["Rank"] = np.arange(1, len(df) + 1)

    df = df[["Rank"] + [col for col in df.columns if col != "Rank"]].copy()

    precision = 4

    for col in percentage_cols:
        df[col] = df[col].apply(lambda x: f"{x:.{precision-2}%}")
        df[col] = df[col].apply(lambda x: x if x != "nan%" else "-")

    for col in raw_data:
        df[col] = df[col].apply(lambda x: f"{x:.{precision}f}").astype(str)
        df[col] = df[col].apply(lambda x: x if x != "nan" else "-")

    df.to_markdown(path.replace(".md", "_new.md"), index=False)

    # print(df)


if __name__ == "__main__":
    run()
