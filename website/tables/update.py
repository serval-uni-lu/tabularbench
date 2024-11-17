import numpy as np
import pandas as pd


def run():

    print("Updating tables")

    for path in [
        "./malware.md",
        "./url.md",
        "./lcld.md",
        "./wids.md",
        "ctu.md",
    ]:
        process(path)

    # process("./malware.md")


def process(path: str):

    df = (
        pd.read_csv(path, sep="|", header=0, skipinitialspace=True)
        .dropna(axis=1, how="all")
        .iloc[1:]
    )
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()
    # print(df)

    percentage_cols = [
        "ID",
        "ADV+CTR",
        "ADV",
        "Accuracy",
        "Precision",
        "Recall",
    ]
    raw_data = ["AUC", "MCC"]

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
