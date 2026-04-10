import pandas as pd

def load_data(path1="data/raw/mhqa.csv", path2="data/raw/mhqa-b.csv"):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    df = pd.concat([df1, df2], ignore_index=True)
    return df


def basic_info(df):
    print("Shape:", df.shape)
    print("Columns:", df.columns)
    print(df.info())