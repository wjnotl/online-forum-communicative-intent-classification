import pandas as pd

df = pd.read_csv("dataset.csv")
df = df[~df['id'].isin([1, 721])]
df = df.reset_index(drop=True)
df['id'] = df.index + 1
df.to_csv("dataset.csv", index=False)