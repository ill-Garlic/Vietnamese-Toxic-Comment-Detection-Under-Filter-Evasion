import pandas as pd

df = pd.read_csv("vihsd_raw.csv")
print(df.shape)
print(df.isnull().sum())
print(df["label"].value_counts())

# Phát hiện trùng lặp văn bản
dup_mask = df.duplicated(subset=["text"], keep=False)
print(f"Số bản ghi trùng văn bản: {dup_mask.sum()}")