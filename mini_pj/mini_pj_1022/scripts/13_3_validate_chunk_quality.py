import pandas as pd

path = r"C:\mini_pj\data\processed\validation_samples\validation_results\validation_all.csv"

df = pd.read_csv(path)

duplicates = df[df["duplicate_text"] == True]

print("중복 청크:", len(duplicates))

print(
    duplicates[
        ["assembly", "date", "source_id", "chunk_id", "text"]
    ].head(10).to_string(index=False, max_colwidth=100)
)