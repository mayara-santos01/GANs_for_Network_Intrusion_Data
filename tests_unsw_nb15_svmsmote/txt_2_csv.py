import csv
import pandas as pd

arquivo_txt = "svmsmote_decoded_generated_samples.txt"
arquivo_csv = "svmsmote_decoded_generated_samples.csv"

df = pd.read_csv(
    arquivo_txt,
    sep=r"\s+",
    engine="python"
)

df.to_csv("arquivo_csv", index=False, encoding="utf-8-sig")

print("Conversão concluída")

