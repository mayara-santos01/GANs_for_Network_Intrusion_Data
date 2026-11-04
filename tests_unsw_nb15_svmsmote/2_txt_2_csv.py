import pandas as pd

# Arquivos de entrada e saída
arquivo_entrada = "svmsmote_decoded_generated_samples.txt"
arquivo_saida = "svmsmote_decoded_generated_samples.csv"

# Ler os nomes das colunas da primeira linha
with open(arquivo_entrada, "r") as arquivo:
    colunas = arquivo.readline().split()

# Ler os dados, ignorando a primeira coluna (numeração)
df = pd.read_csv(
    arquivo_entrada,
    sep=r"\s+",
    skiprows=1,
    header=None,
    names=["index"] + colunas
)

# Remover a coluna de numeração
df = df.drop(columns=["index"])

# Exportar para CSV
df.to_csv(arquivo_saida, index=False)

print(f"Conversão concluída! Arquivo salvo: {arquivo_saida}")
print(f"Total de registros: {len(df)}")
print(f"Total de colunas: {len(df.columns)}")
