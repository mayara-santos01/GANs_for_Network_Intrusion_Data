import pandas as pd
import numpy as np

arquivo = "smotenc_generated_samples_labels_concatenated.csv"

df = pd.read_csv(arquivo)

variaveis_categoricas = [
    'proto',
    'service',
    'state',
    'attack_cat'
]

df['service'] = df['service'].replace('-', 'Não disponível')

df['attack_cat'] = df['attack_cat'].astype('category')

variaveis_numericas = [
    coluna for coluna in df.columns
    if coluna not in variaveis_categoricas
    and pd.api.types.is_numeric_dtype(df[coluna])
]

estatisticas = []

for coluna in variaveis_numericas:

    dados = pd.to_numeric(df[coluna], errors='coerce')

    estatisticas.append({
        'Variavel': coluna,
        'Total': dados.count(),
        'Valores_Ausentes': dados.isna().sum(),
        'Media': dados.mean(),
        'Mediana': dados.median(),
        'Desvio_Padrao': dados.std(),
        'Variancia': dados.var(),
        'Minimo': dados.min(),
        'Q1': dados.quantile(0.25),
        'Q2': dados.quantile(0.50),
        'Q3': dados.quantile(0.75),
        'IQR': dados.quantile(0.75) - dados.quantile(0.25),
        'Maximo': dados.max(),
        'Assimetria': dados.skew(),
        'Curtose': dados.kurtosis()
    })

tabela_estatisticas = pd.DataFrame(estatisticas)

frequencias = []

total_registros = len(df)

for coluna in variaveis_categoricas:

    contagem = df[coluna].value_counts(
        dropna=False
    )

    for categoria, quantidade in contagem.items():

        percentual = (quantidade / total_registros) * 100

        frequencias.append({
            'Variavel': coluna,
            'Categoria': (
                'Não disponível'
                if pd.isna(categoria)
                else categoria
            ),
            'Frequencia': quantidade,
            'Percentual': percentual
        })

tabela_frequencias = pd.DataFrame(frequencias)

tabela_estatisticas.to_csv(
    'synthetic_data_stats.csv',
    index=False,
    encoding='utf-8-sig'
)

tabela_frequencias.to_csv(
    'synthetic_data_freq.csv',
    index=False,
    encoding='utf-8-sig'
)


print("\n===== ESTATÍSTICAS NUMÉRICAS =====")
print(tabela_estatisticas.to_string(index=False))

print("\n===== FREQUÊNCIAS CATEGÓRICAS =====")
print(tabela_frequencias.to_string(index=False))

print("\nRelatórios gerados com sucesso!")
