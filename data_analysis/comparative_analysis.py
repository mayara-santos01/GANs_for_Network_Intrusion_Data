import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

arquivo_original = "UNSW_NB15_testing-set.csv"
arquivo_sintetico = "smotenc_generated_samples_labels_concatenated.csv"

arquivo_saida = "comparacao_dados_originais_sinteticos.xlsx"

# Variáveis que devem ser tratadas como categóricas
categoricas_forcadas = ["proto", "service", "state", "attack_cat"]

# Colunas que não devem participar da análise
colunas_ignoradas = ["id"]


# ============================================================
# 2. LEITURA DOS ARQUIVOS
# ============================================================

df_original = pd.read_csv(arquivo_original)
df_sintetico = pd.read_csv(arquivo_sintetico)

# Remove a coluna id do conjunto original
df_original = df_original.drop(
    columns=colunas_ignoradas,
    errors="ignore"
)

# Remove id também do sintético, caso exista
df_sintetico = df_sintetico.drop(
    columns=colunas_ignoradas,
    errors="ignore"
)

# Considera somente as colunas presentes nos dois arquivos
colunas_comuns = sorted(
    set(df_original.columns) & set(df_sintetico.columns)
)

df_original = df_original[colunas_comuns]
df_sintetico = df_sintetico[colunas_comuns]

# Identifica variáveis categóricas
colunas_categoricas = [
    col for col in categoricas_forcadas
    if col in colunas_comuns
]

# Identifica variáveis numéricas
colunas_numericas = [
    col for col in colunas_comuns
    if col not in colunas_categoricas
    and pd.api.types.is_numeric_dtype(df_original[col])
    and pd.api.types.is_numeric_dtype(df_sintetico[col])
]

# Identifica outras variáveis não numéricas
outras_categoricas = [
    col for col in colunas_comuns
    if col not in colunas_categoricas
    and col not in colunas_numericas
]

colunas_categoricas.extend(outras_categoricas)


# ============================================================
# 3. PREPARAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# ============================================================

for col in colunas_categoricas:

    df_original[col] = df_original[col].astype("string")
    df_sintetico[col] = df_sintetico[col].astype("string")

    # Na coluna service, o hífen representa ausência de serviço
    if col == "service":
        df_original[col] = df_original[col].replace("-", "Não disponível")
        df_sintetico[col] = df_sintetico[col].replace("-", "Não disponível")

    # Padroniza valores ausentes
    df_original[col] = df_original[col].fillna("Não disponível")
    df_sintetico[col] = df_sintetico[col].fillna("Não disponível")


# ============================================================
# 4. COMPARAÇÃO DAS VARIÁVEIS NUMÉRICAS
# ============================================================

resultados_numericos = []

estatisticas = {
    "Média": lambda x: x.mean(),
    "Mediana": lambda x: x.median(),
    "Desvio padrão": lambda x: x.std(),
    "Variância": lambda x: x.var(),
    "Mínimo": lambda x: x.min(),
    "Máximo": lambda x: x.max(),
    "Q1": lambda x: x.quantile(0.25),
    "Q2": lambda x: x.quantile(0.50),
    "Q3": lambda x: x.quantile(0.75),
    "IQR": lambda x: x.quantile(0.75) - x.quantile(0.25),
    "Assimetria": lambda x: x.skew(),
    "Curtose": lambda x: x.kurtosis()
}

for col in colunas_numericas:

    original = df_original[col].dropna()
    sintetico = df_sintetico[col].dropna()

    for nome, funcao in estatisticas.items():

        valor_original = funcao(original)
        valor_sintetico = funcao(sintetico)

        diferenca = abs(valor_original - valor_sintetico)

        # Diferença relativa em percentual
        if valor_original != 0:
            diferenca_relativa = (
                diferenca / abs(valor_original)
            ) * 100
        else:
            diferenca_relativa = np.nan

        resultados_numericos.append({
            "Variável": col,
            "Estatística": nome,
            "Diferença absoluta": diferenca,
            "Diferença relativa (%)": diferenca_relativa
        })

    # Quantidade de valores únicos
    diferenca_unicos = abs(
        original.nunique() - sintetico.nunique()
    )

    resultados_numericos.append({
        "Variável": col,
        "Estatística": "Quantidade de valores únicos",
        "Diferença absoluta": diferenca_unicos,
        "Diferença relativa (%)": np.nan
    })

    # Teste Kolmogorov-Smirnov
    ks = ks_2samp(original, sintetico)

    # Distância de Wasserstein
    wasserstein = wasserstein_distance(original, sintetico)

    # Wasserstein normalizada pelo desvio padrão original
    desvio_original = original.std()

    if desvio_original != 0:
        wasserstein_normalizada = wasserstein / desvio_original
    else:
        wasserstein_normalizada = np.nan

    resultados_numericos.append({
        "Variável": col,
        "Estatística": "KS - Estatística",
        "Diferença absoluta": ks.statistic,
        "Diferença relativa (%)": np.nan
    })

    resultados_numericos.append({
        "Variável": col,
        "Estatística": "KS - Valor-p",
        "Diferença absoluta": ks.pvalue,
        "Diferença relativa (%)": np.nan
    })

    resultados_numericos.append({
        "Variável": col,
        "Estatística": "Wasserstein",
        "Diferença absoluta": wasserstein,
        "Diferença relativa (%)": np.nan
    })

    resultados_numericos.append({
        "Variável": col,
        "Estatística": "Wasserstein normalizada",
        "Diferença absoluta": wasserstein_normalizada,
        "Diferença relativa (%)": np.nan
    })


tabela_numerica = pd.DataFrame(resultados_numericos)


# ============================================================
# 5. COMPARAÇÃO DAS VARIÁVEIS CATEGÓRICAS
# ============================================================

resultados_categoricos = []
resultados_frequencias = []

for col in colunas_categoricas:

    freq_original = df_original[col].value_counts(normalize=True)
    freq_sintetico = df_sintetico[col].value_counts(normalize=True)

    categorias = sorted(
        set(freq_original.index) | set(freq_sintetico.index)
    )

    # Alinha as frequências para considerar categorias ausentes
    p = freq_original.reindex(categorias, fill_value=0)
    q = freq_sintetico.reindex(categorias, fill_value=0)

    # Diferença absoluta entre proporções
    diferencas = abs(p - q)

    # Distância de variação total
    tvd = diferencas.sum() / 2

    # Distância de Jensen-Shannon
    js = jensenshannon(p, q, base=2) ** 2

    # Diferença na quantidade de categorias
    diferenca_unicos = abs(
        df_original[col].nunique() -
        df_sintetico[col].nunique()
    )

    resultados_categoricos.append({
        "Variável": col,
        "Diferença de categorias únicas": diferenca_unicos,
        "Distância de variação total": tvd,
        "Distância Jensen-Shannon": js
    })

    # Comparação individual das frequências
    for categoria in categorias:

        resultados_frequencias.append({
            "Variável": col,
            "Categoria": categoria,
            "Diferença absoluta de frequência (p.p.)":
                abs(p[categoria] - q[categoria]) * 100
        })


tabela_categorica = pd.DataFrame(resultados_categoricos)

tabela_frequencias = pd.DataFrame(resultados_frequencias)


# ============================================================
# 6. TABELA DE VARIÁVEIS E JUSTIFICATIVAS
# ============================================================

justificativas = {
    "dur": "Avaliar a duração das conexões de rede.",
    "proto": "Identificar os protocolos utilizados nas conexões.",
    "service": "Avaliar os serviços de rede associados ao tráfego.",
    "state": "Analisar os estados das conexões de rede.",
    "spkts": "Avaliar a quantidade de pacotes enviados pela origem.",
    "dpkts": "Avaliar a quantidade de pacotes enviados pelo destino.",
    "sbytes": "Analisar o volume de bytes enviados pela origem.",
    "dbytes": "Analisar o volume de bytes enviados pelo destino.",
    "rate": "Avaliar a taxa de transmissão de dados.",
    "sttl": "Analisar o TTL dos pacotes enviados pela origem.",
    "dttl": "Analisar o TTL dos pacotes enviados pelo destino.",
    "sload": "Avaliar a carga de transmissão da origem.",
    "dload": "Avaliar a carga de transmissão do destino.",
    "sloss": "Identificar perdas de pacotes na origem.",
    "dloss": "Identificar perdas de pacotes no destino.",
    "sinpkt": "Avaliar o intervalo entre pacotes da origem.",
    "dinpkt": "Avaliar o intervalo entre pacotes do destino.",
    "sjit": "Analisar a variação temporal dos pacotes da origem.",
    "djit": "Analisar a variação temporal dos pacotes do destino.",
    "swin": "Avaliar o tamanho da janela TCP da origem.",
    "stcpb": "Analisar o número de sequência TCP da origem.",
    "dtcpb": "Analisar o número de sequência TCP do destino.",
    "dwin": "Avaliar o tamanho da janela TCP do destino.",
    "tcprtt": "Avaliar o tempo de ida e volta da conexão TCP.",
    "synack": "Analisar o tempo entre SYN e SYN-ACK.",
    "ackdat": "Analisar o tempo entre ACK e recebimento de dados.",
    "smean": "Avaliar o tamanho médio dos pacotes da origem.",
    "dmean": "Avaliar o tamanho médio dos pacotes do destino.",
    "trans_depth": "Avaliar a profundidade de transação HTTP.",
    "response_body_len": "Analisar o tamanho do corpo das respostas.",
    "ct_srv_src": "Avaliar conexões recentes ao mesmo serviço e origem.",
    "ct_state_ttl": "Analisar a relação entre estado da conexão e TTL.",
    "ct_dst_ltm": "Avaliar conexões recentes para o mesmo destino.",
    "ct_src_dport_ltm": "Identificar padrões de portas de destino por origem.",
    "ct_dst_sport_ltm": "Identificar padrões de portas de origem por destino.",
    "ct_dst_src_ltm": "Avaliar a frequência de comunicação entre origem e destino.",
    "is_ftp_login": "Identificar a ocorrência de autenticação FTP.",
    "ct_ftp_cmd": "Avaliar a quantidade de comandos FTP.",
    "ct_flw_http_mthd": "Avaliar a quantidade de métodos HTTP.",
    "ct_src_ltm": "Analisar conexões recentes da mesma origem.",
    "ct_srv_dst": "Avaliar conexões recentes ao mesmo serviço e destino.",
    "is_sm_ips_ports": "Identificar conexões com IPs e portas iguais.",
    "attack_cat": "Identificar a categoria de ataque ou tráfego normal."
}

linhas_variaveis = []

for col in colunas_comuns:

    if col in colunas_categoricas:
        tipo = "Categórica"
    else:
        tipo = "Numérica"

    motivo = justificativas.get(
        col,
        "Avaliar a distribuição e a fidelidade estatística da variável."
    )

    linhas_variaveis.append({
        "Variável": col,
        "Tipo": tipo,
        "Motivo da inclusão": motivo
    })

# Registra também a coluna ignorada
linhas_variaveis.append({
    "Variável": "id",
    "Tipo": "Ignorada",
    "Motivo da inclusão": "Coluna utilizada apenas para identificar e numerar registros."
})

tabela_variaveis = pd.DataFrame(linhas_variaveis)

# ============================================================
# 7. EXPORTAÇÃO PARA CSV
# ============================================================

tabela_numerica.to_csv(
    "numeric_vars_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_categorica.to_csv(
    "categoric_vars_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_frequencias.to_csv(
    "categoric_vars_freqs.csv",
    index=False,
    encoding="utf-8-sig"
)

tabela_variaveis.to_csv(
    "variables.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Comparação concluída!")
