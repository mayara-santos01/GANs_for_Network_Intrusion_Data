import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_ENTRADA = Path("datasets_por_classe")
PASTA_RELATORIOS = Path("relatorios_por_classe")

VARIAVEIS_CATEGORICAS = [
    "proto",
    "service",
    "state"
]

VARIAVEIS_BINARIAS = [
    "is_ftp_login",
    "is_sm_ips_ports"
]

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def escrever_titulo(arquivo, titulo):

    arquivo.write("\n")
    arquivo.write("=" * 75 + "\n")
    arquivo.write(titulo.upper() + "\n")
    arquivo.write("=" * 75 + "\n\n")


def formatar_numero(valor):

    if pd.isna(valor):
        return "Não disponível"

    return f"{valor:.6f}"


def interpretar_assimetria(valor):

    if pd.isna(valor):
        return "Não foi possível determinar a assimetria."

    if valor > 1:
        return "Forte assimetria positiva."

    elif valor > 0.5:
        return "Assimetria positiva moderada."

    elif valor < -1:
        return "Forte assimetria negativa."

    elif valor < -0.5:
        return "Assimetria negativa moderada."

    return "Assimetria reduzida, distribuição relativamente simétrica."


def interpretar_curtose(valor):

    if pd.isna(valor):
        return "Não foi possível avaliar a curtose."

    if valor > 1:
        return "Curtose elevada, indicando caudas potencialmente pesadas."

    elif valor < -1:
        return "Curtose baixa, indicando caudas relativamente leves."

    return "Curtose próxima da referência de uma distribuição normal."


# ============================================================
# IDENTIFICAÇÃO DOS TIPOS DE VARIÁVEIS
# ============================================================

def identificar_tipos(dados, features):

    categoricas = [
        coluna for coluna in VARIAVEIS_CATEGORICAS
        if coluna in features
    ]

    binarias = [
        coluna for coluna in VARIAVEIS_BINARIAS
        if coluna in features
    ]

    numericas = [
        coluna for coluna in features
        if coluna not in categoricas
        and coluna not in binarias
    ]

    numericas_validas = []

    for coluna in numericas:

        valores = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

        if valores.notna().any():
            numericas_validas.append(coluna)

    return categoricas, binarias, numericas_validas


# ============================================================
# ANÁLISE DE VALORES AUSENTES
# ============================================================

def analisar_ausentes(dados, arquivo):

    escrever_titulo(
        arquivo,
        "1. Análise de valores ausentes"
    )

    total = dados.size

    ausentes = dados.isna().sum()
    vazios = (dados == "").sum()

    total_ausentes = int(ausentes.sum() + vazios.sum())

    percentual = (
        total_ausentes / total * 100
        if total > 0 else 0
    )

    arquivo.write(f"Total de células: {total}\n")
    arquivo.write(f"Valores ausentes ou vazios: {total_ausentes}\n")
    arquivo.write(f"Percentual: {percentual:.2f}%\n\n")

    if total_ausentes == 0:

        arquivo.write(
            "Não foram identificados valores ausentes.\n"
        )

    else:

        arquivo.write("Variáveis com valores ausentes:\n\n")

        for coluna in dados.columns:

            quantidade = (
                dados[coluna].isna().sum()
                + (dados[coluna] == "").sum()
            )

            if quantidade > 0:

                percentual_coluna = (
                    quantidade / len(dados) * 100
                )

                arquivo.write(
                    f"{coluna}: {quantidade} "
                    f"({percentual_coluna:.2f}%)\n"
                )


# ============================================================
# ANÁLISE DE DUPLICATAS
# ============================================================

def analisar_duplicatas(dados, arquivo):

    escrever_titulo(
        arquivo,
        "2. Análise de registros duplicados"
    )

    duplicatas = dados.duplicated().sum()

    percentual = (
        duplicatas / len(dados) * 100
        if len(dados) > 0 else 0
    )

    arquivo.write(f"Registros duplicados: {duplicatas}\n")
    arquivo.write(f"Percentual: {percentual:.2f}%\n\n")

    if duplicatas == 0:

        arquivo.write(
            "Não foram encontrados registros duplicados.\n"
        )

    else:

        arquivo.write(
            "Foram encontrados registros duplicados. "
            "Recomenda-se investigar sua ocorrência.\n"
        )


# ============================================================
# ANÁLISE DAS VARIÁVEIS NUMÉRICAS
# ============================================================

def analisar_numericas(dados, numericas, arquivo):

    escrever_titulo(
        arquivo,
        "3. Análise estatística das variáveis numéricas"
    )

    for coluna in numericas:

        valores = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        ).dropna()

        arquivo.write("\n")
        arquivo.write("-" * 75 + "\n")
        arquivo.write(f"VARIÁVEL: {coluna}\n")
        arquivo.write("-" * 75 + "\n\n")

        if len(valores) == 0:

            arquivo.write(
                "Não existem valores numéricos válidos.\n"
            )

            continue

        media = valores.mean()
        mediana = valores.median()
        desvio = valores.std()

        minimo = valores.min()
        maximo = valores.max()

        q1 = valores.quantile(0.25)
        q3 = valores.quantile(0.75)

        iqr = q3 - q1

        cv = (
            desvio / abs(media)
            if abs(media) > 1e-10 else np.nan
        )

        assimetria = valores.skew()
        curtose = valores.kurtosis()

        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr

        outliers = (
            (valores < limite_inferior)
            | (valores > limite_superior)
        )

        quantidade_outliers = int(outliers.sum())

        percentual_outliers = (
            quantidade_outliers / len(valores) * 100
        )

        zeros = int((valores == 0).sum())

        percentual_zeros = (
            zeros / len(valores) * 100
        )

        arquivo.write(
            f"Observações válidas: {len(valores)}\n"
        )

        arquivo.write(
            f"Valores distintos: {valores.nunique()}\n"
        )

        arquivo.write(
            f"Média: {formatar_numero(media)}\n"
        )

        arquivo.write(
            f"Mediana: {formatar_numero(mediana)}\n"
        )

        arquivo.write(
            f"Desvio padrão: {formatar_numero(desvio)}\n"
        )

        arquivo.write(
            f"Mínimo: {formatar_numero(minimo)}\n"
        )

        arquivo.write(
            f"Máximo: {formatar_numero(maximo)}\n"
        )

        arquivo.write(
            f"Q1: {formatar_numero(q1)}\n"
        )

        arquivo.write(
            f"Q3: {formatar_numero(q3)}\n"
        )

        arquivo.write(
            f"IQR: {formatar_numero(iqr)}\n"
        )

        arquivo.write(
            f"Coeficiente de variação: {formatar_numero(cv)}\n"
        )

        arquivo.write(
            f"Assimetria: {formatar_numero(assimetria)}\n"
        )

        arquivo.write(
            f"Curtose: {formatar_numero(curtose)}\n"
        )

        arquivo.write(
            f"Outliers: {quantidade_outliers} "
            f"({percentual_outliers:.2f}%)\n"
        )

        arquivo.write(
            f"Valores zero: {zeros} "
            f"({percentual_zeros:.2f}%)\n"
        )

        # Interpretação estatística.

        arquivo.write("\nINTERPRETAÇÃO:\n\n")

        if valores.nunique() <= 1:

            arquivo.write(
                "A variável apresenta apenas um valor distinto, "
                "não possuindo variabilidade neste conjunto.\n"
            )

            continue

        arquivo.write(
            interpretar_assimetria(assimetria) + "\n"
        )

        arquivo.write(
            interpretar_curtose(curtose) + "\n"
        )

        if not pd.isna(cv):

            if cv < 0.15:

                arquivo.write(
                    "Baixa dispersão relativa dos dados.\n"
                )

            elif cv < 0.30:

                arquivo.write(
                    "Dispersão relativa moderada.\n"
                )

            else:

                arquivo.write(
                    "Elevada dispersão relativa dos dados.\n"
                )

        else:

            arquivo.write(
                "Coeficiente de variação não adequado "
                "devido à média próxima de zero.\n"
            )

        if percentual_outliers > 10:

            arquivo.write(
                "Proporção elevada de possíveis valores extremos.\n"
            )

        elif percentual_outliers > 0:

            arquivo.write(
                "Foram identificados possíveis valores extremos.\n"
            )

        else:

            arquivo.write(
                "Não foram identificados outliers pelo critério IQR.\n"
            )

        if percentual_zeros > 50:

            arquivo.write(
                "Mais da metade dos valores é igual a zero.\n"
            )


# ============================================================
# ANÁLISE DAS VARIÁVEIS CATEGÓRICAS
# ============================================================

def analisar_categoricas(dados, categoricas, arquivo):

    escrever_titulo(
        arquivo,
        "4. Análise das variáveis categóricas"
    )

    for coluna in categoricas:

        arquivo.write("\n")
        arquivo.write("-" * 75 + "\n")
        arquivo.write(f"VARIÁVEL: {coluna}\n")
        arquivo.write("-" * 75 + "\n\n")

        valores = dados[coluna].replace("", np.nan).dropna()

        if len(valores) == 0:

            arquivo.write(
                "Não existem valores válidos.\n"
            )

            continue

        frequencias = valores.value_counts()
        percentuais = valores.value_counts(normalize=True) * 100

        arquivo.write(
            f"Observações válidas: {len(valores)}\n"
        )

        arquivo.write(
            f"Categorias distintas: {valores.nunique()}\n\n"
        )

        arquivo.write(
            f"{'Categoria':<30} {'Quantidade':>15} {'Percentual':>15}\n"
        )

        arquivo.write("-" * 65 + "\n")

        for categoria, quantidade in frequencias.items():

            arquivo.write(
                f"{str(categoria):<30.30} "
                f"{quantidade:>15} "
                f"{percentuais[categoria]:>14.2f}%\n"
            )

        arquivo.write("\nINTERPRETAÇÃO:\n")

        dominante = percentuais.iloc[0]

        if dominante > 80:

            arquivo.write(
                "Existe forte concentração em uma única categoria.\n"
            )

        elif dominante > 50:

            arquivo.write(
                "Existe concentração moderada em uma categoria.\n"
            )

        else:

            arquivo.write(
                "A distribuição apresenta maior diversificação.\n"
            )


# ============================================================
# ANÁLISE DAS VARIÁVEIS BINÁRIAS
# ============================================================

def analisar_binarias(dados, binarias, arquivo):

    escrever_titulo(
        arquivo,
        "5. Análise das variáveis binárias"
    )

    for coluna in binarias:

        arquivo.write("\n")
        arquivo.write("-" * 75 + "\n")
        arquivo.write(f"VARIÁVEL: {coluna}\n")
        arquivo.write("-" * 75 + "\n\n")

        valores = dados[coluna].replace("", np.nan).dropna()

        if len(valores) == 0:

            arquivo.write(
                "Não existem valores válidos.\n"
            )

            continue

        frequencias = valores.value_counts()
        percentuais = valores.value_counts(normalize=True) * 100

        for categoria, quantidade in frequencias.items():

            arquivo.write(
                f"Valor {categoria}: {quantidade} "
                f"({percentuais[categoria]:.2f}%)\n"
            )

        arquivo.write("\nINTERPRETAÇÃO:\n")

        if valores.nunique() == 1:

            arquivo.write(
                "A variável apresenta apenas um valor distinto.\n"
            )

        elif percentuais.iloc[0] > 95:

            arquivo.write(
                "Existe forte predominância de um dos valores.\n"
            )

        else:

            arquivo.write(
                "Os valores apresentam distribuição mais diversificada.\n"
            )


# ============================================================
# INFORMAÇÕES DA CLASSE
# ============================================================

def analisar_classe(dados, arquivo):

    escrever_titulo(
        arquivo,
        "6. Identificação da classe"
    )

    class_id = dados.iloc[0, -2]
    class_name = dados.iloc[0, -1]

    arquivo.write(f"Class ID: {class_id}\n")
    arquivo.write(f"Class name: {class_name}\n")
    arquivo.write(f"Total de registros: {len(dados)}\n\n")

    arquivo.write(
        "Este relatório considera exclusivamente as amostras "
        "pertencentes à classe identificada acima.\n"
    )


# ============================================================
# CLASSIFICAÇÃO DOS TIPOS DE VARIÁVEIS
# ============================================================

def analisar_tipos(
    features,
    categoricas,
    binarias,
    numericas,
    arquivo
):

    escrever_titulo(
        arquivo,
        "7. Classificação dos tipos de variáveis"
    )

    total = len(features)

    arquivo.write(f"Total de features: {total}\n\n")

    grupos = [
        ("Numéricas", numericas),
        ("Categóricas", categoricas),
        ("Binárias", binarias)
    ]

    for nome, colunas in grupos:

        percentual = (
            len(colunas) / total * 100
            if total > 0 else 0
        )

        arquivo.write(
            f"{nome}: {len(colunas)} "
            f"({percentual:.2f}%)\n"
        )

        for coluna in colunas:

            arquivo.write(f"  - {coluna}\n")

        arquivo.write("\n")


# ============================================================
# PROCESSAMENTO DE CADA CSV
# ============================================================

def gerar_relatorio(caminho_csv):

    dados = pd.read_csv(
        caminho_csv,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    if len(dados.columns) < 3:
        raise ValueError(
            "O CSV precisa conter features, class_id e class_name."
        )

    # As duas últimas colunas são as classes.
    features = list(dados.columns[:-2])

    coluna_class_id = dados.columns[-2]
    coluna_class_name = dados.columns[-1]

    if dados.empty:
        raise ValueError("O CSV está vazio.")

    # Verifica se o arquivo contém apenas uma classe.
    if dados[coluna_class_id].nunique() != 1:
        raise ValueError(
            "O arquivo contém mais de um class_id."
        )

    if dados[coluna_class_name].nunique() != 1:
        raise ValueError(
            "O arquivo contém mais de um class_name."
        )

    # Identifica os tipos.
    categoricas, binarias, numericas = identificar_tipos(
        dados,
        features
    )

    class_name = dados[coluna_class_name].iloc[0]

    nome_relatorio = f"relatorio_{class_name}.txt"

    caminho_relatorio = (
        PASTA_RELATORIOS / nome_relatorio
    )

    with open(
        caminho_relatorio,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "RELATÓRIO DE ANÁLISE DA QUALIDADE DOS DADOS\n"
        )

        arquivo.write(
            f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        )

        arquivo.write(
            f"Arquivo analisado: {caminho_csv.name}\n"
        )

        arquivo.write(
            f"Quantidade de registros: {len(dados)}\n"
        )

        arquivo.write(
            f"Quantidade de features: {len(features)}\n"
        )

        arquivo.write(
            f"Classe: {class_name}\n"
        )

        arquivo.write(
            f"Class ID: {dados[coluna_class_id].iloc[0]}\n"
        )

        # 1. Valores ausentes
        analisar_ausentes(
            dados,
            arquivo
        )

        # 2. Duplicatas
        analisar_duplicatas(
            dados,
            arquivo
        )

        # 3. Variáveis numéricas
        analisar_numericas(
            dados,
            numericas,
            arquivo
        )

        # 4. Variáveis categóricas
        analisar_categoricas(
            dados,
            categoricas,
            arquivo
        )

        # 5. Variáveis binárias
        analisar_binarias(
            dados,
            binarias,
            arquivo
        )

        # 6. Identificação da classe
        analisar_classe(
            dados,
            arquivo
        )

        # 7. Tipos de variáveis
        analisar_tipos(
            features,
            categoricas,
            binarias,
            numericas,
            arquivo
        )

        escrever_titulo(
            arquivo,
            "Conclusão geral"
        )

        arquivo.write(
            f"O conjunto analisado contém {len(dados)} registros "
            f"pertencentes à classe {class_name}.\n\n"
        )

        arquivo.write(
            "As features numéricas foram avaliadas por meio de "
            "medidas de tendência central, dispersão, assimetria, "
            "curtose e identificação de possíveis valores extremos.\n\n"
        )

        arquivo.write(
            "As variáveis categóricas e binárias foram avaliadas "
            "por suas distribuições de frequência.\n\n"
        )

        arquivo.write(
            "Os resultados devem ser interpretados considerando "
            "as características do conjunto de dados e o contexto "
            "de aplicação.\n"
        )

    print(
        f"Relatório gerado: {caminho_relatorio.name}"
    )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def gerar_relatorios():

    if not PASTA_ENTRADA.exists():

        raise FileNotFoundError(
            f"Pasta não encontrada: {PASTA_ENTRADA.resolve()}"
        )

    PASTA_RELATORIOS.mkdir(
        parents=True,
        exist_ok=True
    )

    arquivos_csv = sorted(
        PASTA_ENTRADA.glob("*.csv")
    )

    if not arquivos_csv:

        raise ValueError(
            "Nenhum arquivo CSV foi encontrado na pasta de entrada."
        )

    print("=" * 60)
    print("GERAÇÃO DE RELATÓRIOS POR CLASSE")
    print("=" * 60)

    print(f"Arquivos encontrados: {len(arquivos_csv)}\n")

    sucessos = 0
    erros = 0

    for caminho_csv in arquivos_csv:

        print(f"\nAnalisando: {caminho_csv.name}")

        try:

            gerar_relatorio(caminho_csv)
            sucessos += 1

        except Exception as erro:

            erros += 1

            print(
                f"ERRO ao analisar {caminho_csv.name}: {erro}"
            )

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)

    print(f"Relatórios gerados: {sucessos}")
    print(f"Arquivos com erro: {erros}")

    print(
        f"Pasta de relatórios: {PASTA_RELATORIOS.resolve()}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    try:

        gerar_relatorios()

    except Exception as erro:

        print("\nERRO DURANTE A EXECUÇÃO:")
        print(erro)
