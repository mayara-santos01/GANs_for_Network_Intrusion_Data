import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_CSV = "svmsmote_merged.csv"
ARQUIVO_RELATORIO = "relatorio_qualidade.txt"

# Variáveis categóricas
VARIAVEIS_CATEGORICAS = [
    "proto",
    "service",
    "state"
]

# Variáveis binárias
VARIAVEIS_BINARIAS = [
    "is_ftp_login",
    "is_sm_ips_ports"
]

# Variáveis que representam classes
VARIAVEIS_CLASSE = [
    "class_id",
    "class_name"
]

# Limite para exibir categorias em tabelas
LIMITE_CATEGORIAS = 20


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def escrever_titulo(arquivo, titulo):
    arquivo.write("\n")
    arquivo.write("=" * 75 + "\n")
    arquivo.write(titulo.upper() + "\n")
    arquivo.write("=" * 75 + "\n\n")


def formatar_numero(valor, casas=4):
    if pd.isna(valor):
        return "não disponível"

    return f"{valor:.{casas}f}"


def formatar_percentual(valor):
    if pd.isna(valor):
        return "não disponível"

    return f"{valor:.2f}%"


def classificar_assimetria(valor):
    if pd.isna(valor):
        return "Não foi possível determinar a assimetria."

    if valor > 1:
        return (
            "A distribuição apresenta forte assimetria positiva, "
            "com uma cauda alongada à direita."
        )

    elif valor > 0.5:
        return (
            "A distribuição apresenta assimetria positiva moderada, "
            "indicando maior concentração de valores baixos e uma "
            "cauda à direita."
        )

    elif valor < -1:
        return (
            "A distribuição apresenta forte assimetria negativa, "
            "com uma cauda alongada à esquerda."
        )

    elif valor < -0.5:
        return (
            "A distribuição apresenta assimetria negativa moderada, "
            "indicando maior concentração de valores altos e uma "
            "cauda à esquerda."
        )

    else:
        return (
            "A distribuição apresenta assimetria reduzida, "
            "sendo relativamente simétrica."
        )


def interpretar_curtose(valor):
    if pd.isna(valor):
        return "Não foi possível avaliar a curtose."

    if valor > 1:
        return (
            "A distribuição apresenta curtose elevada, sugerindo "
            "caudas mais pesadas e possível presença de valores extremos."
        )

    elif valor < -1:
        return (
            "A distribuição apresenta curtose baixa, indicando "
            "caudas relativamente leves e menor concentração de "
            "valores extremos."
        )

    else:
        return (
            "A curtose não indica um afastamento acentuado em relação "
            "à referência de uma distribuição normal."
        )


def interpretar_dispersao(cv, desvio, media):
    if abs(media) < 1e-10:
        return (
            "O coeficiente de variação não é adequado para esta variável, "
            "pois a média é próxima de zero."
        )

    if pd.isna(cv):
        return "Não foi possível avaliar o coeficiente de variação."

    if cv < 0.15:
        return (
            "Os dados apresentam baixa dispersão relativa, indicando "
            "valores relativamente homogêneos."
        )

    elif cv < 0.30:
        return (
            "Os dados apresentam dispersão relativa moderada, "
            "com alguma variabilidade entre as observações."
        )

    else:
        return (
            "Os dados apresentam elevada dispersão relativa, "
            "indicando grande variabilidade entre as observações."
        )


# ============================================================
# LEITURA DO CSV COM FEATURES SEPARADAS
# ============================================================

def carregar_dados(caminho):
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo CSV não encontrado: {caminho.resolve()}"
        )

    # Lê o CSV preservando os valores originais como texto.
    dados = pd.read_csv(
        caminho,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    # Verifica a existência das colunas de classificação.
    colunas_obrigatorias = ["class_id", "class_name"]

    for coluna in colunas_obrigatorias:
        if coluna not in dados.columns:
            raise ValueError(
                f"A coluna '{coluna}' não foi encontrada no CSV."
            )

    # Identifica automaticamente as features.
    nomes_atributos = [
        coluna for coluna in dados.columns
        if coluna not in VARIAVEIS_CLASSE
    ]

    if not nomes_atributos:
        raise ValueError(
            "Nenhuma variável de entrada foi encontrada no CSV."
        )

    # Verifica se existem registros.
    if dados.empty:
        raise ValueError("O arquivo CSV não contém registros.")

    # Retorna os dados e os nomes das features.
    return dados, nomes_atributos


# ============================================================
# IDENTIFICAÇÃO DOS TIPOS DE VARIÁVEIS
# ============================================================

def identificar_tipos(dados, nomes_atributos):

    categoricas = [
        coluna for coluna in VARIAVEIS_CATEGORICAS
        if coluna in nomes_atributos
    ]

    binarias = [
        coluna for coluna in VARIAVEIS_BINARIAS
        if coluna in nomes_atributos
    ]

    numericas = [
        coluna for coluna in nomes_atributos
        if coluna not in categoricas
        and coluna not in binarias
    ]

    # Confere quais variáveis numéricas realmente podem ser convertidas.
    numericas_validas = []

    for coluna in numericas:

        convertida = pd.to_numeric(
            dados[coluna],
            errors="coerce"
        )

        if convertida.notna().any():
            numericas_validas.append(coluna)

    return categoricas, binarias, numericas_validas


# ============================================================
# ANÁLISE DE VALORES AUSENTES
# ============================================================

def analisar_valores_ausentes(dados, arquivo):

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

    arquivo.write(
        f"Total de células analisadas: {total}\n"
    )

    arquivo.write(
        f"Total de valores ausentes ou vazios: {total_ausentes}\n"
    )

    arquivo.write(
        f"Percentual de valores ausentes ou vazios: "
        f"{percentual:.2f}%\n\n"
    )

    if total_ausentes == 0:
        arquivo.write(
            "Interpretação: Não foram identificados valores ausentes "
            "ou células vazias no conjunto de dados.\n"
        )
    else:
        arquivo.write(
            "Interpretação: Foram encontrados valores ausentes ou vazios. "
            "Esses registros devem ser avaliados antes da aplicação "
            "de algoritmos de aprendizado de máquina.\n\n"
        )

        arquivo.write("Variáveis com valores ausentes ou vazios:\n")

        for coluna in dados.columns:

            quantidade = (
                int(dados[coluna].isna().sum())
                + int((dados[coluna] == "").sum())
            )

            if quantidade > 0:
                percentual_coluna = (
                    quantidade / len(dados) * 100
                )

                arquivo.write(
                    f"- {coluna}: {quantidade} "
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

    arquivo.write(
        f"Registros duplicados: {duplicatas}\n"
    )

    arquivo.write(
        f"Percentual de duplicatas: {percentual:.2f}%\n\n"
    )

    if duplicatas == 0:
        arquivo.write(
            "Interpretação: Não foram encontrados registros "
            "completamente duplicados.\n"
        )
    else:
        arquivo.write(
            "Interpretação: Foram identificados registros duplicados. "
            "É recomendável verificar se representam repetições "
            "indevidas ou ocorrências legítimas do mesmo padrão.\n"
        )

        arquivo.write(
            "Observação: em conjuntos de dados utilizados para "
            "balanceamento ou geração de amostras, algumas repetições "
            "podem ser esperadas e não representam necessariamente erros.\n"
        )


# ============================================================
# ANÁLISE DAS VARIÁVEIS NUMÉRICAS
# ============================================================

def analisar_variavel_numerica(dados, coluna, arquivo):

    valores = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    ).dropna()

    if len(valores) == 0:
        arquivo.write(
            f"\nVariável: {coluna}\n"
            "Não há valores numéricos válidos para análise.\n"
        )
        return

    media = valores.mean()
    mediana = valores.median()
    desvio = valores.std()
    minimo = valores.min()
    maximo = valores.max()

    q1 = valores.quantile(0.25)
    q3 = valores.quantile(0.75)

    iqr = q3 - q1

    # Coeficiente de variação.
    cv = (
        desvio / abs(media)
        if abs(media) > 1e-10 else np.nan
    )

    # Assimetria e curtose.
    assimetria = valores.skew()
    curtose = valores.kurtosis()

    # Identificação estatística de possíveis valores extremos.
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

    # Frequência de zeros.
    quantidade_zeros = int((valores == 0).sum())

    percentual_zeros = (
        quantidade_zeros / len(valores) * 100
    )

    # Quantidade de valores distintos.
    valores_unicos = valores.nunique()

    # Identifica variáveis constantes.
    constante = valores_unicos <= 1

    arquivo.write("\n")
    arquivo.write("-" * 75 + "\n")
    arquivo.write(f"VARIÁVEL: {coluna}\n")
    arquivo.write("-" * 75 + "\n\n")

    arquivo.write(
        f"Quantidade de observações válidas: {len(valores)}\n"
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
        f"Valor mínimo: {formatar_numero(minimo)}\n"
    )

    arquivo.write(
        f"Valor máximo: {formatar_numero(maximo)}\n"
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
        f"Possíveis valores extremos: {quantidade_outliers} "
        f"({percentual_outliers:.2f}%)\n"
    )

    arquivo.write(
        f"Valores iguais a zero: {quantidade_zeros} "
        f"({percentual_zeros:.2f}%)\n"
    )

    arquivo.write(
        f"Valores distintos: {valores_unicos}\n\n"
    )

    # --------------------------------------------------------
    # INTERPRETAÇÃO
    # --------------------------------------------------------

    arquivo.write("INTERPRETAÇÃO ESTATÍSTICA:\n\n")

    if constante:
        arquivo.write(
            "A variável apresenta apenas um valor distinto. "
            "Portanto, não possui variabilidade estatística e "
            "pode ter pouca utilidade para diferenciar observações.\n"
        )
        return

    # Interpretação da concentração dos dados.
    if media > mediana:
        arquivo.write(
            "A média é superior à mediana, o que pode indicar "
            "influência de valores elevados sobre a média.\n"
        )

    elif media < mediana:
        arquivo.write(
            "A média é inferior à mediana, o que pode indicar "
            "influência de valores baixos sobre a média.\n"
        )

    else:
        arquivo.write(
            "A média e a mediana apresentam valores próximos, "
            "indicando equilíbrio entre essas medidas de tendência central.\n"
        )

    # Interpretação da assimetria.
    arquivo.write(
        classificar_assimetria(assimetria) + "\n"
    )

    # Interpretação da curtose.
    arquivo.write(
        interpretar_curtose(curtose) + "\n"
    )

    # Interpretação da dispersão.
    arquivo.write(
        interpretar_dispersao(cv, desvio, media) + "\n"
    )

    # Interpretação dos valores extremos.
    if percentual_outliers > 10:
        arquivo.write(
            "Foi identificada uma proporção elevada de possíveis "
            "valores extremos. Isso pode indicar uma distribuição "
            "com caudas pesadas, assimetria acentuada ou observações "
            "que merecem investigação.\n"
        )

    elif percentual_outliers > 0:
        arquivo.write(
            "Foram identificados possíveis valores extremos. "
            "Essas observações devem ser investigadas para verificar "
            "se representam comportamentos legítimos ou problemas "
            "de qualidade dos dados.\n"
        )

    else:
        arquivo.write(
            "Não foram identificados valores extremos pelo critério "
            "do intervalo interquartil.\n"
        )

    # Interpretação dos zeros.
    if percentual_zeros > 50:
        arquivo.write(
            "Mais da metade das observações apresenta valor zero. "
            "É importante verificar se esses zeros representam "
            "ausência real de ocorrência ou alguma forma de codificação "
            "dos dados.\n"
        )

    elif percentual_zeros > 0:
        arquivo.write(
            "A variável apresenta ocorrências iguais a zero. "
            "A interpretação desses valores depende do significado "
            "do atributo no domínio analisado.\n"
        )

    # Recomendações.
    arquivo.write("\nRECOMENDAÇÕES:\n")

    if percentual_outliers > 0:
        arquivo.write(
            "- Investigar os possíveis valores extremos antes de "
            "considerar sua remoção ou transformação.\n"
        )

    if percentual_zeros > 50:
        arquivo.write(
            "- Avaliar a elevada frequência de zeros e sua relação "
            "com o significado da variável.\n"
        )

    if not pd.isna(assimetria) and abs(assimetria) > 1:
        arquivo.write(
            "- Considerar transformações adequadas, como log1p, "
            "quando compatíveis com a natureza dos dados.\n"
        )

    if not pd.isna(cv) and cv >= 0.30:
        arquivo.write(
            "- Avaliar a necessidade de normalização ou padronização "
            "dependendo do algoritmo de aprendizado de máquina utilizado.\n"
        )

    arquivo.write(
        "- Não assumir normalidade apenas com base nas medidas "
        "descritivas; considerar o formato da distribuição e o "
        "objetivo da análise.\n"
    )


def analisar_numericas(dados, numericas, arquivo):

    escrever_titulo(
        arquivo,
        "3. Análise estatística interpretativa das variáveis numéricas"
    )

    arquivo.write(
        "As variáveis numéricas são avaliadas considerando medidas "
        "de tendência central, dispersão, assimetria, curtose, "
        "frequência de zeros e possíveis valores extremos.\n"
    )

    arquivo.write(
        "As interpretações são indicativas e devem ser consideradas "
        "em conjunto com o significado de cada atributo.\n"
    )

    for coluna in numericas:
        analisar_variavel_numerica(
            dados,
            coluna,
            arquivo
        )


# ============================================================
# ANÁLISE DE VARIÁVEIS CATEGÓRICAS
# ============================================================

def analisar_categorica(dados, coluna, arquivo):

    escrever_titulo(
        arquivo,
        f"Análise da variável categórica: {coluna}"
    )

    valores = dados[coluna].replace("", np.nan).dropna()

    if len(valores) == 0:
        arquivo.write(
            "Não existem valores válidos para esta variável.\n"
        )
        return

    frequencias = valores.value_counts()
    percentuais = valores.value_counts(normalize=True) * 100

    arquivo.write(
        f"Quantidade de observações válidas: {len(valores)}\n"
    )

    arquivo.write(
        f"Quantidade de categorias distintas: {valores.nunique()}\n\n"
    )

    arquivo.write("DISTRIBUIÇÃO DE FREQUÊNCIAS:\n\n")

    arquivo.write(
        f"{'Categoria':<30} {'Quantidade':>15} {'Percentual':>15}\n"
    )

    arquivo.write("-" * 65 + "\n")

    for categoria, quantidade in frequencias.head(
        LIMITE_CATEGORIAS
    ).items():

        percentual = percentuais[categoria]

        arquivo.write(
            f"{str(categoria):<30.30} "
            f"{quantidade:>15} "
            f"{percentual:>14.2f}%\n"
        )

    if len(frequencias) > LIMITE_CATEGORIAS:
        arquivo.write(
            f"\nOutras {len(frequencias) - LIMITE_CATEGORIAS} "
            "categorias não foram exibidas por limite de apresentação.\n"
        )

    categoria_mais_frequente = frequencias.index[0]
    percentual_dominante = percentuais.iloc[0]

    arquivo.write("\nINTERPRETAÇÃO:\n\n")

    arquivo.write(
        f"A categoria mais frequente é '{categoria_mais_frequente}', "
        f"representando {percentual_dominante:.2f}% das observações.\n"
    )

    if percentual_dominante > 80:
        arquivo.write(
            "A distribuição apresenta forte concentração em uma única "
            "categoria, indicando possível desequilíbrio entre os valores "
            "dessa variável.\n"
        )

    elif percentual_dominante > 50:
        arquivo.write(
            "A distribuição apresenta concentração moderada em uma "
            "categoria predominante.\n"
        )

    else:
        arquivo.write(
            "Não há uma única categoria concentrando mais da metade "
            "das observações, indicando uma distribuição relativamente "
            "mais diversificada.\n"
        )


def analisar_categoricas(dados, categoricas, arquivo):

    escrever_titulo(
        arquivo,
        "4. Distribuição das variáveis categóricas"
    )

    for coluna in categoricas:
        analisar_categorica(
            dados,
            coluna,
            arquivo
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

        escrever_titulo(
            arquivo,
            f"Variável binária: {coluna}"
        )

        valores = dados[coluna].replace("", np.nan).dropna()

        if len(valores) == 0:
            arquivo.write(
                "Não existem valores válidos para esta variável.\n"
            )
            continue

        frequencias = valores.value_counts()
        percentuais = valores.value_counts(normalize=True) * 100

        arquivo.write(
            f"Quantidade de observações válidas: {len(valores)}\n"
        )

        arquivo.write(
            f"Quantidade de valores distintos: {valores.nunique()}\n\n"
        )

        arquivo.write("DISTRIBUIÇÃO:\n\n")

        for categoria, quantidade in frequencias.items():

            arquivo.write(
                f"- Valor {categoria}: {quantidade} ocorrências "
                f"({percentuais[categoria]:.2f}%)\n"
            )

        arquivo.write("\nINTERPRETAÇÃO:\n\n")

        if valores.nunique() == 1:
            arquivo.write(
                "A variável apresenta apenas um valor em todas as "
                "observações. Portanto, não possui capacidade de "
                "diferenciação neste conjunto de dados.\n"
            )

        elif percentuais.iloc[0] > 95:
            arquivo.write(
                "A variável apresenta forte predominância de um dos "
                "valores, indicando distribuição bastante desequilibrada.\n"
            )

        else:
            arquivo.write(
                "A variável apresenta os dois valores possíveis, "
                "permitindo distinguir diferentes condições ou ocorrências.\n"
            )


# ============================================================
# ANÁLISE DAS CLASSES
# ============================================================

def analisar_classes(dados, arquivo):

    escrever_titulo(
        arquivo,
        "6. Distribuição das classes"
    )

    for coluna in VARIAVEIS_CLASSE:

        if coluna not in dados.columns:
            continue

        escrever_titulo(
            arquivo,
            f"Distribuição de {coluna}"
        )

        valores = dados[coluna].replace("", np.nan).dropna()

        if len(valores) == 0:
            arquivo.write(
                "Não existem valores válidos para esta variável.\n"
            )
            continue

        frequencias = valores.value_counts()
        percentuais = valores.value_counts(normalize=True) * 100

        arquivo.write(
            f"Quantidade de observações: {len(valores)}\n"
        )

        arquivo.write(
            f"Quantidade de classes distintas: {valores.nunique()}\n\n"
        )

        arquivo.write(
            f"{'Classe':<30} {'Quantidade':>15} {'Percentual':>15}\n"
        )

        arquivo.write("-" * 65 + "\n")

        for categoria, quantidade in frequencias.items():

            arquivo.write(
                f"{str(categoria):<30.30} "
                f"{quantidade:>15} "
                f"{percentuais[categoria]:>14.2f}%\n"
            )

        maior = percentuais.max()
        menor = percentuais.min()

        arquivo.write("\nINTERPRETAÇÃO:\n\n")

        if maior > 80:
            arquivo.write(
                "A distribuição apresenta forte predominância de uma "
                "classe. Esse comportamento indica possível desequilíbrio "
                "entre as classes, o que pode influenciar o treinamento "
                "e a avaliação dos modelos de classificação.\n"
            )

        elif maior - menor < 10:
            arquivo.write(
                "As classes apresentam proporções relativamente próximas, "
                "indicando uma distribuição equilibrada.\n"
            )

        else:
            arquivo.write(
                "As classes apresentam diferenças de frequência. "
                "É recomendável avaliar o impacto dessa distribuição "
                "na construção e avaliação dos modelos.\n"
            )


# ============================================================
# ANÁLISE DO TIPO DAS VARIÁVEIS
# ============================================================

def analisar_tipos(
    dados,
    nomes_atributos,
    categoricas,
    binarias,
    numericas,
    arquivo
):

    escrever_titulo(
        arquivo,
        "7. Classificação dos tipos de variáveis"
    )

    total = len(nomes_atributos)

    if total == 0:
        arquivo.write(
            "Não foram identificados atributos.\n"
        )
        return

    quantidade_categoricas = len(categoricas)
    quantidade_binarias = len(binarias)
    quantidade_numericas = len(numericas)

    arquivo.write(
        f"Total de atributos analisados: {total}\n\n"
    )

    grupos = [
        ("Numéricas", quantidade_numericas),
        ("Categóricas", quantidade_categoricas),
        ("Binárias", quantidade_binarias)
    ]

    for nome, quantidade in grupos:

        percentual = quantidade / total * 100

        arquivo.write(
            f"{nome}: {quantidade} atributos "
            f"({percentual:.2f}%)\n"
        )

    arquivo.write("\nINTERPRETAÇÃO:\n\n")

    arquivo.write(
        "A classificação dos atributos permite compreender a composição "
        "do conjunto de dados e orientar a escolha dos métodos de "
        "pré-processamento.\n"
    )

    arquivo.write(
        "Variáveis numéricas podem exigir normalização ou padronização, "
        "dependendo do algoritmo utilizado. Variáveis categóricas "
        "podem exigir técnicas de codificação, enquanto variáveis "
        "binárias devem ter seus valores interpretados conforme "
        "o significado de cada atributo.\n"
    )


# ============================================================
# ANÁLISE ESPECÍFICA DA ÚLTIMA VARIÁVEL
# ============================================================

def analisar_ultima_variavel(dados, nomes_atributos, arquivo):

    escrever_titulo(
        arquivo,
        "8. Análise específica da última variável"
    )

    ultima_variavel = nomes_atributos[-1]

    arquivo.write(
        f"Última variável identificada: {ultima_variavel}\n\n"
    )

    arquivo.write(
        "DISTRIBUIÇÃO DOS VALORES:\n\n"
    )

    valores = dados[ultima_variavel].replace("", np.nan).dropna()

    if len(valores) == 0:
        arquivo.write(
            "Não existem valores válidos para análise.\n"
        )
        return

    frequencias = valores.value_counts()
    percentuais = valores.value_counts(normalize=True) * 100

    arquivo.write(
        f"{'Valor':<25} {'Quantidade':>15} {'Percentual':>15}\n"
    )

    arquivo.write("-" * 55 + "\n")

    for valor, quantidade in frequencias.items():

        arquivo.write(
            f"{str(valor):<25.25} "
            f"{quantidade:>15} "
            f"{percentuais[valor]:>14.2f}%\n"
        )

    arquivo.write("\nINTERPRETAÇÃO:\n\n")

    if valores.nunique() == 2:

        arquivo.write(
            f"A variável {ultima_variavel} apresenta dois valores "
            "distintos. Sua distribuição permite identificar a "
            "frequência relativa de cada condição representada.\n"
        )

        if percentuais.max() > 95:
            arquivo.write(
                "Observa-se forte predominância de um dos valores, "
                "indicando possível desequilíbrio na distribuição "
                "dessa variável.\n"
            )

        else:
            arquivo.write(
                "Os dois valores apresentam participação relevante "
                "na distribuição da variável.\n"
            )

    else:
        arquivo.write(
            f"A variável {ultima_variavel} apresenta "
            f"{valores.nunique()} valores distintos. "
            "Recomenda-se verificar sua natureza e o significado "
            "de cada categoria antes de definir o tratamento adequado.\n"
        )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def gerar_relatorio():

    print("Iniciando análise da qualidade dos dados...")

    dados, nomes_atributos = carregar_dados(
        ARQUIVO_CSV
    )

    categoricas, binarias, numericas = identificar_tipos(
        dados,
        nomes_atributos
    )

    with open(
        ARQUIVO_RELATORIO,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "RELATÓRIO DE ANÁLISE DA QUALIDADE DOS DADOS\n"
        )

        arquivo.write(
            f"Data da análise: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
        )

        arquivo.write(
            f"Arquivo analisado: {ARQUIVO_CSV}\n"
        )

        arquivo.write(
            f"Quantidade de registros: {len(dados)}\n"
        )

        arquivo.write(
            f"Quantidade de atributos: {len(nomes_atributos)}\n"
        )

        arquivo.write(
            "\nEste relatório apresenta uma avaliação da qualidade "
            "dos dados, incluindo valores ausentes, duplicatas, "
            "distribuições, características estatísticas e "
            "interpretações das variáveis.\n"
        )

        # 1. Valores ausentes
        analisar_valores_ausentes(
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

        # 6. Classes
        analisar_classes(
            dados,
            arquivo
        )

        # 7. Tipos de variáveis
        analisar_tipos(
            dados,
            nomes_atributos,
            categoricas,
            binarias,
            numericas,
            arquivo
        )

        # 8. Última variável
        analisar_ultima_variavel(
            dados,
            nomes_atributos,
            arquivo
        )

        escrever_titulo(
            arquivo,
            "Conclusão geral"
        )

        arquivo.write(
            "A análise realizada permite identificar características "
            "estatísticas e possíveis problemas de qualidade no "
            "conjunto de dados.\n\n"
        )

        arquivo.write(
            "As variáveis numéricas foram avaliadas por meio de "
            "interpretações de tendência central, dispersão, "
            "assimetria, curtose e possíveis valores extremos. "
            "As variáveis categóricas e binárias foram avaliadas "
            "por suas distribuições de frequência.\n\n"
        )

        arquivo.write(
            "As recomendações apresentadas devem ser consideradas "
            "em conjunto com o contexto dos dados e os objetivos "
            "da aplicação. A identificação de valores extremos, "
            "por exemplo, não significa necessariamente que esses "
            "valores sejam erros ou devam ser removidos.\n"
        )

    print("\nAnálise concluída!")
    print(f"Relatório salvo em: {Path(ARQUIVO_RELATORIO).resolve()}")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    try:
        gerar_relatorio()

    except Exception as erro:
        print("\nERRO DURANTE A ANÁLISE:")
        print(erro)
