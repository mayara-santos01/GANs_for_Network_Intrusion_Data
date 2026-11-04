import pandas as pd
from pathlib import Path
import re

# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = "svmsmote_merged.csv"
PASTA_SAIDA = "datasets_por_classe"

# ============================================================
# SEPARAÇÃO DOS DADOS
# ============================================================

def separar_classes():

    caminho = Path(ARQUIVO_ENTRADA)
    pasta_saida = Path(PASTA_SAIDA)

    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho.resolve()}"
        )

    pasta_saida.mkdir(parents=True, exist_ok=True)

    print("Lendo arquivo CSV...")

    dados = pd.read_csv(
        caminho,
        encoding="utf-8-sig"
    )

    if len(dados.columns) < 3:
        raise ValueError(
            "O CSV precisa conter features, class_id e class_name."
        )

    # Identifica as duas últimas colunas.
    coluna_class_id = dados.columns[-2]
    coluna_class_name = dados.columns[-1]

    print(f"Coluna de identificação: {coluna_class_id}")
    print(f"Coluna de classe: {coluna_class_name}")

    # Verifica valores ausentes nas classes.
    if dados[[coluna_class_id, coluna_class_name]].isna().any().any():
        raise ValueError(
            "Existem registros sem identificação de classe."
        )

    # Identifica as classes existentes.
    classes = dados[coluna_class_name].unique()

    print(f"\nClasses identificadas: {len(classes)}")

    total_registros = 0

    for classe in classes:

        # Filtra os registros da classe.
        dados_classe = dados[
            dados[coluna_class_name] == classe
        ].copy()

        # Evita caracteres inválidos no nome do arquivo.
        nome_arquivo = re.sub(
            r'[<>:"/\\|?*]',
            "_",
            str(classe)
        )

        caminho_saida = pasta_saida / f"{nome_arquivo}.csv"

        # Exporta o CSV.
        dados_classe.to_csv(
            caminho_saida,
            index=False,
            encoding="utf-8-sig"
        )

        quantidade = len(dados_classe)
        total_registros += quantidade

        class_id = dados_classe[coluna_class_id].iloc[0]

        print(
            f"Classe: {classe:<20} "
            f"ID: {class_id:<5} "
            f"Registros: {quantidade:<10} "
            f"Arquivo: {caminho_saida}"
        )

    print("\n" + "=" * 60)
    print("SEPARAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(f"Total de classes: {len(classes)}")
    print(f"Total de registros: {total_registros}")
    print(f"Pasta de saída: {pasta_saida.resolve()}")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    try:
        separar_classes()

    except Exception as erro:
        print("\nERRO DURANTE A SEPARAÇÃO:")
        print(erro)
