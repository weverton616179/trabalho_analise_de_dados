import polars as pl
from sklearn.decomposition import IncrementalPCA
import numpy as np
import csv

def pca_parquet(
    arquivo_parquet: str,
    arquivo_saida_csv: str,
    n_componentes: int,
    tamanho_bloco: int = 500  # número de linhas por bloco
):
    print("🔍 Abrindo arquivo...")
    scan = pl.scan_parquet(arquivo_parquet)

    # Descobrir nomes das colunas
    colunas = scan.columns
    coluna_seq = colunas[0]        # primeira coluna (string)
    colunas_numericas = colunas[1:]  # ignorar primeira

    print(f"📌 Colunas numéricas: {len(colunas_numericas)}")
    print("🚀 Iniciando Incremental PCA...")

    ipca = IncrementalPCA(n_components=n_componentes)

    # --------------------------
    # 1️⃣ PRIMEIRA PASSADA: PARTIAL_FIT
    # --------------------------
    print("⚙️ Passo 1: partial_fit")

    leitor = scan.select(colunas_numericas).collect(streaming=True)

    # Ler o arquivo em blocos
    for i in range(0, leitor.height, tamanho_bloco):
        bloco = leitor.slice(i, tamanho_bloco)
        X = bloco.to_numpy()

        ipca.partial_fit(X)
        print(f"  - Treinando bloco {i // tamanho_bloco + 1}")

    # --------------------------
    # 2️⃣ SEGUNDA PASSADA: TRANSFORM
    # --------------------------
    print("⚙️ Passo 2: transform (gerando CSV)")

    # Abrir arquivo CSV para saída
    with open(arquivo_saida_csv, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Escrever header
        header = ["sequencia"] + [f"PC{i+1}" for i in range(n_componentes)]
        writer.writerow(header)

        # Ler novamente o parquet, mas com a sequência
        leitor2 = scan.select(colunas).collect(streaming=True)

        for i in range(0, leitor2.height, tamanho_bloco):
            bloco = leitor2.slice(i, tamanho_bloco)

            sequencias = bloco[coluna_seq].to_list()
            X = bloco[colunas_numericas].to_numpy()

            pcs = ipca.transform(X)

            # Combinar sequencias + PCs
            for seq, linha_pc in zip(sequencias, pcs):
                writer.writerow([seq] + list(linha_pc))

            print(f"  - Transformando bloco {i // tamanho_bloco + 1}")

    print("🎉 FINALIZADO! Arquivo salvo em:", arquivo_saida_csv)
