from Bio import SeqIO
from itertools import product
import numpy as np
import csv
import multiprocessing as mp
import os
from sklearn.decomposition import PCA
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"


def kmer_2x2_skip(seq, skip, nprt):
    seq = str(seq).upper()
    n = len(seq)
    kmers = []
    for i in range(n - ((nprt * 2) + skip) + 1):
        pair1 = seq[i:i + nprt]
        pair2 = seq[i + nprt + skip:i + (nprt * 2) + skip]
        
        separador = "_" * skip
        kmers.append(f"{pair1}{separador}{pair2}")
    return kmers


def gerar_kmers_possiveis(skip, nprt):
    separador = "_" * skip
    pares = [''.join(p) for p in product(AA, repeat=nprt)]
    return [f"{p1}{separador}{p2}" for p1 in pares for p2 in pares]


def dividir_lista(lista, n):
    tamanho = len(lista)
    k, m = divmod(tamanho, n)
    return [lista[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def processar_bloco(registros, skip, nprt, todos_kmers, kmer_index):
    linhas = []
    i = 0
    for reg in registros:
        i += 1
        print(f"{i}")
        vetor = np.zeros(len(todos_kmers), dtype=np.uint8)
        for k in set(kmer_2x2_skip(reg.seq, skip, nprt)):
            idx = kmer_index.get(k)
            if idx is not None:
                vetor[idx] = 1
        linhas.append([reg.id] + vetor.tolist())
    return linhas


def extrair_kmers_binario_parallel(fasta_path, skip, nprt, saida_csv, n_processos, range_pca):
    todos_kmers = gerar_kmers_possiveis(skip, nprt)
    kmer_index = {k: i for i, k in enumerate(todos_kmers)}
    print(kmer_index)

    registros = list(SeqIO.parse(fasta_path, "fasta"))
    blocos = dividir_lista(registros, n_processos)

    print(f"💡 Processando {len(registros)} sequências em {n_processos} núcleos...")

    with mp.Pool(n_processos) as pool: #roda com multiprocessing
        resultados = pool.starmap(processar_bloco, [(b, skip, nprt, todos_kmers, kmer_index) for b in blocos])

    # Junta todos os blocos em uma única lista
    todas_linhas = [linha for bloco in resultados for linha in bloco]

    # Salva a matriz binária
    with open(saida_csv, "w", newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["sequencia"] + todos_kmers)
        writer.writerows(todas_linhas)

    print(f"✅ Arquivo salvo em: {saida_csv}")

    # === PCA ===
    print(f"⚙️ Iniciando projeção PCA ({range_pca} componentes)...")
    df = pd.DataFrame(todas_linhas, columns=["sequencia"] + todos_kmers)

    X = df.drop(columns=["sequencia"]).values
    pca = PCA(n_components=range_pca)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(range_pca)])
    df_pca.insert(0, "sequencia", df["sequencia"])

    pca_saida = saida_csv.replace(".csv", f"_PCA{range_pca}.csv")
    df_pca.to_csv(pca_saida, index=False, sep=';')

    print(f"✅ PCA concluído e salvo em: {pca_saida}")
    print(f"📊 Variância explicada total: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    return pca_saida


# === Exemplo de uso ===
if __name__ == "__main__":
    caminho = "proteinas.fa"
    skip = 2
    nprt = 2
    range_pca = 50
    saida_csv = f"kmer_2x2_skip{skip}_binario.csv"
    n_processos = 3
    
    extrair_kmers_binario_parallel(caminho, skip, nprt, saida_csv, n_processos, range_pca)
