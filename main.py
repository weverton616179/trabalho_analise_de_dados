from Bio import SeqIO
from itertools import product
import numpy as np
import csv
import multiprocessing as mp
import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import IncrementalPCA
import duckdb

AA = "ACDEFGHIKLMNPQRSTVWY"

def pca_csv(input_csv, output_csv="pca_output.csv", n_components=50, buffer_size=2000):
    print(">>> ULTRA MODE: PCA com mínimo uso de memória")

    # ====== 1) Ler apenas o header ======
    with open(input_csv, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
    
    # Descobrir quais colunas são numéricas pelo próprio CSV
    # (assume que TODAS são numéricas — se não forem, passe a lista no argumento)
    numeric_idx = list(range(len(header)))

    print(f">>> Detectadas {len(numeric_idx)} colunas numéricas")

    # ====== 2) Treinar o IncrementalPCA com duas passagens ======
    ipca = IncrementalPCA(n_components=n_components)

    ## ---- PRIMEIRA PASSAGEM: partial_fit ----
    print(">>> Passo 1: Treinando PCA incremental...")
    buffer = []

    with open(input_csv, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # pular header

        for i, row in enumerate(reader):
            row = [float(row[c]) for c in numeric_idx]
            buffer.append(row)

            if len(buffer) == buffer_size:
                X = np.asarray(buffer, dtype=np.float32)
                ipca.partial_fit(X)
                buffer.clear()

            if i % 200000 == 0:
                print(f"  Treinando... {i} linhas")
                gc.collect()

    # processar buffer restante
    if buffer:
        ipca.partial_fit(np.asarray(buffer, dtype=np.float32))
        buffer.clear()

    gc.collect()

    # ====== 3) SEGUNDA PASSAGEM: Transformar e salvar ======
    print(">>> Passo 2: Gerando projeção PCA...")

    with open(output_csv, 'w') as out:
        out.write(",".join([f"PC{i+1}" for i in range(n_components)]) + "\n")

    buffer = []

    with open(input_csv, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)

        for i, row in enumerate(reader):
            row = [float(row[c]) for c in numeric_idx]
            buffer.append(row)

            if len(buffer) == buffer_size:
                X = np.asarray(buffer, dtype=np.float32)
                PCs = ipca.transform(X)

                with open(output_csv, 'a') as out:
                    np.savetxt(out, PCs, delimiter=",", fmt="%.8e")

                buffer.clear()

            if i % 200000 == 0:
                print(f"  Transformando... {i} linhas")
                gc.collect()

    # buffer final
    if buffer:
        X = np.asarray(buffer, dtype=np.float32)
        PCs = ipca.transform(X)
        with open(output_csv, 'a') as out:
            np.savetxt(out, PCs, delimiter=",", fmt="%.8e")

    print(">>> Concluído com uso mínimo de RAM!")

def kmer_2x2_skip(seq, skip, nprt): #retorna toda as combinações que a sequencia tem
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


# def processar_bloco(index, registros, skip, nprt, todos_kmers, kmer_index):
#     print(f"🔄 Iniciando processamento do bloco {index + 1}")
#     arquivo = f"k{index + 1}.csv"
#     i = 0
#     g = 0
#     with open(arquivo, "w", newline='') as f:
#         writer = csv.writer(f, delimiter=';')
#         writer.writerow(["sequencia"] + todos_kmers)
#         for reg in registros:
#             i += 1
#             g += 1
#             if g == 50:
#                 g = 0
#                 print(f"Bloco {index + 1} - Sequência {i}")
                
#             vetor = np.zeros(len(todos_kmers), dtype=np.uint8)
#             for k in set(kmer_2x2_skip(reg.seq, skip, nprt)):
#                 idx = kmer_index.get(k)
#                 if idx is not None:
#                     vetor[idx] = 1
            
#             writer.writerow([reg.id] + vetor.tolist())
#         return arquivo

def processar_bloco(index, registros, skip, nprt, todos_kmers, kmer_index, chunk_size=3000):
    print(f"🔄 Iniciando processamento do bloco {index + 1}")

    arquivo = f"k{index + 1}.parquet"

    i = 0
    g = 0

    schema = pa.schema(
        [("sequencia", pa.string())] +
        [(k, pa.uint8()) for k in todos_kmers]
    )

    writer = pq.ParquetWriter(arquivo, schema, compression="zstd")
    buffer_ids = []
    buffer_matrizes = []
    
    for reg in registros:
        i += 1
        g += 1
        if g == 200:
            g = 0
            print(f"Bloco {index + 1} - Sequência {i}")

        vetor = np.zeros(len(todos_kmers), dtype=np.uint8)
        for k in set(kmer_2x2_skip(reg.seq, skip, nprt)):
            idx = kmer_index.get(k)
            if idx is not None:
                vetor[idx] = 1
                
        buffer_ids.append(reg.id)
        buffer_matrizes.append(vetor)
        
        if len(buffer_ids) >= chunk_size:

            arrays = {"sequencia": pa.array(buffer_ids, type=pa.string())}

            matriz_np = np.vstack(buffer_matrizes)
            for col_idx, nome_kmer in enumerate(todos_kmers):
                arrays[nome_kmer] = pa.array(matriz_np[:, col_idx])

            table = pa.Table.from_arrays(
                [arrays[col] for col in schema.names],
                schema=schema
            )

            writer.write_table(table)
            buffer_ids.clear()
            buffer_matrizes.clear()

    if buffer_ids:
        arrays = {"sequencia": pa.array(buffer_ids, type=pa.string())}

        matriz_np = np.vstack(buffer_matrizes)
        for col_idx, nome_kmer in enumerate(todos_kmers):
            arrays[nome_kmer] = pa.array(matriz_np[:, col_idx])

        table = pa.Table.from_arrays(
            [arrays[col] for col in schema.names],
            schema=schema
        )
        writer.write_table(table)
        buffer_ids.clear()
        buffer_matrizes.clear()

    writer.close()
    return arquivo

def unificar_arquivos(arquivos, output):
    dfs = []
    print("🔗 Unificando arquivos CSV...")
    for arquivo in arquivos:
        dfs.append(pd.read_csv(arquivo))

    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv(output, index=False)

def unir_parquets(arquivos, arquivo_saida):
    tables = []
    print("Iniciando uniao tabelas...")
    for arq in arquivos:
        print(f"Lendo metadados de: {arq}")
        # Lê sem carregar tudo — apenas os row groups (muito rápido)
        pq_file = pq.ParquetFile(arq)
        tables.append(pq_file.read())  # Apenas 4.000 linhas = rápido

    print("Concatenando tabelas...")
    final_table = pa.concat_tables(tables, promote=True)

    print("Salvando parquet final...")
    pq.write_table(final_table, arquivo_saida, compression="zstd")

    print("Pronto:", arquivo_saida)

def extrair_kmers_binario_parallel(fasta_path, skip, nprt, saida_csv, n_processos, range_pca):
    todos_kmers = gerar_kmers_possiveis(skip, nprt)
    kmer_index = {k: i for i, k in enumerate(todos_kmers)}
    print(kmer_index)

    registros = list(SeqIO.parse(fasta_path, "fasta"))
    blocos = dividir_lista(registros, n_processos)

    print(f"💡 Processando {len(registros)} sequências em {n_processos} núcleos...")
    del registros 
    
    with mp.Pool(n_processos) as pool: #roda com multiprocessing
        resultados = pool.starmap(processar_bloco, [(index, b, skip, nprt, todos_kmers, kmer_index) for index, b in enumerate(blocos)])

    print(resultados)
    unir_parquets(resultados, saida_csv)
    pca_csv(saida_csv, n_components=range_pca)
    return
    # Junta todos os blocos em uma única lista
    # todas_linhas = [linha for bloco in resultados for linha in bloco]

    # Salva a matriz binária
    # with open(saida_csv, "w", newline='') as f:
    #     writer = csv.writer(f, delimiter=';')
    #     writer.writerow(["sequencia"] + todos_kmers)
    #     writer.writerows(todas_linhas)

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
    caminho = "proteinas.txt"
    skip = 1
    nprt = 2
    range_pca = 50
    saida_csv = f"kmer_2x2_skip{skip}_binario.parquet"
    n_processos = 3
    
    extrair_kmers_binario_parallel(caminho, skip, nprt, saida_csv, n_processos, range_pca)
