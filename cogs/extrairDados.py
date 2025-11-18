from Bio import SeqIO
from itertools import product
import numpy as np
import multiprocessing as mp
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

AA = "ACDEFGHIKLMNPQRSTVWY"

def combinacoes_sequencia(seq, skip, nprt): #retorna toda as combinações que a sequencia tem
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

def escrever_bloco(buffer_ids, buffer_matrizes, todos_kmers, schema, writer):
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

def processar_bloco(index, registros, skip, nprt, todos_kmers, kmer_index, chunk_size=1000):
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
        for k in set(combinacoes_sequencia(reg.seq, skip, nprt)):
            idx = kmer_index.get(k)
            if idx is not None:
                vetor[idx] = 1
                
        buffer_ids.append(reg.id)
        buffer_matrizes.append(vetor)
        
        if len(buffer_ids) >= chunk_size:

            escrever_bloco(buffer_ids, buffer_matrizes, todos_kmers, schema, writer)

    if buffer_ids:
        escrever_bloco(buffer_ids, buffer_matrizes, todos_kmers, schema, writer)

    writer.close()
    return arquivo

def unir_parquets(arquivos, saida):
    print("unificando arquivos ", arquivos)
    lf = pl.concat(
        [pl.scan_parquet(a) for a in arquivos],
        how="vertical",
        rechunk=False
    )

    lf.sink_parquet(saida, compression="zstd")
    print("arquivos unificados em ", saida)
    return 

def extrair_dados(fasta_path, skip, nprt, saida_csv, n_processos):
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
    return saida_csv