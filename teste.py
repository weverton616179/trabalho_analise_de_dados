import polars as pl

def pegar_primeira_coluna_ultima_linha(arquivo):
    df = pl.scan_parquet(arquivo)           # Lazy, não carrega tudo na RAM
    ultima = df.tail(1).collect()           # Coleta só a última linha
    primeira_coluna = ultima.columns[0]     # Nome da primeira coluna
    return ultima[0, primeira_coluna]

valor = pegar_primeira_coluna_ultima_linha("kmer_2x2_skip1_binario.parquet")
print(valor)