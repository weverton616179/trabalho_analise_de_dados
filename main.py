from cogs.extrairDados import extrair_dados
from cogs.pca import pca_parquet

if __name__ == "__main__":
    caminho = "teste.fa"
    skip = 1
    nprt = 2
    range_pca = 300
    saida_csv = f"kmer_2x2_skip{skip}_binario.parquet"
    n_processos = 2
    
    extrair_dados(caminho, skip, nprt, saida_csv, n_processos)
    pca_parquet(saida_csv, f"pca_{range_pca}_componentes.csv", range_pca)