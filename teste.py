import polars as pl

# 1. ler somente a primeira linha
first_row = pl.read_parquet("k1.parquet", n_rows=1)

# 2. transformar linha em dict
row_dict = first_row.row(0, named=True)

# 3. pegar colunas onde valor == 1
cols_with_one = [val for col, val in row_dict.items()]

print("Colunas com valor 1 na primeira linha:", len(cols_with_one))
print(cols_with_one)