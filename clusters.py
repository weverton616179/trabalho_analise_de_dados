import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Importando algoritmos de clusterização
from sklearn.cluster import (
    KMeans, 
    DBSCAN, 
    AgglomerativeClustering, 
    MeanShift, 
    OPTICS, 
    Birch,
    MiniBatchKMeans
)
from sklearn.mixture import GaussianMixture

# 1. Carregamento e Preparação dos Dados
# Substitua 'seu_arquivo.csv' pelo caminho do seu arquivo real
file_path = 'pca_300_componentes.csv' 

try:
    # Lê o CSV
    df = pd.read_csv(file_path)
    
    # Define a coluna 'sequencia' como índice para manter o ID, mas não usar no cálculo
    if 'sequencia' in df.columns:
        df.set_index('sequencia', inplace=True)
    
    # Remove valores nulos se houver (limpeza básica)
    df.dropna(inplace=True)
    
    print(f"Dados carregados. Dimensões: {df.shape}")
    
    # Seleciona apenas as colunas de PC (PC1, PC2...) para o treino
    # O código assume que todas as colunas restantes são numéricas
    X = df.values

except Exception as e:
    print(f"Erro ao carregar arquivo: {e}")
    exit()

# 2. Configuração dos Algoritmos
# NOTA: Alguns algoritmos precisam de parâmetros ajustados (como eps no DBSCAN ou n_clusters no KMeans)
# Configurei valores padrão comuns, mas para resultados científicos você deve ajustá-los.

n_clusters_default = 3 # Defina quantos grupos você ESTIMA que existem

modelos = {
    "K-Means": KMeans(n_clusters=n_clusters_default, random_state=42, n_init=10),
    "MiniBatch K-Means": MiniBatchKMeans(n_clusters=n_clusters_default, random_state=42, n_init=10), # Mais rápido para arquivos grandes
    "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters_default),
    "Birch": Birch(n_clusters=n_clusters_default), # Ótimo para datasets grandes
    "Gaussian Mixture (GMM)": GaussianMixture(n_components=n_clusters_default, random_state=42),
    
    # Algoritmos baseados em densidade (não precisam definir número de clusters, mas precisam de ajuste de distância)
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5), 
    "OPTICS": OPTICS(min_samples=5),
    # "MeanShift": MeanShift(), # CUIDADO: Muito lento em datasets grandes, descomente se tiver < 10k linhas
}

# 3. Execução e Plotagem
# Configura o estilo dos gráficos
sns.set_theme(style="whitegrid")

# Loop para rodar cada modelo e gerar o gráfico
for nome, modelo in modelos.items():
    print(f"Rodando: {nome}...")
    
    try:
        # Treina e prediz os clusters
        if "Gaussian Mixture" in nome:
            labels = modelo.fit_predict(X)
        else:
            labels = modelo.fit_predict(X)
        
        # Adiciona os labels ao dataframe temporário para plotagem
        df_plot = df.copy()
        df_plot['Cluster'] = labels.astype(str) # Converte para string para o gráfico tratar como categoria
        
        # Criação do Gráfico (Usando PC1 e PC2 como eixos X e Y)
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(
            data=df_plot,
            x='PC1',
            y='PC2',
            hue='Cluster',
            palette='viridis',
            s=60, # Tamanho dos pontos
            alpha=0.7, # Transparência
            edgecolor='k' # Borda dos pontos
        )
        
        plt.title(f"Clusterização com {nome}", fontsize=15)
        plt.xlabel("PC1 (Componente Principal 1)", fontsize=12)
        plt.ylabel("PC2 (Componente Principal 2)", fontsize=12)
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Salvar gráfico ou mostrar
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Não foi possível executar {nome}: {e}")

print("Processo finalizado.")