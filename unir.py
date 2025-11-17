import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def plot_clustering(data, labels, title):
    """Função auxiliar para plotar resultados de clustering"""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel('Primeira Componente Principal')
    plt.ylabel('Segunda Componente Principal')
    plt.show()

def kmeans_clustering(pca_file):
    """K-means clustering"""
    df = pd.read_csv(pca_file)
    data = df.values
    
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    
    plot_clustering(data_scaled, labels, 'K-means Clustering')
    return labels

def dbscan_clustering(pca_file):
    """DBSCAN clustering"""
    df = pd.read_csv(pca_file)
    data = df.values
    
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(data_scaled)
    
    plot_clustering(data_scaled, labels, 'DBSCAN Clustering')
    return labels

def hierarchical_clustering(pca_file):
    """Clustering Hierárquico Aglomerativo"""
    df = pd.read_csv(pca_file)
    data = df.values
    
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Aplicar clustering hierárquico
    hierarchical = AgglomerativeClustering(n_clusters=3)
    labels = hierarchical.fit_predict(data_scaled)
    
    plot_clustering(data_scaled, labels, 'Clustering Hierárquico')
    return labels

def gmm_clustering(pca_file):
    """Gaussian Mixture Model clustering"""
    from sklearn.mixture import GaussianMixture
    
    df = pd.read_csv(pca_file)
    data = df.values
    
    # Normalizar os dados
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Aplicar GMM
    gmm = GaussianMixture(n_components=3, random_state=42)
    labels = gmm.fit_predict(data_scaled)
    
    plot_clustering(data_scaled, labels, 'GMM Clustering')
    return labels

# Exemplo de uso:
if __name__ == "__main__":
    pca_file = "pca_output.csv"
    
    # K-means
    kmeans_labels = kmeans_clustering(pca_file)
    
    # DBSCAN
    dbscan_labels = dbscan_clustering(pca_file)
    
    # Hierárquico
    hierarchical_labels = hierarchical_clustering(pca_file)
    
    # GMM
    gmm_labels = gmm_clustering(pca_file)