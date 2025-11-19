import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import f1_score, adjusted_rand_score

sns.set(style="whitegrid")

def carregar_dados(caminho_arquivo):
    """
    Carrega o arquivo CSV gerado pelo PCA.
    """
    print(f"\n[PASSO 1] Carregando dados de: {caminho_arquivo}")
    try:
        df = pd.read_csv(caminho_arquivo)
        print(f"--> Arquivo carregado! Dimensões: {df.shape}")
        return df
    except FileNotFoundError:
        print("--> ERRO: Arquivo não encontrado.")
        return None

def preparar_dados(df):
    """
    Separa as features (PC1...PC300) e extrai a classe real do header.
    """
    print("\n[PASSO 2] Preparando dados (Features X e Labels y)...")
    
    cols_features = [c for c in df.columns if c.startswith('PC')]
    X = df[cols_features].values
    
    def extrair_classe(txt):
        return str(txt).split('.')[0]
    
    y_labels = df['sequencia'].apply(extrair_classe).values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)
    
    n_classes = len(np.unique(y_encoded))
    
    print(f"--> Features extraídas: {X.shape}")
    print(f"--> Classes reais identificadas ({n_classes}): {le.classes_}")
    
    return X, y_encoded, n_classes, le.classes_

def alinhar_labels_para_f1(y_true, y_pred):
    """
    Mapeia os clusters numéricos previstos para as classes reais mais prováveis
    para permitir o cálculo justo do F1-Score.
    """
    labels_mapped = np.zeros_like(y_pred)
    for i in np.unique(y_pred):
        mask = (y_pred == i)
        if np.sum(mask) > 0:
            # Atribui a classe real mais frequente nesse cluster
            labels_mapped[mask] = mode(y_true[mask], keepdims=True)[0][0]
    return labels_mapped

def executar_benchmark_clusters(X, y_true, n_clusters_reais):
    """
    Roda múltiplos algoritmos fixando o K igual ao número real de classes
    e compara métricas internas vs externas.
    """
    print(f"\n[PASSO 3] Rodando algoritmos com n_clusters fixo em {n_clusters_reais}...")
    
    algoritmos = [
        ('KMeans', KMeans(n_clusters=n_clusters_reais, random_state=42, n_init=10)),
        ('Agglomerative', AgglomerativeClustering(n_clusters=n_clusters_reais)),
        ('GaussianMixture', GaussianMixture(n_components=n_clusters_reais, random_state=42))
    ]
    
    resultados = []
    
    for nome, modelo in algoritmos:
        print(f"  --> Testando {nome}...")
        try:
            if nome == 'GaussianMixture':
                modelo.fit(X)
                y_pred = modelo.predict(X)
            else:
                y_pred = modelo.fit_predict(X)
            
            sil = silhouette_score(X, y_pred)
            calinski = calinski_harabasz_score(X, y_pred)
            davies = davies_bouldin_score(X, y_pred)
            
            y_pred_mapped = alinhar_labels_para_f1(y_true, y_pred)
            f1 = f1_score(y_true, y_pred_mapped, average='weighted')
            ari = adjusted_rand_score(y_true, y_pred)
            
            resultados.append({
                'Algoritmo': nome,
                'Silhouette': sil, 'CH_Score': calinski, 'DB_Score': davies,
                'F1_Score': f1, 'ARI': ari
            })
            
        except Exception as e:
            print(f"  --> Erro em {nome}: {e}")
            
    return pd.DataFrame(resultados)

def otimizar_parametros(X, y_true, max_k=10):
    """
    Varia o número de clusters (k) para encontrar a 'Melhor Configuração'
    sugerida matematicamente, sem saber a resposta real.
    """
    print(f"\n[PASSO 4] Variando parâmetros (K de 2 a {max_k}) para sugerir melhor configuração...")
    
    inertias = []
    silhouettes = []
    f1_scores = []
    
    range_k = range(2, max_k + 1)
    
    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, y_pred))
        
        y_pred_mapped = alinhar_labels_para_f1(y_true, y_pred)
        f1_scores.append(f1_score(y_true, y_pred_mapped, average='weighted'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(range_k, inertias, 'bo-')
    ax1.set_title('Método do Cotovelo (Elbow Method)')
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inércia (Soma dos erros quadráticos)')
    
    ax2.plot(range_k, silhouettes, 'go-', label='Silhouette (Matemática)')
    ax2.plot(range_k, f1_scores, 'r--', label='F1 Score (Biologia/Real)')
    ax2.set_title('Silhouette vs F1 Score por K')
    ax2.set_xlabel('Número de Clusters (k)')
    ax2.set_ylabel('Score')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("--> Gráficos de otimização gerados.")
    
    melhor_k_idx = np.argmax(silhouettes)
    melhor_k = range_k[melhor_k_idx]
    print(f"--> A análise matemática (Silhouette) sugere que o melhor K é: {melhor_k}")
    return melhor_k

if __name__ == "__main__":
    arquivo = 'pca_300_componentes.csv'
    
    df = carregar_dados(arquivo)
    
    if df is not None:
        X, y_true, n_classes_reais, nomes_classes = preparar_dados(df)
        
        df_resultados = executar_benchmark_clusters(X, y_true, n_classes_reais)
        
        print("\n--- RESULTADOS DA COMPARAÇÃO (PASSO 3) ---")
        print(df_resultados.sort_values(by='F1_Score', ascending=False).to_string(index=False))
        
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df_resultados, x='Silhouette', y='F1_Score', hue='Algoritmo', s=200)
        plt.title('Correlação: Métrica Interna vs Externa')
        plt.show()
        
        melhor_k_sugerido = otimizar_parametros(X, y_true, max_k=n_classes_reais + 5)
        
        print("\n--- CONCLUSÃO DO TRABALHO ---")
        print(f"Número real de classes biológicas: {n_classes_reais}")
        print(f"Número de clusters sugerido pela IA (Não-supervisionado): {melhor_k_sugerido}")
        
        if melhor_k_sugerido == n_classes_reais:
            print("CONCLUSÃO: A metodologia foi perfeitamente consistente com a biologia!")
        else:
            print("CONCLUSÃO: A divisão matemática difere ligeiramente da biológica (o que é normal em dados complexos).")