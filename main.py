
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

#data set link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

def save_table_as_image(df, filename="tabla_resultados.jpg", title="Tabla de Resultados"):
    fig, ax = plt.subplots(figsize=(8, 3 + len(df) * 0.3))
    ax.axis('off')
    ax.axis('tight')

    tabla = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.2, 1.2)

    plt.title(title, fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Imagen guardada como '{filename}'")

print("Cargando dataset...")
df = pd.read_csv("creditcard.csv")

# Tomamos una muestra balanceada
df_sample = pd.concat([
    df[df["Class"] == 0].sample(5000, random_state=42),  # no fraude
    df[df["Class"] == 1]                                 # todos los fraudes
])

df = df_sample.reset_index(drop=True)

print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
print(df.head())

# Preprocesamiento
# Eliminamos columnas no necesarias (si las hubiera)
X = df.drop(columns=["Class"])
y = df["Class"]

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducción de dimensionalidad (opcional para visualización)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Aplicar DBSCAN
print("Ejecutando DBSCAN...")
dbscan = DBSCAN(eps=1.8, min_samples=5, n_jobs=-1)
labels = dbscan.fit_predict(X_pca)

# Añadimos las etiquetas de cluster al dataframe
df["cluster"] = labels

# Análisis de resultados
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"\nNúmero de clusters encontrados: {n_clusters}")
print(f"Puntos considerados ruido: {n_noise}")

# Crear tabla resumen
summary = (
    df.groupby("cluster")["Class"]
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={0: "No Fraude", 1: "Fraude"})
)
summary["Total"] = summary.sum(axis=1)
summary["% Fraudes"] = (summary["Fraude"] / summary["Total"] * 100).round(2)

print("\nDistribución por cluster:")
print(summary)
save_table_as_image(summary.reset_index(), "clusters_fraude.jpg", "Distribución de Fraudes por Cluster")

# Visualización
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=labels,
    palette="tab10",
    s=15,
    legend="full"
)
plt.title("Clusters detectados por DBSCAN (PCA reducido)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# Métricas básicas (si se desea comparar con la clase real)
# Los puntos con label = -1 se pueden interpretar como anomalías
outliers = df[df["cluster"] == -1]
print(f"\nPorcentaje de anomalías detectadas por DBSCAN: {len(outliers)/len(df)*100:.2f}%")

frauds_in_outliers = outliers["Class"].sum()
print(f"Fraudes reales dentro de los outliers: {frauds_in_outliers} de {outliers.shape[0]} detectados.")
