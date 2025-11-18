import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    auc, precision_recall_curve
)

df = pd.read_csv("creditcard.csv")
print(df.head())

plt.figure(figsize=(6,4))
sns.countplot(x=df["Class"])
plt.title("Distribución de Clases (0 = Normal, 1 = Fraude)")
plt.show()

plt.figure(figsize=(7,4))
sns.histplot(df["Amount"], bins=50, kde=True)
plt.title("Distribución del Monto de Transacciones")
plt.show()


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="coolwarm_r")
plt.title("Mapa de Correlación")
plt.show()

X = df.drop("Class", axis=1)
y = df["Class"]

# Escalar Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

# División del dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Entrenar Random Forest BALANCEADO
model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Reporte
print(classification_report(y_test, y_pred))


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# 10. Importancia de características
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(12,5))
sns.barplot(x=importances, y=features)
plt.title("Importancia de Características en Random Forest")
plt.show()

# 11. Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC")
plt.legend()
plt.show()

# 12. Curva Precision–Recall
precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision–Recall")
plt.show()
