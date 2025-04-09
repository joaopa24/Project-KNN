import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()
dados = iris.data
rotulos = iris.target

# Usar train_test_split para dividir os dados (80% treino, 20% teste)
X_mem, X_teste, y_mem, y_teste = train_test_split(
    dados, rotulos, test_size=0.2, random_state=0, shuffle=True
)

def calcularDistancia(ponto1, ponto2):
    soma = 0
    for i in range(len(ponto1)):
        soma += (ponto1[i] - ponto2[i]) ** 2
    return np.sqrt(soma)

def preverPonto(X_mem, y_mem, ponto_novo, k):
    distancias = []
    for i in range(len(X_mem)):
        d = calcularDistancia(ponto_novo, X_mem[i])
        distancias.append((d, y_mem[i]))
    distancias.sort()
    vizinhos = distancias[:k]
    classes = [d[1] for d in vizinhos]
    contagem = Counter(classes)
    return contagem.most_common(1)[0][0]

def classificarTeste(X_mem, y_mem, X_teste, k):
    return [preverPonto(X_mem, y_mem, ponto, k) for ponto in X_teste]

def calcularMetricas(y_verdadeiro, y_previsto):
    total = len(y_verdadeiro)
    acertos = sum([1 for i in range(total) if y_verdadeiro[i] == y_previsto[i]])
    acuracia = acertos / total

    classes = sorted(set(y_verdadeiro))
    metricas = {}
    for classe in classes:
        tp = sum((y == classe and y_hat == classe) for y, y_hat in zip(y_verdadeiro, y_previsto))
        fp = sum((y != classe and y_hat == classe) for y, y_hat in zip(y_verdadeiro, y_previsto))
        fn = sum((y == classe and y_hat != classe) for y, y_hat in zip(y_verdadeiro, y_previsto))
        precisao = tp / (tp + fp) if (tp + fp) != 0 else 0
        revocacao = tp / (tp + fn) if (tp + fn) != 0 else 0
        metricas[classe] = {"precisao": precisao, "revocacao": revocacao}

    return acuracia, metricas

for k in [1, 3, 5, 7]:
    print(f"\n>>> usando k={k}")

    print("\n KNN sem libs ")
    previsoesManual = classificarTeste(X_mem, y_mem, X_teste, k)
    acuraciaManual, metricasManual = calcularMetricas(y_teste, previsoesManual)
    print(f"Acurácia-z: {acuraciaManual:.2f}")
    for classe, valores in metricasManual.items():
        print(f"Classe {classe}:")
        print(f"  Precisão:  {valores['precisao']:.2f}")
        print(f"  Revocação: {valores['revocacao']:.2f}")
    print(f"Acurácia geral: {acuraciaManual:.2f}")

    matrizManual = np.zeros((3, 3), dtype=int)
    for y_real, y_pred in zip(y_teste, previsoesManual):
        matrizManual[y_real][y_pred] += 1
    sns.heatmap(matrizManual, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz Confusão k = {k} sem libs')
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe real')
    plt.show()

    print("\n resultados com libs: ")

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_mem, y_mem)
    previsoesSklearn = knn.predict(X_teste)
    acuraciaSklearn = accuracy_score(y_teste, previsoesSklearn)
    print(f"Acurácia (scikit-learn): {acuraciaSklearn:.2f}")

    relatorioSklearn = classification_report(y_teste, previsoesSklearn, output_dict=True)
    for classe in relatorioSklearn:
        if classe in ['0', '1', '2']:
            print(f"Classe {classe}:")
            print(f"  Precisão:  {relatorioSklearn[classe]['precision']:.2f}")
            print(f"  Revocação: {relatorioSklearn[classe]['recall']:.2f}")
    print(f"Acurácia geral: {relatorioSklearn['accuracy']:.2f}")

    matrizSklearn = confusion_matrix(y_teste, previsoesSklearn)
    sns.heatmap(matrizSklearn, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Matriz Confusão k = {k} com libs (train_test_split)')
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe real')
    plt.show()

    print("\n>>> COMPARAÇÃO ACURÁCIA <<<")
    print(f"KNN sem libs:        {acuraciaManual:.4f}")
    print(f"KNN com scikit-learn: {acuraciaSklearn:.4f}")
