import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
dados = iris.data
rotulos = iris.target

# Embaralhar os dados e dividir entre treino e teste (80/20)
# print(len(dados))
np.random.seed(42)
indices = np.random.permutation(len(dados))
# print(indices)
tamanho_treino = int(0.8 * len(dados))
indices_treino = indices[:tamanho_treino]
indices_teste = indices[tamanho_treino:]

X_mem = dados[indices_treino]
y_mem = rotulos[indices_treino]
X_teste = dados[indices_teste]
y_teste = rotulos[indices_teste]

def calcularDistancia(ponto1, ponto2):
    soma = 0
    for i in range(len(ponto1)):
        soma += (ponto1[i] - ponto2[i]) ** 2
    distancia = np.sqrt(soma)
    return distancia

def preverPonto(X_mem, y_mem, ponto_novo, k):
    distancias = []
    for i in range(len(X_mem)):
        d = calcularDistancia(ponto_novo, X_mem[i])
        distancias.append((d, y_mem[i]))
    distancias.sort()
    vizinhos = distancias[:k]
    # print(vizinhos)
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

    # Usar os mesmos dados embaralhados da versão manual
    X_treino_lib = X_mem
    y_treino_lib = y_mem
    X_teste_lib = X_teste
    y_teste_lib = y_teste

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_treino_lib, y_treino_lib)
    previsoesSklearn = knn.predict(X_teste_lib)
    acuraciaSklearn = accuracy_score(y_teste_lib, previsoesSklearn)
    print(f"Acurácia (scikit-learn): {acuraciaSklearn:.2f}")

    relatorioSklearn = classification_report(y_teste_lib, previsoesSklearn, output_dict=True)
    for classe in relatorioSklearn:
        if classe in ['0', '1', '2']:
            print(f"Classe {classe}:")
            print(f"  Precisão:  {relatorioSklearn[classe]['precision']:.2f}")
            print(f"  Revocação: {relatorioSklearn[classe]['recall']:.2f}")
    print(f"Acurácia geral: {relatorioSklearn['accuracy']:.2f}")

    matrizSklearn = confusion_matrix(y_teste_lib, previsoesSklearn)
    sns.heatmap(matrizSklearn, annot=True, fmt='d', cmap='Greens')
    plt.title(f'Matriz Confusão k = {k} com libs (mesmos dados)')
    plt.xlabel('Classe prevista')
    plt.ylabel('Classe real')
    plt.show()

    print("\n>>> COMPARAÇÃO ACURÁCIA <<<")
    print(f"KNN sem libs:        {acuraciaManual:.4f}")
    print(f"KNN com scikit-learn: {acuraciaSklearn:.4f}")
