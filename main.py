# Importando as bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Carregando o conjunto de dados
iris = load_iris()
X = iris.data
y = iris.target

# Dividindo o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializando o classificador de árvore de decisão
clf = DecisionTreeClassifier()

# Treinando o classificador
clf.fit(X_train, y_train)

# Fazendo previsões
y_pred = clf.predict(X_test)

# Imprimindo as previsões
print(y_pred)
