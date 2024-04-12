from sklearn.neural_network import MLPClassifier

pato1 = [1, 0, 0]
pato2 = [0, 1, 0]
pato3 = [0, 0, 1]
ganso1 = [1, 1, 1]
ganso2 = [0, 1, 1]
ganso3 = [1, 1, 0]
gato1 = [2, 1, 1]
gato2 = [1, 2, 0]
gato3 = [1, 0, 2]
gato4 = [0, 1, 1]

treino_x = [pato1, pato2, pato3, ganso1, ganso2, ganso3]
treino_y = [0, 0, 0, 1, 1, 1]

modelo = MLPClassifier(hidden_layer_sizes=(500, 500, 500), max_iter=10000)
treinado = modelo.fit(treino_x, treino_y)

animal_misterioso = [0, 1, 1]