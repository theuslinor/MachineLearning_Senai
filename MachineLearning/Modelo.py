from sklearn.neural_network import MLPClassifier

pato1 = [1, 0, 0, 0]
pato2 = [0, 1, 0, 0]
pato3 = [0, 0, 1, 0]
pato4 = [0, 0, 0, 1]
ganso1 = [1, 1, 1, 0]
ganso2 = [0, 1, 1, 1]
ganso3 = [1, 1, 0, 1]
ganso4 = [1, 0, 1, 1]
gato1 = [2, 1, 1, 0]
gato2 = [1, 2, 0, 1]
gato3 = [1, 0, 2, 1]
gato4 = [0, 1, 1, 2]

treino_x = [pato1, pato2, pato3, pato4, ganso1, ganso2, ganso3, ganso4, gato1, gato2, gato3, gato4]
treino_y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

modelo = MLPClassifier(hidden_layer_sizes=(500, 500, 500, 500), max_iter=100000).fit(treino_x, treino_y)

animal_misterioso = [0, 1, 1, 1]