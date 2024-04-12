from Modelo import modelo
from sklearn.metrics import accuracy_score, confusion_matrix


misterio1 = [1, 1, 1]
misterio2 = [1, 0, 0]
misterio3 = [0, 0, 0]

teste_x = [misterio1, misterio2, misterio3]
teste_y = [1, 1, 0]

previsoes_y = modelo.predict(teste_x)
acuraria = accuracy_score(teste_y, previsoes_y)
print(f'Acurácia: {acuraria:.3f}')