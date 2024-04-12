import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import load_csv
from tflearn.datasets import titanic
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.framework import ops


titanic.download_dataset('titanic_dataset.csv')
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)

df = pd.DataFrame(data)
X_train, X_test, Y_train, Y_test = train_test_split(df, labels,
                                                    test_size=0.33, random_state=42)

def preprocess(r):
    r = r.drop([1], axis=1, errors='ignore')
    r[2] = r[2].astype('category')
    r[2] = r[2].cat.codes
    r[6] = r[6].astype('category')
    r[6] = r[6].cat.codes

    for column in r.columns:
        r[column] = r[column].astype(np.float32)
        return r.values

X_train = preprocess(X_train)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, 7])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(X_train, Y_train, n_epoch=20, batch_size=32, show_metric=True)

metric_train = model.evaluate(X_train, Y_train)
metric_test = model.evaluate(X_test, Y_test)
print('Acurácia com dados de treinamento: %.9f' % metric_train[0])
print('Acurácia com dados de teste: %.9f' % metric_test[0])

dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'NA', 5.0000]
winslet = [1, 'Rose', 'female', 17, 1, 2, 'N/A', 100.0000]
dicaprio, winslet = preprocess(pd.DataFrame([dicaprio, winslet]))

pred = model.predict([dicaprio, winslet])

print('Probabilidade de sobrevivencia de DiCaprio: ', pred[0][1])
print('Probabilidade de sobrevivencia de Winslet: ', pred[1][1])


