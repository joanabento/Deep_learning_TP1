#Biblioteca para importar os dados

import pandas as pd


#Ler os dados para a variavel dataset

dataset = pd.read_csv("Dataset.csv", encoding ="UTF-8", sep = ";") 


#Separar a coluna da qualidade dos restantes atributos

labels = dataset["quality"] 
dados = dataset[{"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"}]


#Dividir os dados em treino e  (60% para treino, 40% para teste)

ltreino = labels.loc[1:3000]
lteste = labels.loc[3001:4898]
dtreino = dados.loc[1:3000]
dteste = dados.loc[3001:4898]


#Ajustar formato de dados à rede entrada

dtreino = dtreino.astype("float32")/255
dteste = dtreino.astype("float32")/255


#Ajustar formato de dados à rede saída

from keras.utils import to_categorical

ltreino = to_categorical(ltreino)
lteste = to_categorical(lteste)


#Modelo

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(128, input_shape=(11,), activation="sigmoid"))
network.add(layers.Dense(64, activation="sigmoid"))
network.add(layers.Dense(11, activation="softmax"))

network.compile(optimizer="rmsprop",
                loss="categorical_cossentropy",
                metrics=["accuracy"])


#Treino

network.fit(dtreino,ltreino,epochs=5, batchsize=128)


#Avaliar a rede

test_loss, test_acc = network.evaluate(dteste,lteste)
print("test_acc",test_acc)