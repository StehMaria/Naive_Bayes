import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import naive_bayes
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('iris.csv',sep=',',index_col=None)
df = df.drop(columns=['Id'])
print(df.head(15))

X = df[['SepalLengthCm','SepalWidthCm', 'PetalWidthCm', 'PetalLengthCm']].values
Y = df[['Species']].values

cv = skl.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=2020 )

treino_indice, teste_indice = list(cv.split(X,Y))[0]
print(treino_indice)
X_treino, X_teste = X[treino_indice,:],X[teste_indice,:]

Y_treino, Y_teste = Y[treino_indice],Y[teste_indice]
#binario, reais e categoricos, como é categorico usa método gaussiano

classificador = skl.naive_bayes.GaussianNB()
classificador.fit(X_treino,Y_treino)

Y_predito = classificador.predict(X_teste)
nomes_classes = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']

matrix = skl.metrics.confusion_matrix(Y_teste, Y_predito, nomes_classes)
plt.figure(figsize=(18,3))
sns.heatmap(matrix, xticklabels=nomes_classes, yticklabels=nomes_classes,annot=True)

plt.xlabel('Classes Preditas')
plt.ylabel('Classes Reais')
plt.show()

from sklearn.metrics import accuracy_score
acuracia = accuracy_score(Y_teste, Y_predito)
print('Acurácia: %f' % acuracia)