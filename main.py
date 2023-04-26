import pandas as pd
import numpy as np
from numpy import mean, sqrt
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

csv = pd.read_csv("SIA2ªAF.csv", sep=",")
dados = csv.values
entrada = dados[:,0:10]
saida = dados[:,10]


# Criar o modelo de regressão e árvore com regularização de alpha 1
model = Lasso(alpha=1)
modelo_arvore = DecisionTreeRegressor()

#estimador de regressão
modelo_mean = VotingRegressor(estimators=[('lasso', model), ('arvore', model)])

loo = LeaveOneOut()
rmse_treino = []
rmse_val = []
for treino, validacao in loo.split(entrada):
    entrada_treino, entrada_validacao = entrada[treino], entrada[validacao]
    saida_treino, saida_validacao = saida[treino], saida[validacao]
   
    modelo_mean.fit(entrada_treino, saida_treino)
    previsao_treino = modelo_mean.predict(entrada_treino)
    previsao_validacao = modelo_mean.predict(entrada_validacao)

# Treinar o modelo

rmse_treino.append(sqrt(mean_squared_error(saida_treino, previsao_treino)))
rmse_val.append(np.sqrt(mean_squared_error(saida_validacao, previsao_validacao)))



print('Valor médio do RMSE para a base de treino:', mean(rmse_treino))
print('Valor médio do RMSE para a base de validacao:', mean(rmse_val))