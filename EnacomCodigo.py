#O código desenvolvido aqui, não foi o solicitado, mas esse modo foi o qual eu aprendi um pouco usando o gradiente descendente
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Preparando o Dataset
x1_numpy, x2_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

#Convertendo de um Array Numpy para o tipo tensor
x1 = torch.from_numpy(x1_numpy.astype(np.float32))
x2 = torch.from_numpy(x2_numpy.astype(np.float32))
x2 = x2.view(x2.shape[0], 1)

print(x1.shape)
print(x2.shape)

plt.plot(x1_numpy, x2_numpy, 'ro')

# MODELO
input_size = 1
output_size = 1
modelo = nn.Linear(input_size, output_size)

# DEFINIÇÃO DA FUNÇAO DE CUSTO E OTIMIZADOR
learning_rate = 0.01
criterion = nn.MSELoss()
otimizador = torch.optim.SGD(modelo.parameters(), lr=learning_rate)
print(modelo.parameters())

# LOOP DE TREINAMENTO
interacoes = 100
contador_custo = []
for epoca in range(interacoes):
    resultado = modelo(x1)
    perda = criterion(resultado, x2)
    contador_custo.append(perda)

    #Calcular gradientes
    perda.backward()

    #Atualizar os pesos
    otimizador.step()

    if (epoca + 1) % 10 == 0:
        print('Epoch: ', epoca)
        print('Custo: {:.5f}'.format(perda.item()))
        print('x1: {:.5f}'.format(modelo.weight.data.detach().item()))
        print('x1 (gradiente): {:.5f}'.format(modelo.weight.grad.detach().item()))
        print('x2: {:.20f}'.format(modelo.bias.data.detach().item()))
        print('x2 (gradiente): {:.5f}'.format(modelo.bias.grad.detach().item()))
        previsao_final = resultado.detach().numpy()
        plt.plot(x1_numpy, x2_numpy, 'ro')
        plt.plot(x1_numpy, previsao_final, 'b')
        plt.show()

    # limpando o otimizador
    otimizador.zero_grad()

# PLOTANDO O GRÁFICO DA FUNÇÃO DE CUSTO
plt.plot(contador_custo, 'b')
plt.show()