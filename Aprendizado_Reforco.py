# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

# bibliotecas utilizadas
import numpy as np

# método para imprimir o mundo de grades
def imprime(Q):
# quantidade de linhas
  rows = len(Q); 
# quantidade de colunas
  cols = len(Q[0])
  for i in range(rows):
# imprime os índices por linha
    print("%d " % i, end="")
    if i < 10: 
        print(" ", end="")
    for j in range(cols): 
# imprime as linhas e colunas
        print(" %6.2f" % Q[i,j], end="")
    print("")
  print("")

# método para retornar a lista com os próximos estados
def prox_estados(s, F, ns):
  prox_estados = []
  for j in range(ns):
    if F[s,j] == 1: 
        prox_estados.append(j)
  return prox_estados

# método que retorna o próximo estado aleatóriamente
def prox_estado_randomico(s, F, ns):
  proximos_estados_possiveis = prox_estados(s, F, ns)
# a sugestão de próximos estados vai de 0 até a quantidade de estados possíveis com o tamanho da lista de próximos estados
  prox_estado = proximos_estados_possiveis[np.random.randint(0, len(proximos_estados_possiveis))]
  return prox_estado 

# função de aprendizado por reforço
def QL(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs):
  for i in range(0,max_epochs):
    curr_s = np.random.randint(0,ns)

    while(True):
      next_s = prox_estado_randomico(curr_s, F, ns)
      poss_next_next_states = prox_estados(next_s, F, ns)

      max_Q = -9999.99
      for j in range(len(poss_next_next_states)):
        nn_s = poss_next_next_states[j]
        q = Q[next_s,nn_s]
        if q > max_Q:
          max_Q = q
      Q[curr_s][next_s] = ((1 - lrn_rate) * Q[curr_s] [next_s]) + (lrn_rate * (R[curr_s][next_s] + (gamma * max_Q)))
      curr_s = next_s
      
      if curr_s == goal: 
          break
      
# método principal
def main():
# define a aleatoriedade
    np.random.seed(1)
  
# matriz de zeros com o mundo de grades (matriz 4x4)
    F = np.zeros(shape=[4, 4], dtype=np.int)
# define o primeiro e o ultimo estado como estados targets
    F[0, 1] = 1; F[1, 0] = 1;

# matriz de apoio para as trasições dos estados
    R = np.zeros(shape=[15,15], dtype=np.int)
    R[0,1] = -0.1; R[1,0] = -0.1;

# nova matriz 4x4 para apoio ao mundo de grades
    Q = np.zeros(shape=[4,4], dtype=np.float32)
  
# parâmetros utilizados no modelo
    goal = 1
    ns = 2
    gamma = 0.9
    lrn_rate = 0.1
    max_epochs = 1000
  
# chama a função de aprendizado por reforço
    QL(F, R, Q, gamma, lrn_rate, goal, ns, max_epochs)

# executa o método principal
main()