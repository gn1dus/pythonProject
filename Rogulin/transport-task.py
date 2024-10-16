import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx

# Генерация случайных данных
I = 20
J = I * 2
price = np.random.randint(26, size=(I, J))
m, n = price.shape
c = list(np.reshape(price, n * m))
a = np.random.randint(100, size=I)
b = np.random.randint(100, size=J)

# Создание матрицы Gamma1
Gamma1 = np.zeros([m, m * n])
for i in range(n):
    for j in range(n * m):
        if i * n <= j <= n + i * n - 1:
            Gamma1[i, j] = 1

# Создание матрицы Gamma2
Gamma2 = np.zeros([n, m * n])
for i in range(n):
    k = 0
    for j in range(n * m):
        if j == k * n + i:
            Gamma2[i, j] = 1

# Решение транспортной задачи
if np.sum(a) < np.sum(b):
    res = linprog(c=c, A_ub=Gamma2, b_ub=b, A_eq=Gamma1, b_eq=a, method='highs')
elif np.sum(a) > np.sum(b):
    res = linprog(c=c, A_ub=Gamma1, b_ub=a, A_eq=Gamma2, b_eq=b, method='highs')
elif np.sum(a) == np.sum(b):
    A_eq = np.concatenate((Gamma1, Gamma2), axis=0)
    b_eq = np.concatenate((a, b), axis=0)
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, method='highs')

result = np.array(list(res['x'])).reshape(I, J)

# Создание графа
G_result = nx.Graph()

# Добавление узлов для предложения и спроса
for i in range(I):
    for j in range(J):
        if result[i, j] != 0:  # Проверяем, что вес ребра не равен нулю
            G_result.add_edge(f'Supply_{i+1}', f'Demand_{j+1}', weight=result[i, j])

# Визуализация графа
plt.figure(figsize=(12, 8))

# Разделение узлов на два списка: предложение и спрос
supply_nodes_result = [node for node in G_result.nodes if 'Supply' in node]
demand_nodes_result = [node for node in G_result.nodes if 'Demand' in node]

# Определение позиций узлов
pos_result = {}
pos_result.update((node, (0, index * 1)) for index, node in enumerate(supply_nodes_result))  # Увеличиваем расстояние между узлами предложения
pos_result.update((node, (1, index * 1)) for index, node in enumerate(demand_nodes_result))  # Увеличиваем расстояние между узлами спроса

# Отрисовка узлов
nx.draw_networkx_nodes(G_result, pos_result, nodelist=supply_nodes_result, node_color='lightblue', node_shape='o', node_size=200, label='Предложение')
nx.draw_networkx_nodes(G_result, pos_result, nodelist=demand_nodes_result, node_color='lightcoral', node_shape='s', node_size=200, label='Спрос')

# Отрисовка рёбер с ненулевым весом
for u, v, d in G_result.edges(data=True):
    if d['weight'] != 0:
        nx.draw_networkx_edges(G_result, pos_result, edgelist=[(u, v)], width=1, alpha=0.5)

# Добавление подписей к узлам
nx.draw_networkx_labels(G_result, pos_result, font_size=10, font_color='black')

# Добавление легенды
plt.legend()

plt.title('Двудольный граф оптимальных цен')
plt.axis('off')
plt.show()

print(f"Решение: x в таблице trans_x.csv, f = {res['fun']}, {res['message']}")
np.savetxt("trans_x.csv", result, fmt='%d', delimiter=",")

