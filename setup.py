import numpy as np
import array
from numpy import exp, array, random, dot

import sqlite3
import pickle

connect = sqlite3.connect("db.db") # Connecting DB
cursor = connect.cursor() # indicate cursor, for manipulating db

cursor.execute( "SELECT * FROM data") # Catch the synaptic weights from db
res = cursor.fetchall()



synaptic_weights = pickle.loads(res[0][0]) # Преобразование данных в массив numpy



task = array([[1,1,0,0]]) # Массив - задача

output = 1 / (1 + exp(-(dot(task, synaptic_weights)))) # Расчет выхода, сначала считаем выход, потом находим ошибку


# Тут нужно будет сделать норм функцию для обучения каждого нейрона
def autolearn(synaptic_weights):
    
    # Решение: Это система нахождения фактического ответа для обучения
    if task[0, 0] == 1:
        task_output = array([[1]]).T
    else:
        task_output = array([[0]]).T

    print(task_output) # Debug

    synaptic_weights += dot(task.T, (task_output - output) * output * (1 - output))

    return synaptic_weights



new_synaptic_weights = autolearn(synaptic_weights) # Получили новое значение синаптического веса


# Это можно заключить в функцию, но пока не нада
db_array = pickle.dumps(new_synaptic_weights) # Преобразование данных в бинарный массив

query = f"UPDATE data SET weight = ? WHERE id = 1"

cursor.execute(query, [db_array])
connect.commit()



print(f"Не округленный выход сети:\n{output}\nНовый вес нейрона получен: {db_array}") # Result







