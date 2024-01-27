import numpy as np
import random
import array
from numpy import exp, array, random, dot

import sqlite3
import pickle

connect = sqlite3.connect("db.db") # Connecting DB
cursor = connect.cursor() # indicate cursor, for manipulating db

# This is a start training for the first setup

number_of_training_iterations = 10000

#********DATASET*********************************************************************************************************

trannig_set_inputs = array([[1,1,0,1],[0,1,0,1],[1,0,0,0],[0,1,1,0]]) # training set inputs
training_set_outputs = array([[1,0,1,0]]).T # training set outputs

#********SIZE************************************************************************************************************ 

synaptic_weights = 2 * random.random((4, 1)) - 1 # set the random synaptic weights

#********NEURON**********************************************************************************************************

for iteration in range(number_of_training_iterations):
    output = 1 / (1 + exp(-(dot(trannig_set_inputs,synaptic_weights))))
    synaptic_weights += dot(trannig_set_inputs.T, (training_set_outputs - output) * output * (1 - output))

    print (1 / (1 + exp(-(dot(trannig_set_inputs, synaptic_weights))))) # print result

# Commit synaptic_weights changes

db_array = pickle.dumps(synaptic_weights) # Преобразование данных в бинарный массив

query = f"UPDATE data SET weight = ? WHERE id = 1"

cursor.execute(query, [db_array])
connect.commit()

# print(f"\n{arr}")

#************************************************************************************************************************