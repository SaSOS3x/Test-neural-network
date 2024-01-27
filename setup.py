import numpy as np
import array
from numpy import exp, array, random, dot

import sqlite3
import pickle

connect = sqlite3.connect("db.db") # Connecting DB
cursor = connect.cursor() # indicate cursor, for manipulating db

cursor.execute( "SELECT * FROM data") # Catch the synaptic weights from db

res = cursor.fetchall()

synaptic_weights = pickle.loads(res[0][0])

task = array([[1,0,0,1]])

output = 1 / (1 + exp(-(dot(task, synaptic_weights))))

if output == 0.5:
    print("похуй")
else:
    print(output)
