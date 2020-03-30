# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:15:02 2020

@author: User SaraRod
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint

#import big_o

import time

import csv

# ---------------------------------------------------------------------------------------------------------
"""
    Método auxiliar que retorna el estado de la memoria
"""

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

# ---------------------------------------------------------------------------------------------------------

"""
    Método auxiliar que lee el tiempo que toma leer cierto dataset
"""

def read_dataset_time(dataseto):
    col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]
    start_time = time.time()
    startHeap = memory_usage_psutil()
    df = pd.read_csv(dataseto,sep=';',encoding='UTF-8',usecols=col_list)
    read_time = (time.time() - start_time)
    finalHeap = memory_usage_psutil() - startHeap
    df = df.rename(columns={"exito": "label"})
    
    #Vamos a llenar los nulos y pasar el sí/no a booleanos
    df = df.replace("Si", "SI")
    df = df.replace("No", "NO")
    df['profundiza'] = df['profundiza'].fillna(0.0)
    df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
    df = df.replace(np.nan, "Desconocido")
    
    return read_time
  

# ----------------------------------------------------------------------------------------------------------
 
"""
    Método auxiliar que lee el tiempo que toma construir un array numpy
"""    

def build_array_time(dataseto):
    col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]
    
    df = pd.read_csv(dataseto,sep=';',encoding='UTF-8',usecols=col_list)
    df = df.rename(columns={"exito": "label"})
    
    #Vamos a llenar los nulos y pasar el sí/no a booleanos
    df = df.replace("Si", "SI")
    df = df.replace("No", "NO")
    df['profundiza'] = df['profundiza'].fillna(0.0)
    df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
    df = df.replace(np.nan, "Desconocido")
    
    #startHeap = memory_usage_psutil()
    start_time = time.time()
    array_data = df.to_numpy()
    read_time = (time.time() - start_time)
    #finalHeap = memory_usage_psutil() - startHeap
    
    return read_time

# -----------------------------------------------------------------------------------------------------------

"""
    Método auxiliar que lee el tiempo que toma construir una lista de python
"""       

def build_list_time(dataseto):
    col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]
    
    df = pd.read_csv(dataseto,sep=';',encoding='UTF-8',usecols=col_list)
    df = df.rename(columns={"exito": "label"})
    
    #Vamos a llenar los nulos y pasar el sí/no a booleanos
    df = df.replace("Si", "SI")
    df = df.replace("No", "NO")
    df['profundiza'] = df['profundiza'].fillna(0.0)
    df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
    df = df.replace(np.nan, "Desconocido")
    
    #startHeap = memory_usage_psutil()
    start_time = time.time()
    array_data = df.values.tolist()
    read_time = (time.time() - start_time)
    #finalHeap = memory_usage_psutil() - startHeap
    
    return read_time

# -----------------------------------------------------------------------------------------------------------
    
"""
    Método auxiliar que lee el uso de memoria con una lista python
"""   
    
def build_list_heap(dataseto):
    col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]
    
    df = pd.read_csv(dataseto,sep=';',encoding='UTF-8',usecols=col_list)
    df = df.rename(columns={"exito": "label"})
    
    #Vamos a llenar los nulos y pasar el sí/no a booleanos
    df = df.replace("Si", "SI")
    df = df.replace("No", "NO")
    df['profundiza'] = df['profundiza'].fillna(0.0)
    df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
    df = df.replace(np.nan, "Desconocido")
    
    startHeap = memory_usage_psutil()
    #start_time = time.time()
    array_data = df.values.tolist()
    #read_time = (time.time() - start_time)
    finalHeap = memory_usage_psutil() - startHeap
    
    return finalHeap

# -----------------------------------------------------------------------------------------------------------
    
"""
    Método auxiliar que toma el tiempo de lectura + conversión a array de un dataset
"""  
    
def build_allSet_time(dataseto):
    col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]
    
    start_time = time.time()
    startHeap = memory_usage_psutil()
    df = pd.read_csv(dataseto,sep=';',encoding='UTF-8',usecols=col_list)
    df = df.rename(columns={"exito": "label"})
    
    #Vamos a llenar los nulos y pasar el sí/no a booleanos
    df = df.replace("Si", "SI")
    df = df.replace("No", "NO")
    df['profundiza'] = df['profundiza'].fillna(0.0)
    df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
    df = df.replace(np.nan, "Desconocido")
    
    array_data = df.to_numpy()
    read_time = (time.time() - start_time)
    finalHeap = memory_usage_psutil() - startHeap
    
    return read_time

    # Para retornar espacio en memoria > finalHeap

# -----------------------------------------------------------------------------------------------------------


col_list = [ 4,5,6,7,8,9,10,11,12,13,26,28,29,30,31,32,33,35,36,45,46,47,65,66,67,68,69,10,71,72,73,74,75,76,77 ]

df = pd.read_csv("0_train_balanced_15000.csv",sep=';',encoding='UTF-8',usecols=col_list)
df = df.rename(columns={"exito": "label"})

#Vamos a llenar los nulos y pasar el sí/no a un solo formato
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)

df = df.replace(np.nan, "Desconocido")

#print(df.head())
#print(df.info())
#print(df['fami_numlibros'])


train_dataSet = df # TRAIN VALUES IN DATASET FOMAT
#train_data = train_dataSet.values
training_data = df.to_numpy()

print(training_data)
print(train_dataSet)

# pandas.DataFrame.values = DataFrame.to_numpy. Se recomienda el .to_numpy porque es mas nuevo
# drop rows with missing value
#print(df.dropna())
#training_data = []
#training_data = df.values.tolist()
#print(training_data)

# ---------------------------------------------------------------------------------------------------------

"""
trabajo con CSV READER
"""

print("LECTURA COMPARATIVA CON CSV READER")


def csv_reading_time(dataseto):
    data_Reader = []
    start_time = time.time()
    with open(dataseto,encoding="utf8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                #print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                data_Reader.append(row)
                line_count += 1
        read_time = (time.time() - start_time)
        print(f'Processed {line_count} lines.')
        
        return read_time
    
    #print(data_Reader)
    
    #print('My list:', *data_Reader, sep='\n- ')




# ---------------------------------------------------------------------------------------------------------

df = pd.read_csv("0_test_balanced_5000.csv",sep=';',encoding='UTF-8',usecols=col_list)
df = df.rename(columns={"exito": "label"})

#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")

test_dataSet = df
#test_data = test_dataSet.values
testing_data = df.to_numpy()

#print(testing_data)

test_size= testing_data.size


print(csv_reading_time("0_test_balanced_5000.csv"))
print(csv_reading_time("0_train_balanced_15000.csv"))
print(csv_reading_time("5_test_balanced_19255.csv"))
print(csv_reading_time("2_test_balanced_25000.csv"))
print(csv_reading_time("3_test_balanced_35000.csv"))
print(csv_reading_time("1_train_balanced_45000.csv"))
print(csv_reading_time("5_train_balanced_57765.csv"))
print(csv_reading_time("2_train_balanced_75000.csv"))
print(csv_reading_time("3_train_balanced_105000.csv"))
print(csv_reading_time("4_train_balanced_135000.csv"))

print(read_dataset_time("0_test_balanced_5000.csv"))
print(read_dataset_time("0_train_balanced_15000.csv"))
print(read_dataset_time("5_test_balanced_19255.csv"))
print(read_dataset_time("2_test_balanced_25000.csv"))
print(read_dataset_time("3_test_balanced_35000.csv"))
print(read_dataset_time("1_train_balanced_45000.csv"))
print(read_dataset_time("5_train_balanced_57765.csv"))
print(read_dataset_time("2_train_balanced_75000.csv"))
print(read_dataset_time("3_train_balanced_105000.csv"))
print(read_dataset_time("4_train_balanced_135000.csv"))

# ---------------------------------------------------------------------------------------------------------  

"""
    método auxiliar para sacar un test-sample del mismo dataframe del que se hace el train.
    Por ahora no tiene uso, pero se deja por si acaso.
"""
def train_test_split(df, test_size):
     
    #Por si se quieren trabajar con porcentajes de la longitud total
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

random.seed(0)
#train_df, test_df = train_test_split(df, test_size=20)

# -----------------------------------------------------------------------------------------------------------


"""
    Mira que tan mezclado está el data, si es pure retorna true, mezclado feo es false.

"""

def check_purity(data):
    
    #label column tiene todos los , en nuestro caso solo debe haber 0 y 1
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

#print(check_purity(training_data[training_data.punt_lenguaje>50]))
# Forma de filtrar el data
    
 # -----------------------------------------------------------------------------------------------------------
    
"""
    Función que clasifica 

"""
    
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification 

print(classify_data(training_data))

label_column = training_data[:,-1]
print(np.unique(label_column, return_counts = True))

# print(classify_data(train_dataSet[train_dataSet.punt_biologia>50]))