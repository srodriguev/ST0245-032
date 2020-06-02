# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:15:02 2020

@author: User SaraRod
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

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
    Método auxiliar que lee el espacio que toma leer cierto dataset
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

"""

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
"""

# ---------------------------------------------------------------------------------------------------------

"""
trabajo con CSV READER
"""



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




# ---------------------------------------------------------------------------------------------------------

"""
df = pd.read_csv("0_test_balanced_5000.csv",sep=';',encoding='UTF-8',usecols=col_list)
df = df.rename(columns={"exito": "label"})

#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")

test_dataSet = df
test_data = test_dataSet.values
testing_data = df.to_numpy()

print(testing_data)

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
"""
# ---------------------------------------------------------------------------------------------------------  

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


def generate_data(n, specific_outliers=[], n_random_outliers=None):
    
    # create data
    data = np.random.random(size=(n, 2)) * 10
    data = data.round(decimals=1)
    df = pd.DataFrame(data, columns=["x", "y"])
    df["label"] = df.x <= 5

    # add specific outlier data points
    for outlier_coordinates in specific_outliers:
        df = df.append({"x": outlier_coordinates[0],
                        "y": outlier_coordinates[1],
                        "label": True}, 
                       ignore_index=True)

    ## add random outlier data points
    if n_random_outliers:
        outlier_x_values =  (6 - 5) * np.random.random(size=n_random_outliers) + 5  # value between 5 and 6
        outlier_y_values = np.random.random(size=n_random_outliers) * 10

        df_outliers = pd.DataFrame({"x": outlier_x_values.round(decimals=2),
                                    "y": outlier_y_values.round(decimals=2),
                                    "label": [True] * n_random_outliers})

        df = df.append(df_outliers, ignore_index=True)
    
    return df


def plot_decision_boundaries(tree, x_min, x_max, y_min, y_max):
    color_keys = {True: "orange", False: "blue"}
    
    # recursive part
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        yes_answer, no_answer = tree[question]
        feature, _, value = question.split()
    
        if feature == "x":
            plot_decision_boundaries(yes_answer, x_min, float(value), y_min, y_max)
            plot_decision_boundaries(no_answer, float(value), x_max, y_min, y_max)
        else:
            plot_decision_boundaries(yes_answer, x_min, x_max, y_min, float(value))
            plot_decision_boundaries(no_answer, x_min, x_max, float(value), y_max)
        
    # "tree" is a leaf
    else:
        plt.fill_between(x=[x_min, x_max], y1=y_min, y2=y_max, alpha=0.2, color=color_keys[tree])
    
    return


def create_plot(df, tree=None, title=None):
    
    sns.lmplot(data=df, x="x", y="y", hue="label", 
               fit_reg=False, height=4, aspect=1.5, legend=False)
    plt.title(title)
    
    if tree or tree == False: # root of the tree might just be a leave with "False"
        x_min, x_max = round(df.x.min()), round(df.x.max())
        y_min, y_max = round(df.y.min()), round(df.y.max())

        plot_decision_boundaries(tree, x_min, x_max, y_min, y_max)
    
    return

# 1. Decision Tree helper functions
# 1.1 Data pure?
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

    
# 1.2 Create Leaf
def create_leaf(data, ml_task):
    
    label_column = data[:, -1]
    if ml_task == "regression":
        leaf = np.mean(label_column)
        
    # classfication    
    else:
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    
    return leaf


# 1.3 Determine potential splits
def get_potential_splits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1): # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


# 1.4 Determine Best Split
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


def calculate_mse(data):
    actual_values = data[:, -1]
    if len(actual_values) == 0:   # empty data
        mse = 0
        
    else:
        prediction = np.mean(actual_values)
        mse = np.mean((actual_values - prediction) **2)
    
    return mse


def calculate_overall_metric(data_below, data_above, metric_function):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_metric =  (p_data_below * metric_function(data_below) 
                     + p_data_above * metric_function(data_above))
    
    return overall_metric


def determine_best_split(data, potential_splits, ml_task):
    
    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            
            if ml_task == "regression":
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_mse)
            
            # classification
            else:
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function=calculate_entropy)

            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


# 1.5 Split data
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


# 2. Decision Tree Algorithm
# 2.1 Helper Function
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# 2.2 Algorithm
def decision_tree_algorithm(df, ml_task, counter=0, min_samples=2, max_depth=5):
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)
        return leaf

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits, ml_task)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, ml_task, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, ml_task, counter, min_samples, max_depth)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree


# 3. Make predictions
# 3.1 One example
def predict_example(example, tree):
    
    # tree is just a root node
    if not isinstance(tree, dict):
        return tree
    
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)

    
# 3.2 All examples of a dataframe
def make_predictions(df, tree):
    
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        # "df.apply()"" with empty dataframe returns an empty dataframe,
        # but "predictions" should be a series instead
        predictions = pd.Series()
        
    return predictions


# 3.3 Accuracy
def calculate_accuracy(df, tree):
    predictions = make_predictions(df, tree)
    predictions_correct = predictions == df.label
    accuracy = predictions_correct.mean()
    
    return accuracy


random.seed(0)

column_list = [ 9,10,11,12,26,28,29,30,31,32,33,35,36,45,46,47,64,65,66,67,68,69,70,71,72,73,74,75,76,77 ]

# ----------------------------------------------------------------------------------------------------------

#Lectura de Train con Pandas
df = pd.read_csv("0_train_balanced_15000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("0_test_balanced_5000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

# ----------------------------------------------------------------------------------------------------------

print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#   train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=3)

read_time = (time.time() - start_time)
finalHeap = memory_usage_psutil() - startHeap
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 0 - Depth 3 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)

start_time = time.time()
startHeap = memory_usage_psutil()


tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)

read_time = (time.time() - start_time)
finalHeap = memory_usage_psutil() - startHeap
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 0 - Depth 5 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)



# ----------------------------------------------------------------------------------------------------------

"""

print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#Lectura de Train con Pandas
df = pd.read_csv("1_train_balanced_45000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("1_test_balanced_15000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

#train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)

read_time = (time.time() - start_time)
finalHeap = memory_usage_psutil() - startHeap
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 1 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)

# ----------------------------------------------------------------------------------------------------------
print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#Lectura de Train con Pandas
df = pd.read_csv("2_train_balanced_75000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("2_test_balanced_25000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

#train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)
finalHeap = memory_usage_psutil() - startHeap
read_time = (time.time() - start_time)
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 2 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)

# ----------------------------------------------------------------------------------------------------------
print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#Lectura de Train con Pandas
df = pd.read_csv("3_train_balanced_105000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("3_test_balanced_35000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

#train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)
read_time = (time.time() - start_time)
finalHeap = memory_usage_psutil() - startHeap
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 3 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)

# ----------------------------------------------------------------------------------------------------------
print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#Lectura de Train con Pandas
df = pd.read_csv("4_train_balanced_135000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("4_test_balanced_45000.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

#train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)
read_time = (time.time() - start_time)
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 4  ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)

# ----------------------------------------------------------------------------------------------------------
print(" - - ")
print(" --- --- --- --- --- --- --- --- --- ")
print(" - - ")

#Lectura de Train con Pandas
df = pd.read_csv("5_train_balanced_57765.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#train_df = df.values.tolist()
train_df = df

#Lectura de test con Pandas
df = pd.read_csv("5_test_balanced_19255.csv",sep=';',encoding='UTF-8',usecols=column_list)
df = df.rename(columns={"exito": "label"})
#Vamos a llenar los nulos y pasar el sí/no a booleanos
df = df.replace("Si", "SI")
df = df.replace("No", "NO")
df['profundiza'] = df['profundiza'].fillna(0.0)
df['puntaje_prof'] = df['puntaje_prof'].fillna(0.0)
df = df.replace(np.nan, "Desconocido")
#testing_data2 = df.values.tolist()
test_df = df

#train_df,test_df 
start_time = time.time()
startHeap = memory_usage_psutil()

tree = decision_tree_algorithm(train_df, ml_task="classification", max_depth=5)
read_time = (time.time() - start_time)
finalHeap = memory_usage_psutil() - startHeap
accuracy = calculate_accuracy(test_df, tree)

print("--- DataSet 5 ---")
pprint(tree, width=50)
print(accuracy)
print(read_time)
print(finalHeap)



"""
