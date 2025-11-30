# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.backend import clear_session
import deep_learning as project
from function import select
from function import pop_crossover
from function import pop_mutate
from function import pop_generate
from function import pop_check
from Setting import dataset
import math

path_train_fake = dataset.path_train_0
path_train_real = dataset.path_train_1
path_train = [path_train_fake,path_train_real]

path_fit_fake = dataset.path_fit_0
path_fit_real = dataset.path_fit_1
path_fit = [path_fit_fake,path_fit_real]

# Training configuration dictionary (example template)
train_config = {
    "height": 299,          # Input image height 
    "width": 299,           # Input image width 
    "channel_1": None,       # Number of channels for branch 1 
    "channel_2": None,       # Number of channels for branch 2 
    "batch_size": None,      # Training batch size
    "learning_rate": None,   # Learning rate for optimizer 
    "epochs": None,          # Number of training epochs
    "steps_per_epoch": None, # Number of steps per epoch during training
    "val_batch_size": None,  # The size of the fitness subset
}

## Evolution parameters
DNA_SIZE = 2
DNA_SIZE_MAX = 15
POP_SIZE = 80
N_GENERATIONS = 100

def get_fitness(x_1, x_2):
    return project.classify(path_train, path_fit, num_1=x_1, num_2=x_2, feature_len=50, train_config=train_config)

def evolution(pop, each_generation, pop_num_last, pop_num):
    fitness_1 = np.zeros([pop_num_last, ])
    fitness_2 = np.zeros([pop_num_last, ])
    for i in range(pop_num_last):
        
        pop_list_1 = list(pop[0][i])
        pop_list_2 = list(pop[1][i])

        # Trim unused zeros and convert to integers
        for j, each in enumerate(pop_list_1):
            if each == 0.0:
                pop_list_1 = pop_list_1[:j]
                break
        pop_list_1 = [int(each) for each in pop_list_1]

        for j, each in enumerate(pop_list_2):
            if each == 0.0:
                pop_list_2 = pop_list_2[:j]
                break
        pop_list_2 = [int(each) for each in pop_list_2]

        # Fitness calculation
        fitness = get_fitness(pop_list_1, pop_list_2)
        fitness_1[i] = fitness[0]
        fitness_2[i] = fitness[1]
        clear_session()

        print('Generation %d: Individual %d fitness => %.5f, %.5f' % 
              (each_generation, i + 1, fitness_1[i], fitness_2[i]))
        print('Chromosome:', pop_list_1, pop_list_2)

        f_1 = "E:/Structure_Search/record.txt"
        with open(f_1, "a") as file:
            file.write('Generation {}: Individual {} fitness {:.5f}, {:.5f}\n'.format(
                each_generation, i + 1, fitness_1[i], fitness_2[i]))
            file.write('Branch 1: {}\n'.format(pop_list_1))
            file.write('Branch 2: {}\n'.format(pop_list_2))

    # Evolution operations
    pop = select(pop, 0.001, fitness_1, fitness_2, pop_num_last, pop_num)
    pop_new = pop.copy()
    child = pop_crossover(pop_new, 0.9, pop_num, mu=20)
    child = pop_mutate(child, pop_num, mu=20)
    child = pop_check(child, pop_num)

    # Record evolution results
    with open(f_1, "a") as file:
        file.write('Generation {} result:\n'.format(each_generation + 1))
        file.write('Branch 1:\n{}\n'.format(child[0]))
        file.write('Branch 2:\n{}\n'.format(child[1]))
        
    return child


# Initialize population
pop = pop_generate(POP_SIZE, DNA_SIZE, DNA_SIZE_MAX)

for each_generation in range(N_GENERATIONS):
    if each_generation == 0:
        pop = evolution(pop, each_generation, 80, 60)
    else:
        pop = evolution(pop, each_generation, 60, 60)














