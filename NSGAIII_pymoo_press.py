#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:04:45 2022

@author: alexandre
"""
import numpy as np
from pack_nsga_press import *
from multiprocessing import cpu_count
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_reference_directions
from pymoo.factory import get_sampling, get_mutation, get_crossover
from pymoo.operators.mixed_variable_operator import MixedVariableSampling
from pymoo.operators.mixed_variable_operator import MixedVariableMutation
from pymoo.operators.mixed_variable_operator import MixedVariableCrossover
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import sys


home = str(Path.home())

#Find optimal pop size
cores = cpu_count()/4
local_pop_size = 4 
minimal_pop =  120# based on reference directions
while minimal_pop * local_pop_size % cores != 0:
    minimal_pop += 1
pop_size = minimal_pop

flock_model = int(sys.argv[1])
comm_type = int(sys.argv[2])
obst_file = sys.argv[3] # forest ; calib ; city

# Obstacle type 
obst_type = obst_file.split('/')[-1].split('.')[0]
# Number of objective functions
n_obj = 7

# Termination criteria (format hh:mm:ss)
time_to_run = sys.argv[4]
# Date of today
d = sys.argv[5]
# Folder internal name
folder = d + '_' + obst_type + '_' + str(flock_model) + '_' + str(comm_type)
 
file_to_save = d + '_' + obst_type + '_' + str(flock_model) + '_comm_' + str(comm_type) + '.pkl' 
unit_file = modify_unitparams_file(flock_model, comm_type, home + '/optim/robotsim-master/parameters/unitparams.dat')


if flock_model == 0:
    #lower_bounds = np.array([5, 5130, 10, 0, 0, 0, 0.26, 940, 125, 0.02, 0, 5])
    #upper_bounds = np.array([188, 9000, 1510, 0.03, 10, 1000, 10, 1500, 999, 0.3, 0.01, 10])
    #lower_bounds = np.array([5, 7000, 100, 0, 2, 100, 3, 600, 200, 0.1, 0, 5])
    #upper_bounds = np.array([100, 11000, 1000, 0.02, 10, 1000, 10, 1000, 1000, 0.4, 0.1, 10])
    lower_bounds = np.array([5, 1000, 300, 0, 2, 100, 3, 850, 600, 0.1, 0, 4])
    upper_bounds = np.array([150, 4000, 800, 0.02, 20, 1000, 10, 1400, 1200, 1, 0.15, 8])
    n_var = 12
    mask = ["real", "real", "real", "real", "real", "real", 
            "real", "real", "real", "real", "real", "int"]
    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })
    crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx"),
    "int": get_crossover("int_sbx")
    })

    mutation = MixedVariableMutation(mask, {
    #"real": get_mutation("real_pm", prob=0.5, eta=2),
    #"int": get_mutation("int_pm", prob=0.5, eta=2)
    "real": get_mutation("real_pm"),
    "int": get_mutation("int_pm")
    })

elif flock_model == 1:
    #lower_bounds = np.array([5, 7000, 100, 0.1, 0, 2, 100, 3, 600, 200, 0.1, 5])
    #upper_bounds = np.array([100, 11000, 1000, 0.8, 0.02, 10, 1000, 10, 1000, 1000, 0.4, 10])
    lower_bounds = np.array([5, 1000, 100, 0.25, 0, 2, 100, 3, 850, 200, 0.1, 4])
    upper_bounds = np.array([150, 4000, 800, 0.8, 0.02, 20, 1000, 10, 1300, 1200, 1, 8])
    n_var = 12
    mask = ["real", "real", "real", "real", "real", "real", 
            "real", "real", "real", "real", "real", "int"]
    sampling = MixedVariableSampling(mask, {
        "real": get_sampling("real_random"),
        "int": get_sampling("int_random")
    })
    crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx"),
    "int": get_crossover("int_sbx")
    })

    mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm"),
    "int": get_mutation("int_pm")
    #"real": get_mutation("real_pm", prob=0.6, eta=2),
    #"int": get_mutation("int_pm", prob=0.6, eta=2)
    })

if __name__ == '__main__':

    problem = Myproblem(flock_model, pop_size, local_pop_size, n_var, n_obj,
                        lower_bounds, upper_bounds, unit_file, obst_file, folder)
    

    ref_dirs = get_reference_directions("energy", n_obj, 120, seed=1)

    algorithm = NSGA3(ref_dirs = ref_dirs,
                      pop_size= pop_size,
                      sampling=sampling,
                      mutation=mutation,
                      crossover=crossover)
    
    
    termination = get_termination("time", time_to_run)
    #termination = get_termination("n_gen", 2)
    res = minimize(problem, 
                   algorithm,
                   termination,
                   display= MyDisplay(),
                   seed = 1,
                   save_history = True,
                   verbose = True)
    
    save_object(res, file_to_save)
