#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:40:53 2022

@author: alexandre
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 17:33:44 2020

@author: alexandre
"""
import os
from math import cos, exp, pi, floor
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from pymoo.core.problem import Problem
from pymoo.util.display import Display
import pickle
from pathlib import Path
import time


home = str(Path.home())
start = time.time()

num_cores = multiprocessing.cpu_count()
# num_cores = 1
class MyDisplay(Display):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        end = time.time()
        self.output.append("time", (end - start)/60)
        self.output.append("n_nds", len(algorithm.opt))
        self.output.append("Mean Obj", np.mean(algorithm.opt.get("F"), axis=0))
        self.output.append("Std Obj", np.std(algorithm.opt.get("F"), axis=0))

class Myproblem(Problem):
    def __init__(self, flock_model, pop_size, inner_pop_size, 
                 n_var, n_obj, lower_bounds, upper_bounds, 
                 unit_file, obst_file, folder,  **kwargs):
        super().__init__(n_var = n_var,
                         n_obj = n_obj,
                         n_constr = 0,
                         xl = lower_bounds,
                         xu = upper_bounds,
                         elementwise_evaluation = None,
                         **kwargs)
        
        self.flock_model = flock_model
        self.pop_size    = pop_size
        self.inner_pop_size = inner_pop_size
        self.unit_file = unit_file
        self.obst_file = obst_file
        self.folder = folder
               
    
    def _evaluate(self, x, out, *args, **kwargs):
        results = Parallel(n_jobs=num_cores, verbose=0)\
            (delayed(eval_func)(x[floor(i/self.inner_pop_size)], i, self.flock_model, 
                                self.unit_file, self.obst_file, self.folder)\
              for i in range(self.pop_size * self.inner_pop_size))
        out["F"] = []
        for j in range(0, self.pop_size * self.inner_pop_size, self.inner_pop_size):
            batch = results[j:j + self.inner_pop_size]
            obj = np.median(batch, axis=0).tolist()
            std = np.std(batch, axis=0)
            norm = np.linalg.norm(std)
            obj.append(norm)
            out["F"].append(obj)

def modify_unitparams_file(flocking_model, comm_type, path_to_file):
    if os.path.isfile(path_to_file):
        file = open(path_to_file, 'r')
        name = path_to_file.split('.')[0] + '_comm_' + str(comm_type)\
            + '_flock_' + str(flocking_model) + '.dat'
        out  = open(name, 'w')
        lines = file.readlines()
        for line in lines:
            if 'Flocking_type' in line:
                new_val = line.split('=')[0] + '=' + str(flocking_model) + '\n'
                out.write(new_val)
            elif 'Communication_type' in line:
                new_val = line.split('=')[0] + '=' + str(comm_type) + '\n'
                out.write(new_val)
            else:
                out.write(line)
        file.close()
        out.close()       
        return name

def write_file_from_vector(x, output_file, original_file, flocking_model):
    if os.path.isfile(original_file):
        to_read = open(original_file, 'r')
        out_file = open(output_file, 'w')
        lines = to_read.readlines()
        if flocking_model == 0:
            for line in lines:
                if 'V_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[0]) +'\n'
                    out_file.write(to_write)
                    
                elif 'R_0_Offset_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[1]) +'\n'
                    out_file.write(to_write)
                    
                elif 'R_0_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[2]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Slope_Att' in line:
                    to_write = line.split('=')[0] + '=' + str(x[3]) +'\n'
                    out_file.write(to_write)
                
                elif 'Slope_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[4]) +'\n'
                    out_file.write(to_write)                         
                    
                elif 'Acc_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[5]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Slope_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[6]) +'\n'
                    out_file.write(to_write)
                    
                elif 'V_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[7]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Acc_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[8]) +'\n'
                    out_file.write(to_write)
                    
                elif 'C_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[9]) +'\n'
                    out_file.write(to_write)
                    
                elif 'K_Press' in line:
                    to_write = line.split('=')[0] + '=' + str(x[10]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Size_Neighbourhood' in line:
                    to_write = line.split('=')[0] + '=' + str(x[11]) +'\n'
                    out_file.write(to_write)
        
                else:
                    out_file.write(line)
        
        elif flocking_model == 1:
            for line in lines:
                if 'V_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[0]) +'\n'
                    out_file.write(to_write)
                    
                elif 'R_0_Offset_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[1]) +'\n'
                    out_file.write(to_write)
                    
                elif 'R_0_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[2]) +'\n'
                    out_file.write(to_write)
                
                elif 'Slope_Rep' in line:
                    to_write = line.split('=')[0] + '=' + str(x[3]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Slope_Att' in line:
                    to_write = line.split('=')[0] + '=' + str(x[4]) +'\n'
                    out_file.write(to_write)
                
                elif 'Slope_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[5]) +'\n'
                    out_file.write(to_write)                         
                    
                elif 'Acc_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[6]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Slope_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[7]) +'\n'
                    out_file.write(to_write)
                    
                elif 'V_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[8]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Acc_Shill' in line:
                    to_write = line.split('=')[0] + '=' + str(x[9]) +'\n'
                    out_file.write(to_write)
                    
                elif 'C_Frict' in line:
                    to_write = line.split('=')[0] + '=' + str(x[10]) +'\n'
                    out_file.write(to_write)
                    
                elif 'Size_Neighbourhood' in line:
                    to_write = line.split('=')[0] + '=' + str(x[11]) +'\n'
                    out_file.write(to_write)
                    
                else:
                    out_file.write(line)
        else:
            raise Exception('This flocking mode is not ready for optimization')
        to_read.close()
        out_file.close()
        
def sigmoid(x, x0, d):
        if x < x0 - d:
            value = 1
        elif x >= x0 - d and x < x0:
            value = 0.5 * (1 - cos(pi/d * (x - x0)))
        else:
            value = 0
        return value
    
def heaviside(x):
    if x > 0:
        return 1
    else:
        return 0

def func1(phi, phi0, d):
    return (1 - sigmoid(phi, phi0, d))

def func2(phi, s):
    return exp(-(phi/s)**2)

def func3(phi, a):
    return (a**2/(a + phi)**2)

        
def eval_func(x, i, flock_model, unit_file, obst_file, folder):
        result = []
        flock_path = home + '/optim/robotsim-master/parameters/flocking_folder/' + folder + '/flockingparams' + str(i) + '.dat'
        original_flock_file = home + '/optim/robotsim-master/parameters/flockingparams.dat'
        write_file_from_vector(x, flock_path, original_flock_file, flock_model)
        output_path = home + '/optim/robotsim-master/output_folder/' + folder +'/output_default' + str(i) 
        cmd = home + '/optim/robotsim-master/robotflocksim_main_server \
                  -i ' + home + '/optim/robotsim-master/parameters/initparams.dat \
                  -o ' + output_path + ' \
                  -verb 0 \
                  -c ' + home + '/optim/robotsim-master/config/defaultcolors.ini \
                  -outputconf ' + home + '/optim/robotsim-master/config/output_config.ini \
                  -u ' + unit_file + ' \
                  -arena ' + home + '/optim/robotsim-master/parameters/arenas.default \
                  -obst ' + obst_file + ' \
                  -f ' + flock_path +' -novis'
        os.system(cmd)
        
        ff = open(flock_path, 'r')
        vflock = float(ff.read().split()[14].split('=')[1])
        ff = open(flock_path, 'r')
        d_target = float(ff.read().split()[72].split('=')[1])
        ff.close()
        
        vtol = (1.5/4) * vflock
        atol = 0.00003
        rtol = 100
        ptol = 100
        dtol=200
        
        ff = open(home +'/optim/robotsim-master/parameters/initparams.dat', 'r')
        numb_agent = float(ff.read().split()[4].split('=')[1])
        ff.close()
        
        file = output_path + '/cluster_dependent_correlation.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            cluster_dependent_correlation_avg = float(f.read().split()[6])
            f.close()
            
        file = output_path + '/cluster_parameters.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            agent_not_in_cluster = float(f.read().split()[8])
            f.close()
        
        file = output_path + '/collision_ratio.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            ratio_of_collisions = float(f.read().split()[3])
            f.close()

        
        file = output_path + '/pressure.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            pressure = float(f.read().split()[21])
            f.close()
        

        file = output_path + '/distance_from_arena.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            dists = f.read().split()
            #distance_from_arena = float(dists[7]) + float(dists[-2])
            distance_from_arena = float(dists[-1])
            f.close()
        
        file = output_path + '/dist_between_neighbours.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            dist_between_neighbours = float(f.read().split()[17])
            f.close()
        
        file = output_path + '/velocity.dat'
        if os.path.isfile(file):
            f = open(file, 'r')
            velocity = float(f.read().split()[37])
            f.close()
    
        Fspeed = func1(velocity, vflock, vtol)
        Fcoll  = func3(ratio_of_collisions, atol)
        Fdisc  = func3(agent_not_in_cluster, numb_agent / 2)
        #Fwall  = func2(distance_from_arena, rtol)
        Fwall  = func3(distance_from_arena, 3)
        Fcorr  = heaviside(cluster_dependent_correlation_avg) * cluster_dependent_correlation_avg
        Fpress = func2(pressure, ptol)
        Fdist = func1(dist_between_neighbours, d_target, dtol)
        
        result.append(-Fspeed)
        result.append(-Fcoll)
        result.append(-Fdisc)
        result.append(-Fwall)
        result.append(-Fcorr)
        if flock_model == 0:
            result.append(-Fpress)
        else :
            result.append(-Fdist)
                
        return result

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
