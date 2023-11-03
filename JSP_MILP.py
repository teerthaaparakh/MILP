#!/usr/bin/env python
#written by Anal Parakh
import MILPconfig as cfg
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import shutil
import sys
import os.path

MAX_COLORS = cfg.parameters["max_colors"]
colors = sns.color_palette("hls", MAX_COLORS)

NUM_CATAPULTS = cfg.parameters["num_catapults"]
v = cfg.parameters["v"]
num_planes = cfg.parameters["num_planes"]
num_ops = 14
num_phases = 10
num_positions = num_planes
K_flying = num_planes
K_greenm = cfg.parameters["K_greenm"]
K_red = cfg.parameters["K_red"]
K_grape = cfg.parameters["K_grape"]
K_brown = num_planes-2
K_blue = cfg.parameters["K_blue"]
K_yellow = cfg.parameters["K_yellow"]
K_green_c_op = cfg.parameters["K_green_c_op"]
K_green_c_w = cfg.parameters["K_green_c_w"]
K_yellow_c_of = cfg.parameters["K_yellow_c_of"]
K_catapult = NUM_CATAPULTS
K_landing = cfg.parameters["K_landing"]
t_greenm = cfg.parameters["t_greenm"]
t_red = cfg.parameters["t_red"]
t_grape = cfg.parameters["t_grape"]
t_blue = cfg.parameters["t_blue"]
t_yellow = cfg.parameters["t_yellow"]
t_greenc = cfg.parameters["t_greenc"]
t_greenop = cfg.parameters["t_greenop"]
t_yellowof = cfg.parameters["t_yellowof"]
t_flying = cfg.parameters["t_flying"]
t_resting_gear = cfg.parameters["t_resting_gear"]
t_landing = cfg.parameters["t_landing"]
t_parking = cfg.parameters["t_parking"]
t_chaining = cfg.parameters["t_chaining"]
t_weapon = cfg.parameters["t_weapon"]

phase_names = ['Startup', 'chocks', 'direct', 'catapult', 'fly',
               'land_gear', 'land', 'park', 'chain', 'unload']

operations = ['maintain', 'ordnance', 'fuel',
              'chocks', 'direct',
              'c_operate', 'w_check',
              'c_officer', 'flying', 'landing_gear', 'landing',
              'parking', 'chaining', 'unloading']

t_total = [t_greenm, t_red , t_grape,
           t_blue, t_yellow, t_greenc, t_greenop,
           t_yellowof, t_flying, t_resting_gear,
           t_landing, t_parking, t_chaining, t_weapon]

machines = ['maintainer', 'ordnant', 'fueller',
            'chocker', 'director', 'c_operator','w_checker',
            'c_officer', 'fly', 'land'] #, 'catapult']

num_machines = [K_greenm, K_red , K_grape,
                K_blue, K_yellow, K_green_c_op, K_green_c_w,
                K_yellow_c_of, K_flying, K_landing] #, K_catapult]

ops_to_phase = [0,0,0,1,2,3,3,3,4,5,6,7,8,9]
ops_to_machines = [0,1,2,3,4,5,6,7,8,0,9,0,3,1]

machine_to_ops = [[0, 9, 11], [1, 13], [2], [3, 12], [4], [5], [6], [7], [8], [10]]
machine_num_ops = [len(i) for i in machine_to_ops]

cata_phase_start = 3
cata_phase_end = 4

cap_phase_start = 1
cap_phase_end = 8

land_phase_start = 5
land_phase_end = 7

model = gp.Model('ConstraintOptimization')
# model.setParam('FeasibilityTol', 1e-4)
# model.setParam('IntFeasTol', 1e-4)

# Known
O = np.zeros((num_phases, num_ops))
O[0, 0:3] = 1
O[1, 3] = 1
O[2, 4] = 1
O[3, 5:8] = 1
O[4, 8] = 1
O[5, 9] = 1
O[6, 10] = 1
O[7, 11] = 1
O[8, 12] = 1
O[9, 13] = 1

# Known
t_dict = {}
for i in range(len(operations)):
    operation = operations[i]
    machine_type = ops_to_machines[i]
    len_positions = machine_num_ops[machine_type]*num_planes
    t_dict[operation] = np.zeros((num_planes, num_machines[ops_to_machines[i]], len_positions))
    t_dict[operation][:, :, :] = t_total[i]

# Variables
Z_dict = {}
for i in range(len(operations)):
    machine_type = ops_to_machines[i]
    len_positions = machine_num_ops[machine_type]*num_planes
    Z_dict[operations[i]] = model.addMVar(shape = (num_planes,
                                                   num_machines[ops_to_machines[i]],
                                                   len_positions),  #num_positions
                                    vtype=GRB.BINARY, name = 'Z_'+operations[i])

Z_catapult = model.addMVar(shape = (num_planes, K_catapult, num_planes), vtype=GRB.BINARY,
                           name = 'Z_catapult')
Z_captain = model.addMVar(shape = (num_planes, K_brown, num_planes), vtype=GRB.BINARY,
                           name = 'Z_captain')
Z_land =  model.addMVar(shape = (num_planes, num_planes), vtype=GRB.BINARY,
                           name = 'Z_land')

M_dict = {}
for i in range(len(machines)):
    M_dict[machines[i]] = model.addMVar(shape = (num_machines[i],
                                                 machine_num_ops[i]*num_planes + 1),
                                        name = 'M_'+machines[i])
    
M_catapult = model.addMVar(shape = (K_catapult, num_positions + 1),
                           name = 'M_catapult')
M_captain = model.addMVar(shape = (K_brown, num_positions + 1),
                           name = 'M_captain')
M_land = model.addMVar(shape = (num_positions + 1),
                           name = 'M_land')
    
B_iJ = model.addMVar(shape = (num_planes, num_phases+1), name = 'B_iJ')
C_i = model.addMVar(shape = num_planes, name = 'C_i')
C_max = model.addVar(name = 'C_max')

t_combined = {}
for i in range(len(operations)):
    machine_type = ops_to_machines[i]
    len_positions = machine_num_ops[machine_type]*num_planes
    t_combined[operations[i]] = model.addMVar(shape = (num_planes,
                                                       num_machines[ops_to_machines[i]],
                                                       len_positions))


def get_t_combined(model):
    for plane_id in range(num_planes):
        for op in range(num_ops):
            machine_type = ops_to_machines[op]
            len_positions = machine_num_ops[machine_type]*num_planes
            for machine_id in range(num_machines[machine_type]):
                for pos in range(len_positions):
                    model.addConstr(t_combined[operations[op]][plane_id, machine_id, pos] == \
                                    t_dict[operations[op]][plane_id, machine_id, pos] * \
                                    Z_dict[operations[op]][plane_id, machine_id, pos])
    return model


def catapult_constraint(model):

    """catapult 1 and 2 cannot be used at the same time, similarly catapult 3 and 4 cannot be  used at the same time."""
    
    y1 = model.addMVar(shape = (num_positions, num_positions), vtype=GRB.BINARY)
    y2 = model.addMVar(shape = (num_positions, num_positions), vtype=GRB.BINARY)
    y3 = model.addMVar(shape = (num_positions, num_positions), vtype=GRB.BINARY)
    y4 = model.addMVar(shape = (num_positions, num_positions), vtype=GRB.BINARY)


    for pos1 in range(num_positions):
        for pos2 in range(num_positions):
            s0 = M_catapult[0, pos1]
            e0 = M_catapult[0, pos1+1]
            s1 = M_catapult[1, pos2]
            e1 = M_catapult[1, pos2+1]
            
            model.addConstr(y1[pos1, pos2] + y2[pos1, pos2] == 1)
            model.addConstr(-v*(1-y1[pos1, pos2]) + (e1-s0) <= 0)
            model.addConstr(-v*(1-y2[pos1, pos2]) + (e0-s1) <= 0)

    for pos1 in range(num_positions):
        for pos2 in range(num_positions):
            s0 = M_catapult[2, pos1]
            e0 = M_catapult[2, pos1+1]
            s1 = M_catapult[3, pos2]
            e1 = M_catapult[3, pos2+1]
            
            model.addConstr(y3[pos1, pos2] + y4[pos1, pos2] == 1)
            model.addConstr(-v*(1-y3[pos1, pos2]) + (e1-s0) <= 0)
            model.addConstr(-v*(1-y4[pos1, pos2]) + (e0-s1) <= 0)
            
    return model

def makespan_constraint(model):
    """C_max is the maximum of makespan of all jobs"""
    for i in range(num_planes):
        model.addConstr(C_i[i] <= C_max)
    return model
    
    
def constraint_1(model):
    """ only 1 machine and 1 position can be assigned to a particular operation of a job"""
    for plane_id in range(num_planes):
        for op in operations:
            model.addConstr(Z_dict[op][plane_id, :, :].sum() == 1)
        model.addConstr(Z_catapult[plane_id, :, :].sum() == 1)
        model.addConstr(Z_captain[plane_id, :, :].sum() == 1)
        model.addConstr(Z_land[plane_id, :].sum() == 1)
        
    return model
        
def constraint_2(model):
    """next phase of ith plane starts only after previous phase has ended"""
    for op_id, op_name in enumerate(operations): # (K)
        for plane_id in range(num_planes): # (i)
            for phase_no in range(num_phases): #(J)
                
                model.addConstr(O[phase_no, op_id]*t_combined[op_name][plane_id, :, :].sum() <= \
                                (B_iJ[plane_id, phase_no+1] - B_iJ[plane_id, phase_no]))
    return model
    
def constraint_3(model):
    """ending time of last phase should be less than equal to the makespan of a particular job"""
    for i in range(num_planes):
        model.addConstr(B_iJ[i, num_phases] <= C_i[i])
        for phase in range(num_phases):
            model.addConstr(B_iJ[i, phase] <= B_iJ[i, phase+1])
    return model
    
def constraint_4(model):
    """there will only be atmost plane on a given machine and at a given position"""
    for i in range(len(machines)):
        allowed_ops = machine_to_ops[i]
        len_positions = machine_num_ops[i]*num_planes
        for machine_id in range(num_machines[i]):
            for pos in range(len_positions):
                result1 = 0
                result2 = 0
                for op in allowed_ops:
                    result1 += Z_dict[operations[op]][:,machine_id, pos].sum()
                    if (pos < (len_positions-1)):
                        result2 += Z_dict[operations[op]][:,machine_id, pos+1].sum()
                model.addConstr(result1 <= 1)
                model.addConstr(result1 >= result2)
                
    #catapult
    for cata_id in range(K_catapult):
        for pos in range(num_positions):
            model.addConstr(Z_catapult[:, cata_id, pos].sum() <= 1)
            if (pos < (num_positions-1)):
                model.addConstr(Z_catapult[:, cata_id, pos].sum() >= \
                               Z_catapult[:, cata_id, pos+1].sum())
                
    #land
    for pos in range(num_positions):
        model.addConstr(Z_land[:, pos].sum() <= 1)
        if (pos < (num_positions-1)):
            model.addConstr(Z_land[:, pos].sum() >= \
                           Z_land[:, pos+1].sum())

    # captain
    for cap_id in range(K_brown):
        for pos in range(num_positions):
            model.addConstr(Z_captain[:, cap_id, pos].sum() <= 1)
            if (pos < (num_positions-1)):
                model.addConstr(Z_captain[:, cap_id, pos].sum() >= \
                               Z_captain[:, cap_id, pos+1].sum())
    return model
    
def constraint_5(model):
    """length of position should be atleast the time required to finishing a particular operation"""
    for i in range(len(machines)):
        allowed_ops = machine_to_ops[i]
        len_positions = machine_num_ops[i]*num_planes
        for machine_id in range(num_machines[i]):
            for pos in range(len_positions):
                m = M_dict[machines[i]]
                for op in allowed_ops:
                    for plane_id in range(num_planes):
                        model.addConstr((m[machine_id, pos+1] - m[machine_id, pos]) >= \
                                    t_combined[operations[i]][plane_id, machine_id, pos])
    return model
                                    
def constraint_6(model):
    """Bounding Machines finishing time"""
    for machine_type in range(len(machines)):
        len_positions = machine_num_ops[machine_type]*num_planes
        for machine_id in range(num_machines[machine_type]):
            M_last = M_dict[machines[machine_type]][machine_id, len_positions]
            model.addConstr(M_last <= C_max)
            
    for catapult_id in range(K_catapult):
        M_last_cata = M_catapult[catapult_id, num_positions]
        model.addConstr(M_last_cata <= C_max)

    for cap_id in range(K_brown):
        M_last_cap = M_captain[cap_id, num_positions]
        model.addConstr(M_last_cap <= C_max)
        
    return model

def constraint_6(model):
    # all machines except catapult
    for machine_type in range(len(machines)): # no of personnel types
        len_positions = machine_num_ops[machine_type]*num_planes
        for plane_id in range(num_planes):
            for machine_id in range(num_machines[machine_type]):
                for position in range(len_positions):
                    M_kp1 = M_dict[machines[machine_type]][machine_id, position]
                    M_kp2 = M_dict[machines[machine_type]][machine_id, position+1]
                    
                    allowed_ops = machine_to_ops[machine_type]
                    for op in allowed_ops:
                        
                        phase = ops_to_phase[op]

                        Z_ijkp = Z_dict[operations[op]][plane_id, machine_id, position]
                        t_ijkp = t_dict[operations[op]][plane_id, machine_id, position]

                        B1 = B_iJ[plane_id, phase]
                        B2 = B_iJ[plane_id, phase+1]

                        model.addConstr(M_kp2 - B1 - t_ijkp + v*(1-Z_ijkp) >= 0)

                        model.addConstr(B2 - M_kp1 - t_ijkp + v*(1-Z_ijkp) >= 0)
                        
    return model
    
def constraint_7(model):
    """for catapult"""
    for plane_id in range(num_planes):
        for cata_id in range(K_catapult):
            for position in range(num_positions):
                M_kp1 = M_catapult[cata_id, position]
                M_kp2 = M_catapult[cata_id, position+1]
                
                Z_ijkp = Z_catapult[plane_id, cata_id, position]

                B1 = B_iJ[plane_id, cata_phase_start]
                B2 = B_iJ[plane_id, cata_phase_end]

                model.addConstr(B1 - M_kp1 + v*(1-Z_ijkp) >= 0)

                model.addConstr(M_kp2 - B2 + v*(1-Z_ijkp) >= 0)
    return model
                
def constraint_8(model):
    """for landing"""
    for plane_id in range(num_planes):
        for position in range(num_positions):
            M_kp1 = M_land[position]
            M_kp2 = M_land[position+1]
            Z_ijkp = Z_land[plane_id, position]
            B1 = B_iJ[plane_id, land_phase_start]
            B2 = B_iJ[plane_id, land_phase_end]

            model.addConstr(B1 - M_kp1 + v*(1-Z_ijkp) >= 0)

            model.addConstr(M_kp2 - B2 + v*(1-Z_ijkp) >= 0)
            
    return model
    
def constraint_9(model):
    # for captain
    for plane_id in range(num_planes):
        for cap_id in range(K_brown):
            for position in range(num_positions):
                M_kp1 = M_captain[cap_id, position]
                M_kp2 = M_captain[cap_id, position+1]
                
                Z_ijkp = Z_captain[plane_id, cap_id, position]

                B1 = B_iJ[plane_id, cap_phase_start]
                B2 = B_iJ[plane_id, cap_phase_end]

                model.addConstr(B1 - M_kp1 + v*(1-Z_ijkp) >= 0)

                model.addConstr(M_kp2 - B2 + v*(1-Z_ijkp) >= 0)
    return model

############ PLOTTING
def get_machine_and_pos(plane_id, op):
    array = np.round(Z_dict[operations[op]].X[plane_id, :, :])
    result = np.where(array == 1)
    if not (len(result[0]) == 1):
        return False
    return (result[0][0], result[1][0])

def get_ops_start_end(plane_id, op_id):
    mac_idx, pos_idx = get_machine_and_pos(plane_id, op_id)
    pdx = ops_to_phase[op_id]
    phase_array = B_iJ.X
    machine_type = ops_to_machines[op_id]
    machine_time_array = M_dict[machines[machine_type]].X
    if op_id == 8:
        _,_, xs, _, xf = get_phase_start_end(plane_id, pdx)
        return mac_idx, pos_idx, xs, xf, xf
    xs = max(phase_array[plane_id, pdx], machine_time_array[mac_idx, pos_idx])
    xf = min(phase_array[plane_id, pdx+1], machine_time_array[mac_idx, pos_idx+1])
    t = t_total[op_id]
    xf = min(xf, xs + t)
    return mac_idx, pos_idx, xs, xs + t, xf

def get_phase_start_end(plane_id, pdx, B = False):
    machine_starts = []
    machine_ends = []
    true_ends = []

    phase_array = B_iJ.X

    phase_start = phase_array[plane_id, pdx]
    phase_end = phase_array[plane_id, pdx+1]
    
    if pdx == 4:
        for op_id in [5,6,7]:
            machine_type = ops_to_machines[op_id]
            machine_time_array = M_dict[machines[machine_type]].X
            mac_idx, pos_idx = get_machine_and_pos(plane_id, op_id)
            machine_end = min(machine_time_array[mac_idx, pos_idx+1], phase_array[plane_id, pdx])
            machine_ends.append(machine_end)
        xs = max(machine_ends)
        xf = phase_end
        return None, None, xs, None, xf
    
    if B:
        return None, None, phase_start, _, phase_end
    
    for op_id in range(len(operations)):
        if ops_to_phase[op_id] == pdx:
            machine_type = ops_to_machines[op_id]
            machine_time_array = M_dict[machines[machine_type]].X
            mac_idx, pos_idx = get_machine_and_pos(plane_id, op_id)
            machine_start = max(machine_time_array[mac_idx, pos_idx], phase_start)
            machine_end = min(machine_time_array[mac_idx, pos_idx+1], phase_end)
            machine_starts.append(machine_start)
            machine_ends.append(machine_end)
            true_ends.append(machine_start + t_total[op_id])
                     
    xs = min(machine_starts)
    xf = max(machine_ends)
    
    true_end = max(true_ends)
    
    return mac_idx, pos_idx, xs, true_end , xf
    

def phase_plot():

    job_names = ['Plane_'+str(plane_id) for plane_id in range(num_planes)]

    makespan_expected = B_iJ.X.max()
#     makespan_obtained = C_max.X
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'black', 'ha':'center', 'va':'center', 'size'   : 8}

    fig, ax = plt.subplots(1,1, figsize=(12, 2+(len(job_names)+len(phase_names))/8))
    
    for jdx, j in enumerate(job_names):
        for pdx, p in enumerate(phase_names):
            _, _, xs, _, xf = get_phase_start_end(jdx, pdx, B = False)
#             xs = phase_array[jdx, pdx]
#             xf = phase_array[jdx, pdx+1]
            ax.plot([xs, xf], [jdx, jdx], c=colors[pdx%7], **bar_style)
            ax.text((xs + xf)/2, jdx, p, **text_style)
                
    ax.set_title('Plane Schedule')
    ax.set_ylabel('Plane')

    idx = 0
    s = job_names
    ax.set_yticks(range(0, len(s)))
    ax.set_yticklabels(s)
    ax.text(makespan_expected, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_expected), ha='center', va='top')
    ax.plot([makespan_expected]*2, ax.get_ylim(), 'g--')
#     ax.text(makespan_obtained, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_obtained), ha='center', va='top')
#     ax.plot([makespan_obtained]*2, ax.get_ylim(), 'r--')
    ax.set_xlabel('Time')
    ax.grid(True)
    plt.savefig('phase_plot.png')
    plt.show()
    fig.tight_layout()
    

def machine_plot(null = False):
    
    makespan_expected = B_iJ.X.max()
    makespan_obtained = 0
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'black', 'ha':'center', 'va':'center', 'size'   : 8}
        
    all_machines = []
    for i in range(len(machines)):
        for machine_id in range(num_machines[i]):
            name = machines[i]+'_'+str(machine_id)
            all_machines.append(name)

    data = {}
    for i in range(len(machines)):
        machine_array = M_dict[machines[i]].X
        for machine_id in range(num_machines[i]):
            len_positions = machine_num_ops[i]*num_planes
            for pos in range(len_positions):
                start = machine_array[machine_id, pos]
                end = machine_array[machine_id, pos+1]
                makespan_obtained = max(makespan_obtained, end)
                data[(i, machine_id, pos)] = ('Null', 'Null', start, end)
                
    for plane_id in range(num_planes):
        for op in range(num_ops):
            m_id, pos, start, _, end = get_ops_start_end(plane_id, op)
            machine_type = ops_to_machines[op]
            data[(machine_type, m_id, pos)] = (plane_id, op, start, end)
   
    fig, ax = plt.subplots(1,1, figsize=(12, 6+(len(all_machines)+num_planes)/4))

    counter = 0
    for i in range(len(machines)):
        for machine_id in range(num_machines[i]):
            len_positions = machine_num_ops[i]*num_planes
            for pos in range(len_positions):
                plane, op, xs, xf = data[(i, machine_id, pos)]
                if plane == 'Null':
                    if not null:
                        continue
                    color = 'k'
                else:
                    color = colors[plane%num_planes]
                ax.plot([xs, xf], [counter]*2, c=color, **bar_style)
                ax.text((xs + xf)/2, counter, 'P'+str(plane)+'_'+operations[op][:4], **text_style)
            counter += 1

    s = all_machines
    ax.set_yticks(range(0, len(s)))
    ax.set_yticklabels(s)
    ax.text(makespan_expected, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_expected), ha='center', va='top')
    ax.plot([makespan_expected]*2, ax.get_ylim(), 'g--')
#     ax.text(makespan_obtained, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_obtained), ha='center', va='top')
#     ax.plot([makespan_obtained]*2, ax.get_ylim(), 'r--')
    ax.set_xlabel('Time')
    ax.grid(True)
    plt.savefig('machine_plot.png')
    plt.show()
    fig.tight_layout()

def ops_plot():
    
    job_names = ['Plane_'+str(plane_id) for plane_id in range(num_planes)]
    ops_names = operations
    makespan_expected = B_iJ.X.max()
#     makespan_obtained = C_max.X
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'black', 'ha':'center', 'va':'center', 'size'   : 8}
  
    fig, ax = plt.subplots(len(job_names),1, figsize=(12, 10+(len(job_names)+len(ops_names))))
    
    for jdx, j in enumerate(job_names):
        for opdx, op in enumerate(ops_names):
            m_idx, _, xs, x_ideal, xf = get_ops_start_end(jdx, opdx)
            m_type = machines[ops_to_machines[opdx]]

            ax[jdx].plot([xs, xf], [opdx, opdx], c=colors[opdx%len(ops_names)], **bar_style)
            ax[jdx].text((xs + xf)/2, opdx, m_type+'_'+str(m_idx), **text_style)

            ax[jdx].text(x_ideal, ax[jdx].get_ylim()[0]-0.2, "{0:0.1f}".format(x_ideal), ha='center', va='top')
            ax[jdx].plot([x_ideal]*2, ax[jdx].get_ylim(), c = colors[opdx%len(ops_names)], linestyle='dashed')
                
        ax[jdx].set_title('Schedule for Plane' + str(jdx))
        ax[jdx].set_ylabel('operations')

        s = ops_names
        ax[jdx].set_yticks(range(0, len(s)))
        ax[jdx].set_yticklabels(s)
        
        ax[jdx].text(makespan_expected, ax[jdx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_expected), ha='center', va='top')
        ax[jdx].plot([makespan_expected]*2, ax[jdx].get_ylim(), 'g--')
#         ax[jdx].text(makespan_obtained, ax[jdx].get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_obtained), ha='center', va='top')
#         ax[jdx].plot([makespan_obtained]*2, ax[jdx].get_ylim(), 'r--')
        
        ax[jdx].set_xlabel('Time')
        ax[jdx].grid(True)
    plt.savefig('ops_plot.png')
    plt.show()
    fig.tight_layout()
    
def get_catapult_and_pos(plane_id):
    array = np.round(Z_catapult.X[plane_id, :, :])
    result = np.where(array == 1)
    if not (len(result[0]) == 1):
        return False
    return result[0][0], result[1][0]

def get_catapult_start_end(plane_id):
    mac_idx, pos_idx = get_catapult_and_pos(plane_id)
    phase_array = B_iJ.X
    machine_time_array = M_catapult.X
    xs = max(phase_array[plane_id, cata_phase_start], machine_time_array[mac_idx, pos_idx])
    xf = min(phase_array[plane_id, cata_phase_end], machine_time_array[mac_idx, pos_idx+1])
#     print(plane_id, mac_idx, pos_idx, xs, xf)
    return mac_idx, pos_idx, xs, xf

def catapult_plot(null = False):
    
    makespan_obtained = C_max.X
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
        
    cata_names = ['Catapult_'+str(cata_id) for cata_id in range(K_catapult)]
    
    data = {}
    
    machine_array = M_catapult.X
    for cata_id in range(len(cata_names)):
        for pos in range(num_positions):
            start = machine_array[cata_id, pos]
            end = machine_array[cata_id, pos+1]
            data[(cata_id, pos)] = ('Null', start, end)
                
    for plane_id in range(num_planes):
            cata_id, pos, xs, xf = get_catapult_start_end(plane_id)
#             m_id, pos = get_machine_and_pos(plane_id, op)
            data[(cata_id, pos)] = (plane_id, xs, xf)
            
    fig, ax = plt.subplots(1,1, figsize=(12, 2+(len(cata_names)+num_planes)/8))

    counter = 0
    
    for cata_id in range(len(cata_names)):
        for pos in range(num_positions):
            plane, xs, xf = data[(cata_id, pos)]
            if plane == 'Null':
                if not null:
                    continue
                color = 'k'
            else:
                color = colors[plane%num_planes]
            ax.plot([xs, xf], [counter]*2, c=color, **bar_style)
            ax.text((xs + xf)/2, counter, 'P'+str(plane)+'_'+str(i), **text_style)
        counter += 1

    s = cata_names
    ax.set_yticks(range(0, len(s)))
    ax.set_yticklabels(s)
#     ax.text(makespan_expected, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_expected), ha='center', va='top')
#     ax.plot([makespan_expected]*2, ax.get_ylim(), 'g--')
    ax.text(makespan_obtained, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_obtained), ha='center', va='top')
    ax.plot([makespan_obtained]*2, ax.get_ylim(), 'r--')
    ax.set_xlabel('Time')
    ax.grid(True)
    plt.savefig('catapult_plot.png')
    plt.show()
    fig.tight_layout()
    
    
def get_captain_and_pos(plane_id):
    array = np.round(Z_captain.X[plane_id, :, :])
#     print(plane_id, array)
    result = np.where(array == 1)
    if not (len(result[0]) == 1):
        return False
    return result[0][0], result[1][0]

def get_captain_start_end(plane_id):
    mac_idx, pos_idx = get_captain_and_pos(plane_id)
    phase_array = B_iJ.X
    machine_time_array = M_captain.X
    xs = max(phase_array[plane_id, cap_phase_start], machine_time_array[mac_idx, pos_idx])
    xf = min(phase_array[plane_id, cap_phase_end], machine_time_array[mac_idx, pos_idx+1])
    return mac_idx, pos_idx, xs, xf

def captain_plot(null = False):
    makespan_obtained = C_max.X
    
    bar_style = {'alpha':1.0, 'lw':25, 'solid_capstyle':'butt'}
    text_style = {'color':'white', 'weight':'bold', 'ha':'center', 'va':'center'}
        
    cap_names = ['Captain_'+str(cap_id) for cap_id in range(K_brown)]
    
    data = {}
    
    machine_array = M_captain.X
    for cap_id in range(len(cap_names)):
        for pos in range(num_positions):
            start = machine_array[cap_id, pos]
            end = machine_array[cap_id, pos+1]
            data[(cap_id, pos)] = ('Null', start, end)
                
    for plane_id in range(num_planes):
            cap_id, pos, xs, xf = get_captain_start_end(plane_id)
#             m_id, pos = get_machine_and_pos(plane_id, op)
            data[(cap_id, pos)] = (plane_id, xs, xf)
            
    fig, ax = plt.subplots(1,1, figsize=(12, 2+(len(cap_names)+num_planes)/8))

    counter = 0
    
    for cap_id in range(len(cap_names)):
        for pos in range(num_positions):
            plane, xs, xf = data[(cap_id, pos)]
            if plane == 'Null':
                if not null:
                    continue
                color = 'k'
            else:
                color = colors[plane%num_planes]
            ax.plot([xs, xf], [counter]*2, c=color, **bar_style)
            ax.text((xs + xf)/2, counter, 'P'+str(plane)+'_'+str(i), **text_style)
        counter += 1

    s = cap_names
    ax.set_yticks(range(0, len(s)))
    ax.set_yticklabels(s)
#     ax.text(makespan_expected, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_expected), ha='center', va='top')
#     ax.plot([makespan_expected]*2, ax.get_ylim(), 'g--')
    ax.text(makespan_obtained, ax.get_ylim()[0]-0.2, "{0:0.1f}".format(makespan_obtained), ha='center', va='top')
    ax.plot([makespan_obtained]*2, ax.get_ylim(), 'r--')
    ax.set_xlabel('Time')
    ax.grid(True)

    fig.tight_layout()


####### MAIN FUNCTION
        
if __name__=='__main__':
    model.setObjective(C_i.sum(), GRB.MINIMIZE)
    model = get_t_combined(model)
    model = catapult_constraint(model)
    model = constraint_1(model)
    model = constraint_2(model)
    model = constraint_3(model)
    model = constraint_4(model)
    model = constraint_5(model)
    model = constraint_6(model)
    model = constraint_7(model)
    model = constraint_8(model)
    model = constraint_9(model)
    
    model.write('OurMILPConstraint.lp')
    model.optimize()
   
    phase_plot()
    machine_plot()
    ops_plot()
    catapult_plot()
                 
             
        

                    
        
        
