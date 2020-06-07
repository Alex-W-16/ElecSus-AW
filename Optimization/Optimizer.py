from __future__ import (division, print_function, absolute_import)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
from datetime import datetime
from elecsus.goal_functions import *


import time
import sys

from elecsus.elecsus_methods_NEW import calculate
from elecsus.libs.durhamcolours import *


def weighted_space(start, stop, num, power):
    '''
    Creates numpy array similar to np.linspace, but with more points
    centrally than at the edges of the range (i.e. not linearly spaced)

    power:      The power the standard linspace is raised to - use odd numbered
                powers to weight grid to middle of range.'''

    M = (start + stop) / 2  # Midpoint of start and stop
    D = stop - M  # Difference between stop and M
    return (np.linspace(-D, D, num) ** power) / (D ** (power - 1)) + M


def rotate_Efield(E_in, Etheta):
    #Rotates standard E_in by Etheta and returns E_in, to be the input electric field in ElecSus
    if Etheta != 0.:
        if E_in.dtype == int:
            E_in = E_in.astype(float)
        if np.array_equal(E_in, [1, 0, 0]):  # standard case
            E_in[0] = np.cos(Etheta)
            E_in[1] = np.sin(Etheta)
        else:
            rotation_matrix = np.array([[np.cos(Etheta), -np.sin(Etheta)],
                                        [np.sin(Etheta), np.cos(Etheta)]])
            if E_in.shape == (3,):
                E_in[:2] = np.dot(rotation_matrix, E_in[:2])
            else:
                for i in range(len(E_in)):
                    E_in[:2, i] = np.dot(rotation_matrix, E_in[:2, i])

    return E_in


def output_transmission(E_out, Etheta):
    '''Uses given output electric field to find output transmission 90 degrees to given Etheta.'''
    E = np.array([1, 0, 0])
    I_in = (E * E.conjugate()).sum(axis=0)
    outputAngle = Etheta + np.pi / 2
    J_out = np.matrix([[np.cos(outputAngle) ** 2, np.sin(outputAngle) * np.cos(outputAngle)],
                       [np.sin(outputAngle) * np.cos(outputAngle), np.sin(outputAngle) ** 2]])
    transmittedE = np.array(J_out * E_out[:2])
    I_out = (transmittedE * transmittedE.conjugate()).sum(axis=0) / I_in

    return I_out


def cascade(Detuning, E_in, p_dict_list, Etheta=0):
    '''Finds the output from directing E_in into a series of one or more cells.

        p_dict_list should be a list of paramater dictionaries that contain parameters for each cell in series

                              #cell 1  #cell 2       #cell n
        i.e.   p_dict_list = [p_dict_1,p_dict_2,...,p_dict_n]

        Will also work for a single cell if p_dict_list has a length of 1.
    '''


    E_in = rotate_Efield(E_in, Etheta)

    for p_dict in p_dict_list:
        #print(p_dict)
        [E_out] = calculate(Detuning, E_in, p_dict, outputs=['E_out'])
        E_in = E_out

    I_out = output_transmission(E_out, Etheta)

    return I_out




def objective_cascade(params, var_names, p_dict_list, Det_base, Det_weight, E_in):
    '''

    Objective function that produces the value used by the optimizer. Called by optimizer().

    params:         List of values chosen by the optimizer for each variable used in the optimization.
                    Last value is ALWAYS Etheta with units of degrees

    var_names:      List of strings that denotes the variable that each value in params is associated with.
                    Each string should be of the form:
                        'Nvariable'
                    where N is the number of the cell and variable should be a valid key for use in a p_dict,
                    which can have a value assigned to it.

    Example of params and var_names usage:
     If params=[80.2, 24.3, 32.5, 88.9] and var_names=['1Bfield', '1T', '2Bfield'],
     then in the first cell, Bfield = 80.2, T=24.3 and in the second cell Bfield=32.5, with Etheta=88.9 degrees.
     This is converted to be used in the p_dict_list.


    p_dict_list:    List of paramater dictionaries that contain parameters for each cell in series.

                                          #cell 1  #cell 2       #cell n
                    i.e.   p_dict_list = [p_dict_1,p_dict_2,...,p_dict_n]
                    Will also work for a single cell if p_dict_list has a length of 1

                    In the optimization, input p_dict_list is from optimize() and should contain default values
                    for variables not being optimized.

    Det_base:       Base detuning grid used for the first evaluation of the filter system, before Det_weight is
                    also applied.

                    Should be numpy array, e.g.  Det_base = np.linspace(-20, 20, 5000) * 1e3

    Det_weight:     Extremely narrow detuning grid used in the second evaluation of the filter system. After a
                    maximum in the filter profile is found in the first evaluation, Det_weight is biased to the
                    location of this maximum and then is combined with Det_base.

                    Should be numpy arrange with narrow range, e.g. Det_weight = weighted_space(-0.05, 0.05, 5000, 3) * 1e3

                    (weighted_space function creates an array with more points in the middle of the range than at the edges)

    E_in:           Default input electric field.

    '''


    global glob_var_progression        #used for printing results during optimization
                                       #could also be used to graph progression of optimization

    # Round the input parameters to a specified number of decimal places. Drastically reduces the time
    # for the optimizer to converge at a potential cost of finding a better filter. This could be customized
    # for each variable type so that optimizations account for accuracy of lab equipment.
    params = np.round(params, decimals=2)
    print_string = 'Etheta= ' + str(params[-1]) + ' '
    glob_var_progression[-2].append(params[-1])

    for i in range(len(var_names)):
        #used for printing results during optimization
        print_string += (var_names[i] + '= ' + str(params[i]) + ', ')
        glob_var_progression[i].append(params[i])

        # Convert degrees to radians for Btheta, Bphi
        if var_names[i][1:] == 'Btheta' or var_names[i][1:] == 'Bphi':
            params[i] = np.radians(params[i])

        #Assign value to correct p_dict in p_dict_list
        cell_no = int(var_names[i][0]) - 1
        p_dict_list[cell_no][var_names[i][1:]]=params[i]

    # print(p_dict_list)

    #First evaluation of filter system
    I_out_temp = cascade(Det_base, E_in, p_dict_list, Etheta=np.radians(params[-1])).real

    #Combining Det_base and biased Det_weight to create grid used in second evaluation
    bias = Det_base[np.argmax(I_out_temp)]
    Detuning = np.concatenate((Det_base, Det_weight + bias))
    Detuning.sort()

    #Second evaluation and FOM calculation
    I_out = cascade(Detuning, E_in, p_dict_list, Etheta=np.radians(params[-1])).real
    FOMval = FOM(Detuning, I_out) * 1e3

    glob_var_progression[-1].append(np.round(FOMval, decimals=3))
    print(glob_var_progression)

    #Print results during optimization process
    print("FnEv:", len(glob_var_progression[0]), print_string, "FOM=", np.round(FOMval, decimals=3), "Max I_out=",
          np.round(I_out.max(), decimals=3), "\t", p_dict_list)
    return -np.round(FOMval, decimals=3)





def optimizer():
    '''
    Function that performs the optimization. Adjustments made to optimization setup are currently  made
    within this function.
    '''

    # glob_var_progression used for printing results during optimization
    # could also be used to graph progression of optimization
    # should be set up so that it is a list of empty lists, with the number of empty lists being
    # the number of optimized parameters + 1
    # stores the variables and FOM for each evaluation of the objective function
    global glob_var_progression
    glob_var_progression = [[], [], [], [], [], [], [], []]

    # var names - denotes the names of variables used in the optimization. 1Bfield means first cell Bfield.
    # bounds - the bounds within which the optimizer chooses the value during a trial.
    # Position in bounds list corresponds to position of value in var_names.
    # Last tuple is always the bounds for Etheta
    var_names = ['1Bfield', '1T', '1Btheta', '2Bfield', '2T', '2Btheta']
    bounds = [(0, 500), (0, 300), (80, 100), (0, 500), (0, 300), (0, 180), (0, 180)]  # last one is Etheta


    p_dict_list = [{'Elem': 'Rb', 'Dline': 'D2', 'lcell': 5e-3, 'GammaBuf': 20.},
                   {'Elem': 'Rb', 'Dline': 'D2', 'lcell': 5e-3, }]
    Det_base = np.linspace(-20, 20, 5000) * 1e3
    Det_weight = weighted_space(-0.05, 0.05, 5000, 3) * 1e3
    E_in = np.array([1, 0, 0])

    print(var_names)
    print(bounds)
    print(p_dict_list)


    #Calling optimization algorithm. Uses objective_cascade as objective function.
    #tol=0.01 is chosen as this provides a good FOM without taking a more time than necessary to converge
    t0 = time.clock()
    result = optimize.differential_evolution(objective_cascade, bounds,
                    args=(var_names, p_dict_list, Det_base, Det_weight, E_in), tol=0.01)
    t1 = time.clock() - t0

    res = np.round(result.x, 2)
    print(res, -result.fun)
    print(result.nfev, result.nit)
    print("Evaluation time = ", t1)
    print("Time per evaluation = ", t1 / result.nfev)



optimizer()