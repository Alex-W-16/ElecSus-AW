from __future__ import (division, print_function, absolute_import)

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker

from PIL import Image
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize

import time
import sys
import imageio

from elecsus.elecsus_methods_NEW import calculate as calculate
from elecsus.elecsus_methods import calculate as calculate_OLD
from elecsus.libs.durhamcolours import *
from elecsus.goal_functions import *


plt.rcParams['mathtext.fontset'] = 'stix'
rc('font',family='serif',serif='STIXGeneral')
plt.rcParams['font.size']=10

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


def p_dict_list_creator(p_dict, num_slices, vary_param=None, vary_array=None):
    '''
    Creates a p_dict_list for use in other heatmap_along_z... functions.

    p_dict:         p_dict for the cell

    num_slices:     number of slices to split the cell into

    vary_param:     If applicable, the str name of a p_dict variable to vary along the length of
                    the cell (e.g. to simulate using a magnetic field whose strength varies along z)

                    If the cell parameters remain the same along the cell, leave blank

    vary_array:     The numpy array that describes how the variable chosen by vary_param
                    changes through the cell. Should have length of num_slices
    '''

    p_dict['lcell']/=num_slices

    p_dict_list=[]

    for i in range(num_slices):
        if vary_param is not None:
            p_dict[vary_param]=vary_array[i]
        p_dict_list.append(p_dict)

    return p_dict_list




#Made obsolete because heatmap_along_z covers both single and multiple cell systems, but left this
#as it's more clear what's going on in this function than in heatmap_along_z
def heatmap_along_z_single_cell(Detuning, E_in, Etheta, p_dict_list, output_list, output_label_list=[],
                                cmap_list=[],print_progress=False):
    '''
    Produces a heatmap showing how a particular value (such as S0) would change if it was
    measured at points along the length of the cell for a single cell filter system for the range
    of detuning specified.
    (Used in Fig. 7)

    Detuning, E_in, Etheta have usual meanings.

    p_dict_list:        Should be a list of p_dicts that describes a cell where each individual p_dict
                        specifies the parameters used for a 'slice' of the cell along z.

                        Each p_dict in p_dict list must have the same 'lcell' value - slices
                        should be evenly sized.

                        Advisable to use p_dict_list_creator function for to create each
                        individual p_dict_list.

                        If using 'I_perp' or 'I_par', slightly different calculation method used.
                        (Not recommended to use Etheta within p_dict - instead specify externally)



    output_list:        Should be a list that determines what ElecSus outputs are used for
                        the heatmap. If more than one, the first will form the first row
                        of the figure, the second the next row etc.
                        Real value of output will be used for plotting.

    output_label_list:  List of strings that sets the axis labels for each output.
                            i.e. if output_list[2]='I_perp', should use e.g. output_label_list[2]=r'$I_\perp$'

                        If fewer output_label_list items are supplied than outputs, will default to using
                        output_list items.

    cmap_list:          List that specifies what type of color map is used for each output's heatmap.
                        Default is to use 'gnuplot' for all heatmaps. If fewer cmaps are supplied than
                        outputs, then default cmap is used for these.

    print_progress:     If true, prints updates on progress of creating data for the heatmap(s).


    '''

    E_in = rotate_Efield(E_in, Etheta)

    num_outputs= len(output_list)
    num_slices = len(p_dict_list)

    output_array = np.zeros((num_outputs,num_slices,len(Detuning)))  #preparing 3d array to store outputs
    output_list.append('E_out') #Adding E_out to end of output list

    # Handling the special case of I_perp and I_par
    perp_index = None
    par_index = None

    if 'I_perp' in output_list:
        angle = Etheta + np.pi / 2
        J_out_perp = np.matrix([[np.cos(angle)**2, np.sin(angle)*np.cos(angle)],
					            [np.sin(angle)*np.cos(angle), np.sin(angle)**2]])
        perp_index = output_list.index('I_perp')

    if 'I_par' in output_list:
        angle = Etheta
        J_out_par = np.matrix([[np.cos(angle) ** 2, np.sin(angle) * np.cos(angle)],
                               [np.sin(angle) * np.cos(angle), np.sin(angle) ** 2]])
        par_index = output_list.index('I_par')


    #Iterating for each slice
    for i in range(num_slices):
        if print_progress:  print(i,p_dict_list[i])

        #Calculating outputs for slice
        temp_output=np.asarray(calculate(Detuning,E_in,p_dict_list[i],outputs=output_list))

        #Setting next E_in
        E_in = temp_output[-1]

        # Assining each output from temp_output to correct location within output_array
        for j in range(num_outputs):

            #Handling the special case of I_perp and I_par
            if j == perp_index:
                transmittedE = np.array(J_out_perp * E_in[:2])
                temp_output[j]=(transmittedE * transmittedE.conjugate()).sum(axis=0)

            if j == par_index:
                transmittedE = np.array(J_out_par * E_in[:2])
                temp_output[j]=(transmittedE * transmittedE.conjugate()).sum(axis=0)

            output_array[j][i] = temp_output[j].real






    #Preparing to plot
    cell_length = num_slices*p_dict_list[0]['lcell']
    cell_z = np.linspace(0,cell_length,num_slices)
    cell_z_mesh, Detuning_mesh = np.meshgrid(cell_z,Detuning)

    num_cols=2     #number of columns - 2 because single cell

    fig = plt.figure(figsize=(3.5*num_cols, 2.5*num_outputs))     #size of figure
    gs = gridspec.GridSpec(num_outputs, num_cols)               #arranging subplots
    gs.update(wspace=0.1)                           #adjusting width between subplots
    gs.update(hspace=0.3)                           #adjusting heignt between subplots

    while len(cmap_list) < num_outputs:     cmap_list.append('gnuplot')

    if len(output_label_list) < num_outputs:    output_label_list.extend(output_list[len(output_label_list):])

    for row in range(num_outputs):
        for col in range(num_cols):

            #Make subplots share y axis with leftmost subplot in column
            if col == 0:
                axis = plt.subplot(gs[row,col])
                axis.set_ylim(np.max(Detuning) * 1e-3, np.min(Detuning) * 1e-3)
                axis.set_ylabel("Detuning (GHz)")
            else:
                axis = plt.subplot(gs[row,col], sharey=axis)
                axis.tick_params(labelleft=False)
                axis.set_xlim(-0.05,1.05)

            #Deciding whether to plot the heatmap or the line plot
            if (col % 2) == 0:

                #Creating heatmap. vmin, vmax chosen for consistency across figure, though could choose to use
                #vmin and vmax for output_array[row]

                map = axis.pcolormesh(cell_z_mesh * 1e3, Detuning_mesh * 1e-3, output_array[row].T,
                                    cmap=cmap_list[row], vmin=output_array.min(), vmax=output_array.max())
                axis.set_xlabel('Cell z position (mm)')

                #axis.xaxis.set_major_locator(ticker.MultipleLocator(2.5))   #can be turned on to control x axis intervals

            else:

                #Creating line graph

                axis.plot(output_array[row][-1],Detuning*1e-3, color=d_red)

                axis.set_xlabel(output_label_list[row]+' after cell')
                axis.xaxis.set_major_locator(ticker.MultipleLocator(0.5))   #can be turned on to control x axis intervals

                # **Comparing with non-cascaded calculation, as in Fig 7**
                if 1==1:   #(Toggle on/off)
                    p_dict = p_dict_list[0]
                    p_dict['lcell'] = cell_length
                    E_in = np.array([1, 0, 0])
                    p_dict['Etheta']=Etheta
                    [OP] = calculate(Detuning, E_in, p_dict, outputs=[output_list[row]])
                    axis.plot(OP,Detuning*1e-3, color=d_lightblue, linestyle='--',linewidth=0.75)


            #Adding colorbar to rightmost column of subplots
            if col == num_cols-1:
                cbar = plt.colorbar(map,ax=[axis])
                cbar.set_label(output_label_list[row])

    #plt.savefig("fig.png", dpi=1000, bbox_inches="tight")
    plt.show()

    return


def heatmap_along_z(Detuning, E_in, Etheta, p_dict_list_LIST, output_list, output_label_list=[],
                                cmap_list=[],print_progress=False):
    '''
    Produces a heatmap showing how a particular value (such as S0) would change if it was
    measured at points along the length of the cell for a filter system with one or more cells
    for the range of detuning specified.
    (Used in Fig. 7, 11)

    Detuning, E_in, Etheta have usual meanings.

    p_dict_list_LIST:   Should be a list of p_dict_lists that describes the filter system where each
                        p_dict_list describes a single cell in the system.

                        An individual p_dict_list should be as described below:

    p_dict_list:        Should be a list of p_dicts that describes a cell where each individual p_dict
                        specifies the parameters used for a 'slice' a the cell along z.

                        Each p_dict in p_dict list must have the same 'lcell' value - slices
                        should be evenly sized.

                        Advisable to use p_dict_list_creator function for to create each
                        individual p_dict_list.

                        (Not recommended to use Etheta within p_dict - instead specify externally)



    output_list:        Should be a list that determines what ElecSus outputs are used for
                        the heatmap. If more than one, the first will form the first row
                        of the figure, the second the next row etc.
                        Real value of output will be used for plotting.

                        If using 'I_perp' or 'I_par', slightly different calculation method used.

    output_label_list:  List of strings that sets the axis labels for each output.
                            i.e. if output_list[2]='I_perp', should use e.g. output_label_list[2]=r'$I_\perp$'

                        If fewer output_label_list items are supplied than outputs, will default to using
                        output_list items.

    cmap_list:          List that specifies what type of color map is used for each output's heatmap.
                        Default is to use 'gnuplot' for all heatmaps. If fewer cmaps are supplied than
                        outputs, then default cmap is used for these.

    print_progress:     If true, prints updates on progress of creating data for the heatmap(s).


    '''

    E_in = rotate_Efield(E_in, Etheta)

    num_outputs= len(output_list)
    num_cells = len(p_dict_list_LIST)

    total_slices = 0
    # Setting to [0] so loop starts from zero later
    num_slices_list = [0]           #num slices in each cell
    total_slices_list = [0]         #total number of slices after each cell

    for list in p_dict_list_LIST:
        total_slices += len(list)
        num_slices_list.append(len(list))
        total_slices_list.append(total_slices)

    output_array = np.zeros((num_outputs,total_slices,len(Detuning)))  #preparing 3d array to store outputs

    output_list.append('E_out') #Adding E_out to end of output list

    # Handling the special case of I_perp and I_par

    if 'I_perp' in output_list:
        angle = Etheta + np.pi / 2
        J_out_perp = np.matrix([[np.cos(angle)**2, np.sin(angle)*np.cos(angle)],
					            [np.sin(angle)*np.cos(angle), np.sin(angle)**2]])

    if 'I_par' in output_list:
        angle = Etheta
        J_out_par = np.matrix([[np.cos(angle) ** 2, np.sin(angle) * np.cos(angle)],
                               [np.sin(angle) * np.cos(angle), np.sin(angle) ** 2]])

    #Iterating for each slice
    for cell in range(num_cells):

        for i in range(num_slices_list[cell+1]):

            if print_progress:
                print("Cell:",cell+1,"\tSlice:",i+1,"\tp_dict:",p_dict_list_LIST[cell][i])

            #Calculating outputs for slice
            temp_output=np.asarray(calculate(Detuning,E_in,p_dict_list_LIST[cell][i],outputs=output_list))

            #Setting next E_in
            E_in = temp_output[-1]

            # Assining each output from temp_output to correct location within output_array
            for j in range(num_outputs):

                #Handling the special case of I_perp and I_par
                if output_list[j] == 'I_perp':
                    transmittedE = np.array(J_out_perp * E_in[:2])
                    temp_output[j]=(transmittedE * transmittedE.conjugate()).sum(axis=0)

                if output_list[j] == 'I_par':
                    transmittedE = np.array(J_out_par * E_in[:2])
                    temp_output[j]=(transmittedE * transmittedE.conjugate()).sum(axis=0)

                output_array[j][total_slices_list[cell]+i] = temp_output[j].real





    #Preparing to plot
    cell_lengths_list=[]
    cell_z_list = []
    cell_z_mesh_list = []
    Detuning_mesh_list = []

    for cell in range(num_cells):
        cell_length = num_slices_list[cell+1]*p_dict_list_LIST[cell][0]['lcell']
        cell_z = np.linspace(0,cell_length,num_slices_list[cell+1])
        cell_z_mesh, Detuning_mesh = np.meshgrid(cell_z, Detuning)

        cell_lengths_list.append(cell_length)
        cell_z_list.append(cell_z)
        cell_z_mesh_list.append(cell_z_mesh)
        Detuning_mesh_list.append(Detuning_mesh)


    num_cols=2*num_cells     #number of columns - 2 for each cell

    fig = plt.figure(figsize=(3.5*num_cols, 2.5*num_outputs))     #size of figure
    gs = gridspec.GridSpec(num_outputs, num_cols)               #arranging subplots
    gs.update(wspace=0.1)                           #adjusting width between subplots
    gs.update(hspace=0.3)                           #adjusting heignt between subplots

    while len(cmap_list) < num_outputs:     cmap_list.append('gnuplot')

    if len(output_label_list) < num_outputs:    output_label_list.extend(output_list[len(output_label_list):])


    for row in range(num_outputs):
        for col in range(num_cols):

            #Making subplots. Creating them in a way so that all subplots share the y-axis
            #and each column shares the x-axis to make zooming in the plt.show() window interesting

            if col == row == 0:
                axis = plt.subplot(gs[row, col])
                axis.set_ylim(np.max(Detuning) * 1e-3, np.min(Detuning) * 1e-3)
                axis.set_ylabel("Detuning (GHz)")
            elif col == 0:
                axis = plt.subplot(gs[row, col], sharey=axis)
                axis.set_ylabel("Detuning (GHz)")
            else:
                axis = plt.subplot(gs[row, col], sharey=axis)
                axis.tick_params(labelleft=False)


            #Deciding whether to plot the heatmap or the line plot
            if (col % 2) == 0:

                #Creating heatmap.

                cell = int(col/2)
                first_slice = total_slices_list[cell]
                last_slice  = total_slices_list[cell+1]

                cmin=output_array[row].min()
                cmax=output_array[row].max()

                map = axis.pcolormesh(cell_z_mesh_list[cell] * 1e3, Detuning_mesh_list[cell] * 1e-3,
                                      output_array[row][first_slice:last_slice].T,
                                      cmap=cmap_list[row], vmin=cmin, vmax=cmax)
                axis.set_xlabel('Cell '+str(cell+1)+' z position (mm)')

                #axis.xaxis.set_major_locator(ticker.MultipleLocator(2.5))   #can be turned on to control x axis intervals

            else:

                #Creating line graph
                xmin = output_array.min()-0.05
                xmax = output_array.max()+0.05

                axis.set_xlim(xmin,xmax)
                axis.plot(output_array[row][last_slice-1],Detuning*1e-3, color=d_red)

                axis.set_xlabel(output_label_list[row]+' after cell '+str(cell+1))
                axis.xaxis.set_major_locator(ticker.MultipleLocator(0.5))   #can be turned on to control x axis intervals

                # **Comparing with non-cascaded calculation, as in Fig 7**       #Only works for single cell
                if num_cells==1 and 1==1:   #(Toggle on/off)
                    p_dict = p_dict_list_LIST[0][0]
                    p_dict['lcell'] = cell_length
                    E_in = np.array([1, 0, 0])
                    p_dict['Etheta']=Etheta
                    [OP] = calculate(Detuning, E_in, p_dict, outputs=[output_list[row]])
                    axis.plot(OP,Detuning*1e-3, color=d_lightblue, linestyle='--',linewidth=0.75)


            #Adding colorbar to rightmost column of subplots
            if col == num_cols-1:
                cbar = plt.colorbar(map,ax=[axis])
                cbar.set_label(output_label_list[row])

    #plt.savefig("fig.png", dpi=1000, bbox_inches="tight")
    plt.show()

    return








def example_heatmap_along_z_single_cell_use():
    '''Example of usage for heatmap_along_z_single_cell function'''

    Detuning = np.linspace(-10,10,2000)*1e3
    E_in = np.array([1,0,0])
    p_dict = {'Elem':'Rb', 'Dline':'D2', 'T':126, 'Bfield':230, 'Btheta':np.radians(83),
              'lcell':5e-3}  #FOM 1.24 single cell
    Etheta = np.radians(6)

    num_slices=100
    output_list=['I_perp','I_par','Il','Ir']
    output_label_list=[r'$I_{\perp}$',r'$I_{\parallel}$',r'$I_L$',r'$I_R$']
    cmap_list=['nipy_spectral','nipy_spectral']

    p_dict_list = p_dict_list_creator(p_dict.copy(),num_slices)


    heatmap_along_z_single_cell(Detuning,E_in,Etheta,p_dict_list,output_list,
        output_label_list=output_label_list,cmap_list=cmap_list,print_progress=True)

    return


def example_heatmap_along_z_use():
    '''Example of usage for heatmap_along_z function.
    Can be used for any number of cells (including one cell)'''

    #####One cell usage#####

    if 1 == 0: #toggle on/off
        Detuning = np.linspace(-10, 10, 2000) * 1e3
        E_in = np.array([1, 0, 0])
        p_dict = {'Elem': 'Rb', 'Dline': 'D2', 'T': 126, 'Bfield': 230, 'Btheta': np.radians(83),
                  'lcell': 5e-3}  # FOM 1.24 single cell
        Etheta = np.radians(6)

        num_slices = 100
        output_list = ['I_perp', 'I_par', 'Il', 'Ir']
        output_label_list = [r'$I_{\perp}$', r'$I_{\parallel}$', r'$I_L$', r'$I_R$']
        cmap_list = ['nipy_spectral', 'nipy_spectral']

        p_dict_list = p_dict_list_creator(p_dict.copy(), num_slices)
        p_dict_list_LIST = [p_dict_list]

        heatmap_along_z(Detuning, E_in, Etheta, p_dict_list_LIST, output_list,
                        output_label_list=output_label_list, cmap_list=cmap_list, print_progress=True)

    #####Multiple cells usage#####
    if 1 == 1: #toggle on/off

        bias = 0.238*1e3           #Rb85/Nat bias
        Det_base = np.linspace(-10,10,5000)*1e3
        Det_weight = weighted_space(-0.05,0.05,5000,3)*1e3+bias
        Detuning = np.concatenate((Det_base,Det_weight))
        Detuning.sort()
        E_in = np.array([1, 0, 0])

        #FOM 4.86 - Rotator then Filter
        p_dict1 = {'Elem': 'Rb', 'Dline': 'D2', 'Bfield': 269.211, 'T': 91.519,'Btheta': np.radians(31.516),'lcell': 5e-3}
        p_dict2 = {'Elem':'Rb', 'Dline':'D2', 'Bfield':292.604, 'T':126.473, 'Btheta':np.radians(92.404), 'lcell':5e-3}
        Etheta = np.radians(2.07)

        num_slices1 = 100
        num_slices2 = 100     #can have different no. of slices for each cell

        output_list = ['I_perp', 'I_par', 'Il', 'Ir']
        output_label_list = [r'$I_{\perp}$', r'$I_{\parallel}$', r'$I_L$', r'$I_R$']
        cmap_list = ['nipy_spectral', 'nipy_spectral']

        p_dict_list1 = p_dict_list_creator(p_dict1.copy(), num_slices1)
        p_dict_list2 = p_dict_list_creator(p_dict2.copy(), num_slices2)
        p_dict_list_LIST = [p_dict_list1,p_dict_list2]

        heatmap_along_z(Detuning, E_in, Etheta, p_dict_list_LIST, output_list,
                        output_label_list=output_label_list, cmap_list=cmap_list, print_progress=True)


    return

#example_heatmap_along_z_single_cell_use()
example_heatmap_along_z_use()


