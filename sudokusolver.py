import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import imageio
import numpy.random as rnd
from smt.sampling_methods import LHS
import time

unsolved = np.array(
    [3, 9, 6, 0, 8, 0, 7, 1, 5, 0, 0, 7, 9, 6, 1, 0, 0, 3, 8, 4, 1, 0, 3, 0, 2, 0, 0, 6, 3, 0, 1, 0, 0, 5, 2, 0, 1, 0,
     0, 7, 2, 5, 3, 9, 0, 0, 5, 2, 6, 9, 3, 8, 0, 1, 4, 0, 8, 3, 7, 6, 0, 0, 2, 2, 0, 5, 8, 0, 9, 0, 3, 4, 9, 6, 3, 0,
     0, 2, 0, 7, 8])


def cost_function(x):
    global unsolved
    a = copy.deepcopy(unsolved)  # copying the unsolved sudoku
    cost = 0  # resetting the penalty cost
    x_count = 0
    for i in range(len(a)):  # filling the unsolved array with input values x
        if a[i] == 0:
            a[i] = x[x_count]
            x_count += 1
    a = np.reshape(a, (9, 9))  # converting to 9x9 grid
    for i in range(9):  # iterating over rows
        row = a[i, :]
        non_dupe = set(row)
        cost += 9 - len(non_dupe)  # cost for each duplicate
    for i in range(9):  # iterating over columns
        col = a[:, i]
        non_dupe = set(col)
        cost += 9 - len(non_dupe)  # cost for each duplicate
    for i in range(3):  # iterating over each 3x3 grid
        for j in range(3):
            kernal = a[i * 3:(i * 3) + 3, j * 3:(j * 3) + 3]
            kernal = np.reshape(kernal, 9)
            non_dupe = set(kernal)
            cost += 9 - len(non_dupe)  # cost for each duplicate
    return cost


def plot_util(unsolved, i_best, fstore, iteration):
    '''
    INPUTS:
    unsolved            : unsolved sudoku array
    i_best              : input array
    fstore              : cost function value array
    iteration           : the iteration number

    OUTPUTS:
    a .png file displaying the sudoku as well as the
    function value over iteration.
    '''
    x_count = 0
    empty_array = np.zeros_like(unsolved)
    for i in range(len(unsolved)):  # filling empty sudoku
        if unsolved[i] == 0:
            empty_array[i] = 1
    empty_array = np.reshape(empty_array, (9, 9))
    empty_array = np.transpose(empty_array)
    a = copy.deepcopy(unsolved)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = i_best[x_count]
            x_count += 1
    a = np.reshape(a, (9, 9))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left=0.05, right=0.95)
    axs[0].set_title('Current Sudoku')
    axs[0].axis('off')
    axs[1].set_xlabel('Iteration')
    axs[1].grid('True')
    axs[1].set_ylabel('Errors')
    for i in range(10):
        if i == 0 or i == 3 or i == 6 or i == 9:
            axs[0].plot([0, 9], [i, i], color='k', linewidth=3)
            axs[0].plot([i, i], [0, 9], color='k', linewidth=3)
        else:
            axs[0].plot([0, 9], [i, i], color='k', linewidth=1)
            axs[0].plot([i, i], [0, 9], color='k', linewidth=1)
    a = np.transpose(a)
    for i in range(9):
        for j in range(9):
            if empty_array[i, j] == 1:
                axs[0].text(i + 0.45, 9 - 0.65 - j, a[i, j], fontweight='extra bold')
            else:
                axs[0].text(i + 0.45, 9 - 0.65 - j, a[i, j])
    axs[1].plot(np.linspace(0, len(fstore), len(fstore)), fstore)
    fig.savefig(str(iteration) + '.png')
    return


def evolution(f, p, it, cull_percen, mut_percen, unsolved):
    '''
    INPUTS:
    f               : sudoku cost function
    p               : population size
    it              : iterations
    cull_percen     : percentage of population to be culled
    mut_percen      : percentage chance of a mutation
    unsolved        : the unsolved sudoku vector
    OUTPUTS:
    i_best          : the vector of optimised input values
    SudokuSolve.gif : a gif showing function value over time as
                        well as the current sudoku
    '''
    empty_entries = 0
    for i in range(len(unsolved)):
        if unsolved[i] == 0:
            empty_entries += 1
    dimension_bounds = [1, 9]
    bounds = np.zeros((empty_entries, 2))
    for i in range(len(bounds)):
        bounds[i, 0] = dimension_bounds[0]
        bounds[i, 1] = dimension_bounds[1]
    d = len(bounds)
    '''ORIGINAL SAMPLE'''
    sampling = LHS(xlimits=bounds)  # LHS Sample
    i_pos = sampling(p)
    '''EVALUATING FITNESSES'''
    i_val = np.zeros((len(i_pos), 1))
    for i in range(len(i_pos)):
        i_val[i, :] = f(i_pos[i, :])
    i_pos = np.concatenate((i_pos, i_val), axis=1)
    i_best = i_pos[np.argmin(i_pos[:, -1])]
    iteration = 0
    fstore = []
    while iteration < it:  # PARAMETER HERE (iterations)
        '''TOURNAMENT SELECTION'''
        i_new = np.zeros((int(p * (cull_percen)), d + 1))  # PARAMETER HERE (percentage to be kept)
        new_count = 0
        while new_count < len(i_new):
            rnd.shuffle(i_pos)
            t_size = rnd.randint(1, 10)  # SORT OF PARAMETER HERE (tournament size)
            t = i_pos[:t_size, :]
            t_best = t[np.argmin(t[:, -1])]
            i_new[new_count, :] = t_best[:]
            new_count += 1
        i_pos = copy.deepcopy(i_new)
        '''COMPLETING WITH RANDOM CANDIDATES'''
        new_psize = p - len(i_pos)
        sampling = LHS(xlimits=bounds)
        i_new = sampling(new_psize)
        i_val_new = np.zeros((len(i_new), 1))
        for i in range(len(i_new)):
            i_val_new[i, :] = f(i_new[i, :])
        i_new = np.concatenate((i_new, i_val_new), axis=1)
        i_pos = np.concatenate((i_new, i_pos), axis=0)
        best_index = np.argmin(i_pos[:, -1])
        i_best = i_pos[best_index]
        i_best_val = i_pos[best_index, -1]
        print(i_best_val, end='\r')
        fstore.append(i_best_val)
        '''CROSSOVER HERE'''
        rnd.shuffle(i_pos)
        cross_index = np.linspace(0, p - 2, (p / 2))
        for i in cross_index:  # SINGLE CROSSOVER
            i = int(i)
            k = rnd.randint(0, d)
            i_pos[i + 1, k:] = i_pos[i, k:]
            i_pos[i + 1, :k] = i_pos[i + 1, :k]
            i_pos[i, :k] = i_pos[i, :k]
            i_pos[i, k:] = i_pos[i + 1, k:]
        '''MUTATION CODE HERE'''
        for i in range(len(i_pos)):
            for j in range(d):
                prob = rnd.uniform()
                if prob < mut_percen:
                    i_pos[i, j] = rnd.uniform(bounds[j, 0], bounds[j, 1])
        i_pos = i_pos[:, :-1]
        i_val = np.zeros((len(i_pos), 1))
        for i in range(len(i_pos)):
            i_val[i, :] = f(i_pos[i, :])
        i_pos = np.concatenate((i_pos, i_val), axis=1)

        plot_util(unsolved, i_best, fstore, iteration)
        iteration += 1

    images = []

    for filename in range(it + 1):
        images.append(imageio.imread(str(filename) + '.png'))
        os.remove(str(filename) + '.png')

    imageio.mimsave('SudokuSolve.gif', images)
    return i_best


solved = evolution(cost_function, 100, 100, 0.8, 0.05, unsolved)