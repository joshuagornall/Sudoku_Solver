import numpy as np
import matplotlib.pyplot as plt
import copy
import imageio
import numpy.random as rnd
import random
import glob
from celluloid import Camera 
import os
    
from IPython import get_ipython
get_ipython().magic('reset -sf')

unsolved = np.array(
    [3, 9, 6, 0, 8, 0, 7, 1, 5, 0, 0, 7, 9, 6, 1, 0, 0, 3, 8, 4, 1, 0, 3, 0, 2, 0, 0, 6, 3, 0, 1, 0, 0, 5, 2, 0, 1, 0,
     0, 7, 2, 5, 3, 9, 0, 0, 5, 2, 6, 9, 3, 8, 0, 1, 4, 0, 8, 3, 7, 6, 0, 0, 2, 2, 0, 5, 8, 0, 9, 0, 3, 4, 9, 6, 3, 0,
     0, 2, 0, 7, 8])


def cost_function(x,unsolved=unsolved):
    # function to determine the duplicate values in the Sudoku
    # by rows, columns, and quadrant

    # x contains random numbers. size array[n = empty entries,]
    
    unsolved_copy = copy.deepcopy(unsolved)   # copying the unsolved sudoku

    # filling the unsolved array with input values x
    x_count = 0
    for i,value in enumerate(unsolved_copy):
        if value == 0:
            unsolved_copy[i] = x[x_count]
            x_count += 1
            
    # converting to 9x9 grid
    dim_sudoku = 9 # number of rows or columns in the Sudoku
    unsolved_copy = np.reshape(unsolved_copy, (dim_sudoku, dim_sudoku))
    
    # Determinate duplicates
                                 
    # duplicates in rows
    num_duplicates_row = 0  # initilise num_duplicates   
    for irow in range(dim_sudoku):
        row = unsolved_copy[irow, :] # select each row
        unique_values = set(row) # find the unique values and sort them
        num_duplicates_row += dim_sudoku - len(unique_values)  #
    
    # duplicates in columns
    num_duplicates_col = 0
    for icol in range(dim_sudoku):
        col = unsolved_copy[:, icol] # select each column
        unique_values = set(col) # find the unique values and sort them
        num_duplicates_col += dim_sudoku - len(unique_values)  # 

    # duplicates in each 3x3 grid quadrant
    num_duplicates_quad=0
    for i in range(3):
        for j in range(3):
            kernal = unsolved_copy[i * 3:(i * 3) + 3,
                                   j * 3:(j * 3) + 3] # select each subwindow
            kernal = np.reshape(kernal, dim_sudoku) # transform to vector
            unique_values = set(kernal) # find the unique values and sort them
            num_duplicates_quad += dim_sudoku - len(unique_values)
    num_duplicate=num_duplicates_row+num_duplicates_col+num_duplicates_quad
            
    return num_duplicate,num_duplicates_row,num_duplicates_col,num_duplicates_quad




def plot_util(unsolved, sample_best_values, errors_best_sample_store, iteration):
    '''
    INPUTS:
    unsolved            : unsolved sudoku array
    sample_best_values              : input array
    errors_best_sample_store              : cost function value array
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
    sudoku_table = copy.deepcopy(unsolved)
    for i in range(len(sudoku_table)):
        if sudoku_table[i] == 0:
            sudoku_table[i] = sample_best_values[x_count]
            x_count += 1
    sudoku_table = np.reshape(sudoku_table, (9, 9))
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left=0.05, right=0.95)
    axs[0].set_title('Current best Sudoku')
    axs[0].axis('off')
    axs[1].set_xlabel('Iteration')
    axs[1].grid('True')
    axs[1].set_ylabel('Errors')
    #axs[1].set_yscale('log')
    for i in range(10):
        if i == 0 or i == 3 or i == 6 or i == 9:
            axs[0].plot([0, 9], [i, i], color='k', linewidth=3)
            axs[0].plot([i, i], [0, 9], color='k', linewidth=3)
        else:
            axs[0].plot([0, 9], [i, i], color='k', linewidth=1)
            axs[0].plot([i, i], [0, 9], color='k', linewidth=1)
    
    
    sudoku_table = np.transpose(sudoku_table)
    
    for i in range(9):
        for j in range(9):
            if empty_array[i, j] == 1:
                axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_table[i, j], fontweight='extra bold')
            else:
                axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_table[i, j])
    axs[1].plot(np.linspace(0, len(errors_best_sample_store), len(errors_best_sample_store)), errors_best_sample_store)
    
    fig.savefig('img/'+str(iteration) + '.png')
    
    
    
    return

def plot_util2(unsolved, solved):
    '''
    INPUTS:
    unsolved            : unsolved sudoku array
    sample_best_values              : input array
    errors_best_sample_store              : cost function value array
    iteration           : the iteration number

    OUTPUTS:
    sudoku_table .png file displaying the sudoku as well as the
    function value over iteration.
    '''
    errors_best_sample_store=[solved[i][range(-4,0)]for i in range(len(solved))]
    
    iterations=np.size(solved,0)

    # filling empty sudoku
    empty_array = np.zeros_like(unsolved)
    
    for i in range(len(unsolved)):  
        if unsolved[i] == 0:
            empty_array[i] = 1
    empty_array = np.reshape(empty_array, (9, 9))
    empty_array = np.transpose(empty_array)
    
    
    # fill zeros with sample values
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left=0.05, right=0.95)
    camera = Camera(fig)# the camera gets our figure
    
    axs[0].set_title('Current best Sudoku')
    axs[0].axis('off')
    axs[1].set_xlabel('Iteration')
    axs[1].grid('True')
    axs[1].set_ylabel('Errors')

                
    for iteration in np.linspace(0,iterations-1,100).astype('int'):
        sudoku_table = copy.deepcopy(unsolved)
        print(iteration)
        sample_best_values=solved[iteration]
        
        x_count = 0
        for i in range(len(sudoku_table)):
            if sudoku_table[i] == 0:
                sudoku_table[i] = sample_best_values[x_count]
                x_count += 1
                
        sudoku_table = np.reshape(sudoku_table, (9, 9))
        sudoku_table = np.transpose(sudoku_table)
        
        for i in range(10):
            if i == 0 or i == 3 or i == 6 or i == 9:
                axs[0].plot([0, 9], [i, i], color='k', linewidth=3)
                axs[0].plot([i, i], [0, 9], color='k', linewidth=3)
            else:
                axs[0].plot([0, 9], [i, i], color='k', linewidth=1)
                axs[0].plot([i, i], [0, 9], color='k', linewidth=1)
        
        for i in range(9):
            for j in range(9):
                if empty_array[i, j] == 1:
                    axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_table[i, j], fontweight='extra bold')
                else:
                    axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_table[i, j])
                    
        new_error=errors_best_sample_store[0:iteration]
        axs[1].plot(np.linspace(0, len(new_error), len(new_error)), new_error)
        #fig.savefig('img/'+str(iteration) + '.png')
        camera.snap()
    animation = camera.animate()
    animation.save('Sudoku_solution.mp4')
    return animation


def sample_error_col(sample_size,sample_matrix):
    ''' p: population size '''
    
    # compute the duplicate for each set of random samples
    sample_num_duplicates = np.zeros((sample_size, 4))
    for i in range(sample_size): sample_num_duplicates[i, :] = cost_function(sample_matrix[i, :])
    # concatenate random sample values and duplicate number
    sample_matrix = np.concatenate((sample_matrix, sample_num_duplicates), axis=1)
    return sample_matrix

def sample_min_error(sample_error_col_obj):
    return sample_error_col_obj[np.argmin(sample_error_col_obj[:, -1])]


def random_permutation(iterable,sample_number=10):
    "Random selection from itertools.permutations(iterable, r)"
    pool = tuple(iterable)
    r = len(pool)
    return np.array([random.sample(pool, r) for i in range(sample_number)])
       
def evolution2(p = 20, it = 100000, cull_percen = 0.999, mut_percen = 0.05, unsolved=unsolved,initial_condition=[0]):
    '''
    INPUTS:
    f               : sudoku cost function
    p               : population size
    it              : iterations
    cull_percen     : percentage of population to be culled
    mut_percen      : percentage chance of a mutation
    unsolved        : the unsolved sudoku vector
    initial_condition: initial guess of the unsolved numbers
    
    OUTPUTS:
    sample_best_values          : the vector of optimised input values

    Default values

    p=20
    it=1000
    cull_percen= 0.8
    mut_percen=0.05
    unsolved

    '''
    print("Solving Sudoku")
   
    unique_items, unique_counts = np.unique(unsolved, return_counts=True) # count unique 0,1,2,3,4,5,6,7,8,9
    empty_entries = unique_counts[0] # count zeros
    unique_items=unique_items[1:]
    unique_counts=9-unique_counts[1:]
    
    initial_sample = np.repeat(unique_items,unique_counts,axis=0)
    sample_random_values=random_permutation(initial_sample,p)
    
    '''EVALUATING FITNESSES'''
    # compute error (duplicated values) and add the last column
    sample_random_values=sample_error_col(sample_size=p, sample_matrix=sample_random_values)
    
    # determine the lowest duplicate set
    sample_best_values = sample_min_error(sample_random_values)
    
    if len(initial_condition)>1:
        sample_random_values[1]=initial_condition
        
   # errors_best_sample_store = []
    sample_best_values_store= []

    
    for iteration in range(it):
        '''TOURNAMENT SELECTION: subset of random values'''
        sample_random_values_subset = np.zeros((int(p * (cull_percen)), empty_entries + 4))  # PARAMETER HERE (percentage to be kept)
        
        
        for new_count in range(len(sample_random_values_subset)):
            rnd.shuffle(sample_random_values)
            t_size = rnd.randint(1, 10)  # SORT OF PARAMETER HERE (tournament size)
            t = sample_random_values[:t_size, :] # select a subset of the random sample
            t_best = t[np.argmin(t[:, -4])] # determine the lowest duplicate set
            sample_random_values_subset[new_count, ] = t_best # fill subset with the best

        sample_random_values = copy.deepcopy(sample_random_values_subset)# selected subset
        
        '''COMPLEMENT WITH RANDOM CANDIDATES'''
        new_psize = p - len(sample_random_values)
        sample_random_values_subset = random_permutation(initial_sample,new_psize)  #sample given the new size
        
        # compute the duplicate for each set of new random samples
        sample_random_values_subset=sample_error_col(new_psize, sample_random_values_subset)
        
        
        ''' COMPLETE SAMPLE'''
        # concatenate random sample values and duplicate number (selection+complement)
        sample_random_values = np.concatenate((sample_random_values_subset, sample_random_values), axis=0)
        
        # determine the best sample from selection+complement
        best_index = np.argmin(sample_random_values[:, -4])
        sample_best_values = sample_random_values[best_index]
        errors_best_sample = sample_random_values[best_index, range(-4,0)]
        
        
        #rnd.shuffle(sample_random_values)
        '''MUTATION CODE HERE'''
        for i in range(len(sample_random_values)):
            for j in range(empty_entries):
                prob = rnd.uniform()
                if prob < mut_percen and best_index!=i :
                    sample_random_values[i, j] = rnd.randint(1,10)
                    
        
        sample_random_values = sample_random_values[:, :-4] # delete the last column (duplicates number)
        sample_random_values = sample_error_col(p, sample_random_values) # calculate new errors
        
        
        # plot the current Sudoku and error per iteraton
        if iteration%100==0: print('iteration:'+str(iteration)+ ' errors:'+str(errors_best_sample), end='\n')
        sample_best_values_store.append(sample_best_values)
        #if iteration%10==0:plot_util(unsolved, sample_best_values, errors_best_sample_store, iteration)
        if errors_best_sample[0]==0: break

    return sample_best_values_store


#solved= [np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 9, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 0, 0,0,0]),
#         np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 8, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 9])]

solved= evolution2(p = 20, it = 100000, cull_percen = 0.999, mut_percen = 0.05, unsolved=unsolved)
plot_util2(unsolved,solved)