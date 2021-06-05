import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import imageio
import numpy.random as rnd
from smt.sampling_methods import LHS


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
    num_duplicates = 0  # initilise num_duplicates                                
    # duplicates in rows
    for irow in range(dim_sudoku):
        row = unsolved_copy[irow, :] # select each row
        unique_values = set(row) # find the unique values and sort them
        num_duplicates += dim_sudoku - len(unique_values)  #

    # duplicates in columns
    for icol in range(dim_sudoku):
        col = unsolved_copy[:, icol] # select each column
        unique_values = set(col) # find the unique values and sort them
        num_duplicates += dim_sudoku - len(unique_values)  # 

    # duplicates in each 3x3 grid quadrant
    for i in range(3):
        for j in range(3):
            kernal = unsolved_copy[i * 3:(i * 3) + 3,
                                   j * 3:(j * 3) + 3] # select each subwindow
            kernal = np.reshape(kernal, dim_sudoku) # transform to vector
            unique_values = set(kernal) # find the unique values and sort them
            num_duplicates += dim_sudoku - len(unique_values)
            
    return num_duplicates




def plot_util(unsolved, sample_best_values, fstore, iteration):
    '''
    INPUTS:
    unsolved            : unsolved sudoku array
    sample_best_values              : input array
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
            a[i] = sample_best_values[x_count]
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
    fig.savefig('img/'+str(iteration) + '.png')
    return

def sample_error_col(sample_size,sample_matrix,f=cost_function):
    ''' p: population size '''
    
    # compute the duplicate for each set of random samples
    sample_num_duplicates = np.zeros((sample_size, 1))
    for i in range(sample_size): sample_num_duplicates[i, :] = f(sample_matrix[i, :])
    # concatenate random sample values and duplicate number
    sample_matrix = np.concatenate((sample_matrix, sample_num_duplicates), axis=1)
    return sample_matrix

def sample_min_error(sample_error_col_obj):
    return sample_error_col_obj[np.argmin(sample_error_col_obj[:, -1])] 
    

def evolution(f=cost_function, p = 50, it = 10, cull_percen = 0.5, mut_percen = 0.01, unsolved=unsolved,initial_condition=[0]):
    '''
    INPUTS:
    f               : sudoku cost function
    p               : population size
    it              : iterations
    cull_percen     : percentage of population to be culled
    mut_percen      : percentage chance of a mutation
    unsolved        : the unsolved sudoku vector
    
    OUTPUTS:
    sample_best_values          : the vector of optimised input values
    SudokuSolve.gif : a gif showing function value over time as
                        well as the current sudoku

    Default values
    f=cost_function
    p=1000
    it=1000
    cull_percen= 0.8
    mut_percen=0.05
    unsolved

    '''
    print("Solving Sudoku")
    empty_entries = unsolved.size - np.count_nonzero(unsolved) # count zeros
    bounds = np.tile([1, 9],(empty_entries,1))

    '''ORIGINAL SAMPLE of quasi-random Latin Hypercubre Sampling (LHS)'''
    sampling = LHS(xlimits=bounds)  # LHS
    sample_random_values = sampling(p).astype('int') #LHS sample given the population size
    
    '''EVALUATING FITNESSES'''
    # compute error (duplicated values) and add the last column
    sample_random_values=sample_error_col(sample_size=p, sample_matrix=sample_random_values)
    
    # determine the lowest duplicate set
    sample_best_values = sample_min_error(sample_random_values)
    
    if len(initial_condition)>1:
        sample_random_values[1]=initial_condition
        
    fstore = []
    val_store= []    
    for iteration in range(it):
        '''TOURNAMENT SELECTION: subset of random values'''
        sample_random_values_subset = np.zeros((int(p * (cull_percen)), empty_entries + 1))  # PARAMETER HERE (percentage to be kept)

        for new_count in range(len(sample_random_values_subset)):
            rnd.shuffle(sample_random_values)
            t_size = rnd.randint(1, 10)  # SORT OF PARAMETER HERE (tournament size)
            t = sample_random_values[:t_size, :] # select a subset of the random sample
            t_best = t[np.argmin(t[:, -1])] # determine the lowest duplicate set
            sample_random_values_subset[new_count, ] = t_best # fill subset with the best

        sample_random_values = copy.deepcopy(sample_random_values_subset)# selected subset
        
        '''COMPLEMENT WITH RANDOM CANDIDATES'''
        new_psize = p - len(sample_random_values)
        sampling = LHS(xlimits=bounds) # LHS
        sample_random_values_subset = sampling(new_psize).astype('int')  #LHS sample given the new size
        # compute the duplicate for each set of new random samples
        sample_random_values_subset=sample_error_col(new_psize, sample_random_values_subset)
        
        
        ''' COMPLETE SAMPLE'''
        # concatenate random sample values and duplicate number (selection+complement)
        sample_random_values = np.concatenate((sample_random_values_subset, sample_random_values), axis=0)
        
        # determine the best sample from selection+complement
        best_index = np.argmin(sample_random_values[:, -1])
        sample_best_values = sample_random_values[best_index]
        sample_best_values_val = sample_random_values[best_index, -1]
        
        print('iteration:'+str(iteration)+ ' errors:'+str(sample_best_values_val), end='\n')
        fstore.append(sample_best_values_val.astype('int'))
        val_store.append(sample_best_values.astype('int'))
        '''CROSSOVER HERE: change the order of the columns in each row'''
        
        
        
        rnd.shuffle(sample_random_values)
        '''MUTATION CODE HERE'''
        for i in range(len(sample_random_values)):
            for j in range(empty_entries):
                prob = rnd.uniform()
                #print(np.argmin(sample_random_values[:, -1]))
                if prob < mut_percen and best_index!=i :
                    sample_random_values[i, j] = rnd.uniform(bounds[j, 0], bounds[j, 1])
                    
        
        
        sample_random_values = sample_random_values[:, :-1] # delete the last column (duplicates number)
        sample_random_values = sample_error_col(p, sample_random_values) # calculate new errors
        
        
        # plot the current Sudoku and error per iteraton
        if it<5: plot_util(unsolved, sample_best_values, fstore, iteration)
    if it<5:   
        images = []
        for filename in range(it):
            images.append(imageio.imread('img/'+ str(filename) + '.png'))
            #os.remove('img/'+ str(filename) + '.png')
        imageio.mimsave('SudokuSolve.gif', images)
    
    return val_store

solved= [np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 9, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 0]),
         np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 8, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 9])]

#solved = evolution(f=cost_function, p = 50, it = 100, cull_percen = 1, mut_percen = 0.01, unsolved=unsolved)
for i in range(1000):
    solved = evolution(f=cost_function, p = 100, it = 100, cull_percen = 1, mut_percen = 0.05, unsolved=unsolved, initial_condition=solved[-1])
