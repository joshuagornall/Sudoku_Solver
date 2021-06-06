import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import copy
import numpy.random as rnd
import random
from celluloid import Camera 
import time
from IPython import get_ipython

get_ipython().magic('reset -sf') # delete all explorer files at the beginning

unsolved = np.array(
    [3, 9, 6, 0, 8, 0, 7, 1, 5, 0, 0, 7, 9, 6, 1, 0, 0, 3, 8, 4, 1, 0, 3, 0, 2, 0, 0, 6, 3, 0, 1, 0, 0, 5, 2, 0, 1, 0,
     0, 7, 2, 5, 3, 9, 0, 0, 5, 2, 6, 9, 3, 8, 0, 1, 4, 0, 8, 3, 7, 6, 0, 0, 2, 2, 0, 5, 8, 0, 9, 0, 3, 4, 9, 6, 3, 0,
     0, 2, 0, 7, 8])

def timing(function,times=1000):
    start                       = time.time()
    
    for i in range(1,times):
            function
    duration                    = time.time()- start
    return duration


def evaluation_function(x,unsolved=unsolved):
    # function to determine the duplicate values in the Sudoku
    # by rows, columns, and quadrant

    # x contains random numbers. size array[n = empty entries,]
    
    unsolved_copy               = copy.deepcopy(unsolved)   # copying the unsolved sudoku

    # filling the unsolved array with input values x
    x_count                     = 0
    for i,value in enumerate(unsolved_copy):
        if value == 0:
            unsolved_copy[i]    = x[x_count]
            x_count += 1
            
    # converting to 9x9 grid
    dim_sudoku                  = 9 # number of rows or columns in the Sudoku
    unsolved_copy               = np.reshape(unsolved_copy, (dim_sudoku, dim_sudoku))
    
    # Determinate duplicates
                                 
    # duplicates in rows
    num_duplicates_row          = 0  # initilise num_duplicates   
    for irow in range(dim_sudoku):
        row                     = unsolved_copy[irow, :] # select each row
        unique_values           = set(row) # find the unique values and sort them
        num_duplicates_row += dim_sudoku - len(unique_values)  #
    
    # duplicates in columns
    num_duplicates_col          = 0
    for icol in range(dim_sudoku):
        col                     = unsolved_copy[:, icol] # select each column
        unique_values           = set(col) # find the unique values and sort them
        num_duplicates_col += dim_sudoku - len(unique_values)  # 

    # duplicates in each 3x3 grid quadrant
    num_duplicates_quad         = 0
    for i in range(3):
        for j in range(3):
            kernal              = unsolved_copy[i * 3:(i * 3) + 3,
                                   j * 3:(j * 3) + 3] # select each subwindow
            kernal              = np.reshape(kernal, dim_sudoku) # transform to vector
            unique_values       = set(kernal) # find the unique values and sort them
            num_duplicates_quad += dim_sudoku - len(unique_values)
    num_duplicate               = num_duplicates_row+num_duplicates_col+num_duplicates_quad
            
    return num_duplicate,num_duplicates_row,num_duplicates_col,num_duplicates_quad

def plot_into_gif(unsolved, solved,number_steps=100):
    '''
    INPUTS:
    unsolved            : unsolved sudoku array
    solved              : partial solutions of sudoku and 4 last columns with respective errors
    
    OUTPUTS:
    animation           : animation file to be exported in different formats
    GIF file Sudoku_solution.gif
    '''
    errors_best_sample_store    = [solved[i][range(-4,0)]for i in range(len(solved))]
    iterations                  = np.size(solved,0) 
    
    # filling empty sudoku
    empty_array                 = np.zeros_like(unsolved)
    
    for i in range(len(unsolved)):  
        if unsolved[i] == 0:
            empty_array[i]      = 1
    empty_array = np.reshape(empty_array, (9, 9))
    empty_array                 = np.transpose(empty_array)
    
    
    # fill zeros with sample values
    fig, axs                    = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(left = 0.05, right = 0.95)
    camera                      = Camera(fig)# the camera gets our figure
    
    axs[0].set_title('Current best Sudoku')
    axs[0].axis('off')
    axs[1].set_xlabel('Iteration')
    
    axs[1].grid('True')
    axs[1].set_ylabel('Errors')
    
    for iteration in np.linspace(0,iterations,number_steps).astype('int'):
        sudoku_grid             = copy.deepcopy(unsolved)
        print(iteration)
        sample_best_values      = solved[iteration-1]
        
        x_count = 0
        for i in range(len(sudoku_grid)):
            if sudoku_grid[i] == 0:
                sudoku_grid[i]  = sample_best_values[x_count]
                x_count += 1
                
        sudoku_grid             = np.reshape(sudoku_grid, (9, 9))
        sudoku_grid             = np.transpose(sudoku_grid)
        
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
                    axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_grid[i, j], fontweight='extra bold')
                else:
                    axs[0].text(i + 0.45, 9 - 0.65 - j, sudoku_grid[i, j])
                    
        new_error               = errors_best_sample_store[:iteration]
        x                       = np.linspace(0, len(new_error), len(new_error))
        axs[1].plot(x, [new_error[i][0] for i in range(len(new_error))], color='k')
        axs[1].plot(x, [new_error[i][1] for i in range(len(new_error))], color='g')
        axs[1].plot(x, [new_error[i][2] for i in range(len(new_error))], color='r')
        axs[1].plot(x, [new_error[i][3] for i in range(len(new_error))], color='b')
        
        if iteration==iterations-1:print(new_error)
        axs[1].legend(['Total errors', 'Row errors', 'Column errors', 'Quadrant errors'], loc='upper right')
        
        #fig.savefig('img/'+str(iteration) + '.png')
        camera.snap()
    animation = camera.animate()
    animation.save('Sudoku_solution.gif',dpi=100,extra_args=['-loop','1'])
    return animation


def sample_error_col(sample_size,sample_matrix):
    ''' population_size: population size '''
    # compute the duplicate for each set of random samples
    sample_num_duplicates       = np.zeros((sample_size, 4))
    
    for i in range(sample_size): sample_num_duplicates[i, :] = evaluation_function(sample_matrix[i, :])
    # concatenate random sample values and duplicate number
    sample_matrix               = np.concatenate((sample_matrix, sample_num_duplicates), axis=1)
    return sample_matrix

def sample_min_error(sample_error_col_obj):
    return sample_error_col_obj[np.argmin(sample_error_col_obj[:, -4])]


def random_permutation(iterable,sample_number=10):
    "Random selection from itertools.permutations(iterable, r)"
    pool                        = tuple(iterable)
    r                           = len(pool)
    return np.array([random.sample(pool, r) for i in range(sample_number)])

def tourment_selection(sample_random_values,population_size,subset_percentage,initial_guess):
    new_size                    = int(population_size * (subset_percentage))
    sample_random_values_subset = np.zeros((new_size, np.size(sample_random_values,1)))  # PARAMETER HERE (percentage to be kept)
    tournament_sample           = np.zeros_like(sample_random_values)
    '''GENERATE SUBSET BASED ON SUBSET PERCENTAGE '''
    for new_count in range(new_size) :
        rnd.shuffle(sample_random_values)
        tournament_sample       = sample_random_values[:rnd.randint(1, 10), :] # subset of the random sample
        sample_random_values_subset[new_count, ] = sample_min_error(tournament_sample) # fill subset with the best
        
    '''COMPLEMENT WITH RANDOM CANDIDATES'''
    complement_size             = population_size - len(sample_random_values_subset)
    complement_sample           = random_permutation(initial_guess,complement_size)  #sample given the new size
    complement_sample           = sample_error_col(complement_size, complement_sample) # compute the duplicate for each set of new random samples
    
    ''' COMPLETE SAMPLE'''
    sample_random_values        = np.concatenate((complement_sample, sample_random_values_subset), axis=0) # concatenate tournament subset + complement
    return sample_random_values

def genethic_algorithm(population_size = 20, n_iterations = 100000, subset_percentage = 0.999, mut_percen = 0.05, unsolved=unsolved,initial_condition=[0]):
    '''
    GENETIC ALGORITHM TO SOLVE THE SUDOKU GRID
    
    INPUTS
    
    population_size             : population size
    n_iterations                : number of iterations
    subset_percentage           : percentage of population in the tournament
    mut_percen                  : probability of mutation
    unsolved                    : the unsolved sudoku vector (81,1)
    initial_condition           : initial guess of the unsolved numbers
    
    OUTPUTS
    
    sample_best_values          : a matrix with each iteration value and errors

    DEFAULT VALUES
    
    population_size             = 20
    n_iterations                = 1000
    subset_percentage           = 0.999
    mut_percen                  = 0.05
    unsolved                    = unsolved
    initial_condition           = [0]
    '''
    print("Solving Sudoku")

    # INITIALISING SAMPLE OF CANDIDATES
    
    unique_items, unique_counts = np.unique(unsolved, return_counts=True) # count unique 0,1,2,3,4,5,6,7,8,9
    empty_entries               = unique_counts[0] # count zeros
    initial_guess               = np.repeat(unique_items[1:],9-unique_counts[1:],axis=0)
    sample_random_values        = random_permutation(initial_guess,population_size)
    
    # INITIALISE VARIABLES
    sample_best_values_store    = [] # variable to store the best candidates and its errors 
    # COMPUTE ERRORS (duplicated values) AND ADD THEM AS COLUMNS
    sample_random_values        = sample_error_col(sample_size=population_size, sample_matrix=sample_random_values)
    
    # CHOOSE SET WITH LESS ERRORS
    sample_best_values          = sample_min_error(sample_random_values)
    
    # ADD INITIAL CONDITION TO THE SAMPLE (IF EXISTS)
    if len(initial_condition)>1: sample_random_values[1] = initial_condition
        
    for iteration in range(n_iterations):
        # SUBSET THE POPULATION SAMPLE (TOURNAMENT)
        sample_random_values    = tourment_selection(sample_random_values,population_size,subset_percentage,initial_guess)
        sample_best_values      = sample_min_error(sample_random_values)
        sample_best_values_store.append(sample_best_values)
        
        # TERMINATION CONDITION
        errors_best_sample      = sample_best_values[range(-4,0)]
        if iteration%100==0: print('iteration:'+str(iteration)+ ' errors:'+str(errors_best_sample), end='\n') # display errors per iteration
        if errors_best_sample[0]==0: break # if zero errors then end the program
        
        # MUTATION
        index_sample_best       = np.argmin(sample_random_values[:, -4]) # row index of the sample with less errors
        for i in range(len(sample_random_values)):
            for j in range(empty_entries):
                prob            = rnd.uniform()
                if prob < mut_percen and index_sample_best!=i:
                    sample_random_values[i, j] = rnd.randint(1,10)
                    
        sample_random_values    = sample_error_col(population_size, sample_random_values[:, :-4]) # initial sample set for the next iteration
        
    return sample_best_values_store


#solved= [np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 9, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 0, 0,0,0]),
#         np.array([2, 4, 5, 2, 4, 8, 5, 7, 6, 9, 8, 4, 8, 7, 8, 4, 6, 7, 4, 1, 9, 5, 7, 1, 6, 4, 5, 1, 9])]

solved                          = genethic_algorithm(population_size   = 20,
                                                     n_iterations      = 100000,
                                                     subset_percentage = 0.999,
                                                     mut_percen        = 0.05,
                                                     unsolved          = unsolved)
animation=plot_into_gif(unsolved,solved,100)





