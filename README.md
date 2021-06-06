<h1>A sudoku solver tool that utilises stochastic optimisation (genetic algorithm). 💻 </h1>

![Visualisation of solving](https://github.com/Fco-Jara/Sudoku_Solver/blob/master/Sudoku_solution.gif)

<h2>Function Use</h2>
The code takes an unsolved Sudoku in the form of a vector (array of length 81), formatted from left to right, top to bottom.
Empty values are represented by zeros. 
The function genetic_algorithm resolves the grid testing random candidates taken from the pool. 
Given the initial unsolved grid is possible to derive the remaining numbers and its frequency, due to each number repeats 9 times exactly in the grid.
The pool is all the possible permutations of those remaining numbers.

The function plot_to_gif takes the "unsolved"" Sudoku vector and the "solved" matrix (columns: unknown candidates + 4 columns of errors, rows: iterations), which contains the best solution and errors in each iteration. 
The left panel of the plot is the 9 x 9 grid, the evolution of the errors (by columns, rows, quadrants, total) in each iteration are displayed in the right panel.
