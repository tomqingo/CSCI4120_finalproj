The evaluator script can evaluate the solution file format correctness, solution legality and computes the total displacement. 
The score will be calculated if the solution is legal. 

The basic usage is:
    python3 evaluator.py <input_file> <solution_file>

Also, the script is able to visualize the cells in the input or the solution file, by
    python3 evaluator.py <input_file> <solution_file> --plot <dir_saving_visualization>
The displacement vector for each cell can also be plotted by providing the --plot_displacement flag, 
    python3 evaluator.py <input_file> <solution_file> --plot <dir_saving_visualization> --plot_displacement

matplotlib need to be installed for performing visualization. 
It can be quite slow when generating the visualization for large benchmarks.
