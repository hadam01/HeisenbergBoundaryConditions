##
## main.py
## Husam Adam
## Purpose - This file explores different boundary conditons
## on the Heisenberg antiferromagnet through various plots.

# Import all of our run functions from boundary condition applications
import matplotlib.pyplot as plt
import numpy as np

# Runs NetKets periodic code with supercell coefficient paramaters (L for run_netket and L1&L2 for run_netket2)
from NetKetPBC import run_square as run_netket_square
from NetKetPBC import run_rectangle as run_netket_rectangle

# Runs any pair of boundary conditions with supercell coefficient and 
# theta parameters (sup_coefficient, sup_coefficient2, theta, and theta2)
from AllBC import run as run_all

# General lattice utility functions
from Lattice_Utils import generate_basis_states, store_all_neighbors_with_type, make_lattice

""" 
Purpose - Reads the first 10 eigenvalues from a given file
Parameters - The filename you're reading from
Returns - An array of eigenvalues
"""
def read_eigenvalues(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    eigenvalues = []
    for i in range(0, 10):
        line = lines[i].strip()
        eigenvalues.append(float(line))
    return np.array(eigenvalues)

"""
Purpose - Reads in all of the eigenvectors from a given file
Parameters - The filename you're reading from
Returns - An array of eigenvectors 
"""
def read_eigenvectors(filename):
    eigenvectors = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split()
            eigenvector = np.array([float(value) for value in values])
            eigenvectors.append(eigenvector)
    return np.array(eigenvectors)

""" 
Purpose - Calculates the dot product between two llists of eigenvectors
Parameters - The lists of eigenvectors you are taking the dot products of
Return - The dot products as an array
NOTE: Make sure the eigenvector dimensions are the same
"""
def calculate_dot_product(eigenvectors1, eigenvectors2):
    dot_products = []
    for vec1, vec2 in zip(eigenvectors1, eigenvectors2):
        dot_product = np.dot(vec1, vec2)
        dot_products.append(dot_product)
    return np.array(dot_products)

""" 
Purpose - Makes a plot comparing the lowest 10 energies using different boundary conditions
Parameters - The supercell coefficients
Returns - Nothing; generates the plot
NOTE: You need to change the sup_coefficients and energy per sites manually for this function
"""
def plot_eigenvalues(sup_coefficient, sup_coefficient2):
    # Find all eigenvalues
    periodic_eigen, periodic_vec = run_all(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2, rowBC='periodic', columnBC='periodic', theta=None, theta2=None)
    antiperiodic_eigen, antiperiodic_vec = run_all(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2, rowBC='antiperiodic', columnBC='antiperiodic', theta=None, theta2=None)
    netket_eigen, netket_vec = run_netket_rectangle(L1=sup_coefficient, L2=sup_coefficient2)
    open_eigen, open_vec = run_all(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2, rowBC='open', columnBC='open', theta=None, theta2=None)

    # Run the random boundary condition multiple times
    random_eigens = []
    for i in range(10):  # Run 10 times for statistical significance
        random_eigen, random_vec = run_all(sup_coefficient=4, sup_coefficient2=4, rowBC='random', columnBC='random', theta=None, theta2=None)
        random_eigens.append(random_eigen[:10]/16)
    
    random_eigens = np.array(random_eigens) 
    
    mean_random_eigen = np.mean(random_eigens, axis=0)
    std_random_eigen = np.std(random_eigens, axis=0)
    
    # Only take first 10 eigenvalues
    periodic_eng_per_site = periodic_eigen[:10]
    antiperiodic_eng_per_site = antiperiodic_eigen[:10]
    netket_eng_per_site = netket_eigen[:10]
    open_eng_per_site = open_eigen[:10]

    # Make the plot
    plt.figure(figsize=(8, 6))
    plt.plot(netket_eng_per_site, marker='o', label='NetKet', alpha=0.7)
    plt.plot(periodic_eng_per_site, marker='o', label='Periodic', alpha=0.7)
    plt.plot(antiperiodic_eng_per_site, marker='o', label='Antiperiodic', alpha=0.7)
    plt.plot(open_eng_per_site, marker='o', label='Open', alpha=0.7)
    
    plt.errorbar(range(10), mean_random_eigen, yerr=std_random_eigen, marker='o', label='Random (Mean ± Std)', alpha=0.7, capsize=5)

    plt.xlabel('Eigenvalue Index', fontsize=8)
    plt.ylabel('Energy per Site', fontsize=8)
    plt.title('Eigenvalues for Different Boundary Conditions (4x4 Lattice)', fontsize=10)
    plt.legend()
    plt.show()

"""
Purpose - Plots dot products given a list of dot products and labels
Parameters - List of dot products and list of labels
Returns - Nothing; makes a plot
"""
def plot_dot_products(dot_products_list, labels):
    # Find the dot products
    plt.figure(figsize=(10, 6))
    for i in range(len(dot_products_list)):
        dot_products = dot_products_list[i]
        x = list(range(len(dot_products)))
        plt.scatter(x, dot_products, label=labels[i])

    # Make the plot
    plt.xlabel('Eigenvector Index')
    plt.ylabel('Dot Product')
    plt.title('Eigenvector Dot Products')
    plt.legend()
    plt.grid(True)
    plt.show()

"""
Purpose - Makes a scatter plot of 2 eigenvectors' coefficients given the vectors
Parameters - The two lists of eigenvectors and  alist of labels
Returns - Nothing; makes a plot
"""
def scatter_plot(evec, evec2, labels):
    # Make the scatter plot
    plt.scatter(evec, evec2, alpha=0.5)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Scatter Plot of Eigenvector Coefficients for Different Boundary Conditions')
    plt.show()

"""
Purpose - Gets all of the eigenvectors from stored files
Parameters - None (filenames are already assumed)
Returns - All the lists of eignvectors from the files they were stored in
"""
def get_all_evecs():
    # Read eigenvectors from the files they're assumed to be stored in
    periodic_vector = read_eigenvectors(filename='PeriodicVector.txt')
    netket_vector = read_eigenvectors(filename='NetKetVector.txt')
    antiperiodic_vector = read_eigenvectors(filename='AntiVector.txt')
    open_vector = read_eigenvectors(filename='OpenVector.txt')
    random_vector = read_eigenvectors(filename='RandomVector.txt')
    random_vector2 = read_eigenvectors(filename='RandomVector2.txt')

    return periodic_vector, netket_vector, antiperiodic_vector, open_vector, random_vector, random_vector2

"""
Purpose - Calculates the dot products between different vectors and calls for them to be plotted
Parameters - All the lists of eigenvectors (periodic, netket, antiperiodic, open, and two random eigenvectors)
Returns - Nothing; makes a plot
"""
def compare_dot_products(periodic_vector, netket_vector, antiperiodic_vector, open_vector, random_vector, random_vector2):
    periodic_dot1 = calculate_dot_product(periodic_vector, netket_vector)
    periodic_dot2 = calculate_dot_product(periodic_vector, antiperiodic_vector)
    periodic_dot3 = calculate_dot_product(periodic_vector, open_vector)
    periodic_dot4 = calculate_dot_product(periodic_vector, random_vector)
    periodic_dot5 = calculate_dot_product(periodic_vector, random_vector2)

    dot_products_list = [periodic_dot1, periodic_dot2, periodic_dot3, periodic_dot4, periodic_dot5]
    labels = ['Periodic vs NetKet', 'Periodic vs Antiperiodic', 'Periodic vs Open', 'Periodic vs Random', 'Periodic vs Random (2)']
    plot_dot_products(dot_products_list=dot_products_list, labels=labels)

"""
Purpose - Sorts an eigenvector in either an 'ascending' or 'descending' order
Parameters - The eigenvector list you want to sort and the order you want to sort it
Returns - A sorted list of eigenvectors
"""
def sort_eigenvectors(eigenvector, order):
    if order == 'ascending':
        sorted_vector = np.sort(eigenvector)
    elif order == 'descending':
        sorted_vector = np.sort(eigenvector)[::-1]
        
    return sorted_vector

"""
Purpose - Stores a sorted eigenvector in a given filename
Parameters - The sorted eigenvector and the filename
Returns - Nothing; stores the sorted eigenvectors
"""
def store_sorted(sorted_eigenvector, filename):
    # Open a file in write mode
    with open(filename, 'w') as file:
        # Store all of the eigenvectors
        for eigenvector in sorted_eigenvector:
            for value in eigenvector:
                file.write(f"{value} ")
            file.write("\n")

"""
Purpose - Plot the ground states energy per site vs lattice size for different boundary conditions
Parameters - None
Returns - Nothing; generates the plot
"""
def run_diff_lattices():
    # Make lists that will be useful for storing our data and supercell coefficients
    lattice_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    periodic_gs = []
    anti_gs = []
    mix_gs = []
    mix_gs2 = []
    labels = []
    
    # Run periodic, antiperiodic, periodic rows vs antiperiodic columns, and antiperiodic rows vs periodic
    # columns for all supercell coefficients my code can handle
    for sup_coefficients in lattice_sizes:
        antivalue, antivector = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='antiperiodic', columnBC='antiperiodic', theta=None, theta2=None)
        periodicvalue, periodicvector = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='periodic', columnBC='periodic', theta=None, theta2=None)
        mixvalue, mixvector = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='periodic', columnBC='antiperiodic', theta=None, theta2=None)
        mixvalue2, mixvector2 = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='antiperiodic', columnBC='periodic', theta=None, theta2=None)
        periodic_gs.append(periodicvalue[0]/(sup_coefficients[0]*sup_coefficients[1]))
        anti_gs.append(antivalue[0]/(sup_coefficients[0]*sup_coefficients[1]))
        mix_gs.append(mixvalue[0]/(sup_coefficients[0]*sup_coefficients[1]))
        mix_gs2.append(mixvalue2[0]/(sup_coefficients[0]*sup_coefficients[1]))
        labels.append(f'{sup_coefficients[0]}x{sup_coefficients[1]}')

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot the ground state eigenvalues for each type of boundary condition
    plt.plot(labels, periodic_gs, marker='o', label='Periodic', alpha=0.7)
    plt.plot(labels, anti_gs, marker='o', label='Antiperiodic', alpha=0.7)
    plt.plot(labels, mix_gs, marker='o', label='Periodic Row vs Antiperiodic Column', alpha=0.7)
    plt.plot(labels, mix_gs2, marker='o', label='Periodic Column vs Antiperiodic Row', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Lattice Size')
    plt.ylabel('Energy per Site')
    plt.title('Ground States for Different Lattice Sizes')

    # Add grid for better readability
    plt.grid(True)

    # Adjust legend to avoid clipping and overlap
    plt.legend(title='Boundary Conditions', fontsize=6, loc='best', frameon=True, markerscale=0.6)

    # Adjust layout to ensure everything fits
    plt.tight_layout(pad=4.0)

    # Show the plot
    plt.show()

"""
Purpose - Plots the ground state energy per site of different lattice configurations as a function of the twist angle theta
Parameters - None
Return - Nothing; generates a plot
"""
def run_diff_thetas():
    # Make lists to store our data
    thetas = []
    ground_states = []
    ground_states2 = []

    # Run different thetas and find the ground states
    theta = 0
    while theta <= 2*np.pi:
        eigenvalues, eigenvectors = run_all(sup_coefficient=4, sup_coefficient2=4, theta=theta, theta2=theta)
        ground_states.append(eigenvalues[0]/16)
        thetas.append(theta/np.pi)
        theta += np.pi/4

    theta = 0
    while theta <= 2*np.pi:
        eigenvalues, eigenvectors = run_all(sup_coefficient=3, sup_coefficient2=3, theta=theta, theta2=theta)
        ground_states2.append(eigenvalues[0]/9)
        theta += np.pi/4
    
    # Make the plot
    plt.figure(figsize=(6, 4))
    plt.plot(thetas, ground_states, marker='o', label='4x4 Lattice', color='blue', alpha =0.7)
    plt.plot(thetas, ground_states2, marker='o', label='3x3 Lattice', color='red', alpha=0.7)

    plt.xlabel('Theta (units of pi)')
    plt.ylabel('Ground State Energy per Site')
    plt.title('Ground State Energy per Site vs. Theta')
    plt.legend()
    plt.legend(fontsize=10)
    plt.tight_layout(pad=3.0)
    plt.savefig('groundvstheta.png')
    plt.close()

"""
Purpose - Plots the energy per site as a function of theta 2 for a fixed theta
Parameters - The supercell coefficients
Return - Nothing; generates a plot
"""
def run_rectangular_twist(sup_coefficient, sup_coefficient2, plot_title):
    theta = 0
    # Make lists to store our data
    thetas = []
    ground_states = []
    all_theta2_values = []
    
    # Hold theta fixed for all theta2 values
    while theta <= 2 * np.pi:
        theta2 = 0
        curr_ground_states = []
        thetas.append(theta/np.pi)
        theta2_values = []
        # Find ground states for theta and theta 2 values
        while theta2 <= 2 * np.pi:
            eigenvalues, eigenvectors = run_all(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2, rowBC='twisted', columnBC='twisted', theta=theta, theta2=theta2)
            curr_ground_states.append(eigenvalues[0]/(sup_coefficient2*sup_coefficient))
            theta2_values.append(theta2/np.pi)
            theta2 += np.pi / 4
        
        ground_states.append(curr_ground_states)
        all_theta2_values.append(theta2_values)
        theta += np.pi / 4

    plt.figure(figsize=(6, 4))

    for i, theta in enumerate(thetas):
        plt.plot(all_theta2_values[i], ground_states[i], marker='o', label=f'Theta = {theta:.2f}π', alpha=0.7)

    plt.xlabel('Theta 2 (units of π)')
    plt.ylabel('Energy per Site')
    plt.title(plot_title)
    plt.legend(title='Fixed Theta Values', fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout(pad=3.0)
    plt.show()

"""
Purpose - Plots the energy per site as a function of theta 2 for a fixed theta
Parameters - A list of sites' neighbors
Return - The total number of interactions given from the list of neighbors
"""
def find_num_interactions(neighbors_data):
    seen = []
    num_interactions = 0
    # Loop through lattice sites and then neighbors
    for k in range(0, len(neighbors_data)):
        neighbors = neighbors_data[k]
        for j, neighbor_type in neighbors:
            # Update the Hamiltonian if we see a new interaction
            if (k, j, neighbor_type) not in seen and (j, k, neighbor_type) not in seen: 
                num_interactions += 1
                seen.append((k, j, neighbor_type))
    return num_interactions

"""
Purpose - Calculates spin-spin correlation for a given basis, eigenvectors, and set of neighbors
Parameters - The basis states, list of eigenvectors, and list of neighbors
Return - The associated correlation given the above parameters
"""
def calculate_spin_correlation(basis, eigenvectors, neighbors_data):
    basis_correlations = []
    curr_contribution = 0
    total_contribution = 0

    # Loop through each basis state, then each site (i), and each neighbor(j) and calculate the 
    # contribution to the spin-spin correlation
    for state in basis:
        for i, neighbors in enumerate(neighbors_data):
            for j, neighbor in enumerate(neighbors):
                neighbor_index = neighbors_data[i][j][0]
                curr_contribution = state[i] * state[neighbor_index]
                total_contribution += curr_contribution
                curr_contribution = 0
        basis_correlations.append(total_contribution)
        total_contribution = 0
    correlation = 0

    # Multiply by the ground state eigenvector
    ground_state_vec = eigenvectors[:, 0]
    for i in range(0, len(ground_state_vec)):
        correlation += basis_correlations[i] * (np.conj(ground_state_vec[i]) * ground_state_vec[i])
    num_interactions = find_num_interactions(neighbors_data=neighbors_data)

    # Return the final value
    return np.real(correlation/num_interactions)

"""
Purpose - Plots spin-spin correlation for different lattice sizes with different boundary conditions
Parameters - None
Return - Nothing; generates a plot
NOTE: You can replace the PBCvOBC, PBCvRBC, etc with any combination of boundary conditions.
"""
def plot_correlations():
    # Define our supercell coefficients
    lattice_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4)]
    boundary_conditions = ["PBCvOBC", "PBCvRBC"]
    correlations = np.zeros((len(lattice_sizes), len(boundary_conditions)))

    # Loop through all supercell coefficients and find the eigenvectors
    for idx, sup_coefficients in enumerate(lattice_sizes):
        lattice = make_lattice(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1])
        basis = generate_basis_states(n=sup_coefficients[0]*sup_coefficients[1])
        PBCvOBC_eigenvalues, PBCvOBC_eigenvectors = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='periodic', columnBC='open', theta=None, theta2=None)
        PBCvRBC_eigenvaues2, PBCvRBC_eigenvectors2 = run_all(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC='periodic', columnBC='random', theta=None, theta2=None)
        neighbors_data = store_all_neighbors_with_type(lattice=lattice, N=sup_coefficients[0]*sup_coefficients[1], sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1])
        all_evecs = [PBCvOBC_eigenvectors, PBCvRBC_eigenvectors2]

        # Calculate the spin-spin correlation and store the data 
        for i, vec in enumerate(all_evecs):
            correlation = calculate_spin_correlation(basis=basis, eigenvectors=vec, neighbors_data=neighbors_data)
            correlations[idx, i] = correlation

    # Make the plot
    plt.figure(figsize=(8, 6))
    for i, bc in enumerate(boundary_conditions):
        plt.plot([f"{size[0]}x{size[1]}" for size in lattice_sizes], correlations[:, i], marker='o', label=bc, alpha=0.7)
    plt.xlabel('Lattice', fontsize=8)
    plt.ylabel('Spin-Spin Correlation', fontsize=8)
    plt.title('Spin-Spin Correlations for Different Boundary Conditions', fontsize=10)
    plt.xticks(fontsize=8) 
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout(pad=4) 
    plt.grid(True)
    plt.savefig('correlation_plot.png')
    plt.close()
    
"""
Purpose - Runs any of the above functions
Parameters - None
Returns - Nothing; makes the plot you want
"""
def run():
    print("Welcome to the run function in main! This is where you can generate any of the above plots.")

run()

# NOTE: This is code I'm saving here for me. I used this code to compare my results from my more modularized approach (the ones
# you have) to the results from my previous code where I dealt with each boundary condition and its Hamiltonian individually on a case
# by case basis. If you would like to see the other code - feel free to reach out at husam.adam@tufts.edu!


# def check_one_bc(sup_coefficients, rowBC, columnBC, title, run_func, reference_func):
#     print(f"Checking {title} boundary conditions: ")
    
#     # Run the test function with the specified boundary conditions
#     eigen, vec = run_func(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC=rowBC, columnBC=columnBC)
    
#     # Run the reference function with the specified boundary conditions
#     eigen2, vec2 = reference_func(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1])
    
#     # Calculate the difference
#     difference = abs(eigen[0] - eigen2[0])
    
#     threshold = 1e-6
#     # Print the error statement if the difference is greater than the threshold
#     if difference > threshold:
#         print(f"Error found in {title} check for rowBC: {rowBC}, columnBC: {columnBC}.")
#         print(f"Computed eigenvalue: {eigen[0]}")
#         print(f"Reference eigenvalue: {eigen2[0]}")
#         print(f"Difference: {difference}")

# def check_one_twist(sup_coefficients, rowBC, columnBC, title, run_func, reference_func, theta):
#     print(f"Checking {title} boundary conditions: ")
    
#     # Run the test function with the specified boundary conditions
#     eigen, vec = run_func(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], rowBC=rowBC, columnBC=columnBC, theta=theta, theta2=theta)
    
#     # Run the reference function with the specified boundary conditions
#     eigen2, vec2 = reference_func(sup_coefficient=sup_coefficients[0], sup_coefficient2=sup_coefficients[1], theta=theta)
    
#     # Calculate the difference
#     difference = abs(eigen[0] - eigen2[0])
    
#     threshold = 1e-6
#     # Print the error statement if the difference is greater than the threshold
#     if difference > threshold:
#         print(f"Error found in {title} check for rowBC: {rowBC}, columnBC: {columnBC}.")
#         print(f"Computed eigenvalue: {eigen[0]}")
#         print(f"Reference eigenvalue: {eigen2[0]}")
#         print(f"Difference: {difference}")

# def check_all_bc_periodic():
#     lattice_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
#     functions = [[run_periodic, 'periodic'], [runPBCvABC, 'antiperiodic'], [runPBCvRBC, 'random'], [runPBCvOBC, 'open']]
    
#     for sup_coefficients in lattice_sizes:
#         rowBC = 'periodic'
#         for function, columnBC in functions:
#             title = f"{rowBC}-{columnBC}"
            
#             # Pass the functions that handle the specific boundary conditions
#             check_one_bc(
#                 sup_coefficients=sup_coefficients, 
#                 rowBC=rowBC, 
#                 columnBC=columnBC, 
#                 title=title,
#                 run_func=run_all,  # Replace with the actual function for your case
#                 reference_func=function
#             )

# def check_all_bc_anti():
#     lattice_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
#     functions = [[run_anti, 'antiperiodic'], [runABCvPBC, 'periodic']]
    
#     for sup_coefficients in lattice_sizes:
#         rowBC = 'antiperiodic'
#         for function, columnBC in functions:
#             title = f"{rowBC}-{columnBC}"
            
#             # Pass the functions that handle the specific boundary conditions
#             check_one_bc(
#                 sup_coefficients=sup_coefficients, 
#                 rowBC=rowBC, 
#                 columnBC=columnBC, 
#                 title=title,
#                 run_func=run_all,  # Replace with the actual function for your case
#                 reference_func=function
#             )

# def check_all_bc_twisted():
#     lattice_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
#     functions = [[run_twist, 'twisted'], [runTBCvABC, 'antiperiodic'], [runTBCvOBC, 'open'], [runTBCvRBC, 'random']]
#     thetas = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
#     for sup_coefficients in lattice_sizes:
#         rowBC = 'twisted'
#         for function, columnBC in functions:
#             for theta in thetas:
#                 title = f"{rowBC}-{columnBC}"
                
#                 # Pass the functions that handle the specific boundary conditions
#                 check_one_twist(
#                     sup_coefficients=sup_coefficients, 
#                     rowBC=rowBC, 
#                     columnBC=columnBC, 
#                     title=title,
#                     run_func=run_all,  # Replace with the actual function for your case
#                     reference_func=function, theta=theta
#                 )