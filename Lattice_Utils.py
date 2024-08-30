##
## Lattice_Utils.py
## Husam Adam
## Purpose - This file contains utility functions for constructing a lattice for
## a 2D spin model, handling neighbors, generating basis states, analyzing lattices,
## and functions needed for special boundary conditions.

# Import lattice2D class from Louis' code
from lattice2D import Lattice2D
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
plt.rcParams['figure.dpi']=300

"""
Purpose - Defines a lattice given the supercell coefficients
Paramaters - Supercell coefficients
Returns - The lattice
"""
def make_lattice(sup_coefficient, sup_coefficient2):
    # Use Louis' code to make our lattice
    primitive_lattice_vectors = np.array([[1,0],[0,1]])
    atom_coords = np.array([[0,0]])
    supercell_coefficients = np.array([sup_coefficient,0,0,sup_coefficient2])
    lattice = Lattice2D(primitive_lattice_vectors, atom_coords, supercell_coefficients)
    return lattice

"""
Purpose - Plots the given latttice
Paramaters - Lattice
Returns - Nothing; plots the lattice
"""
def plot_lattice(lattice):
    lattice.plot_supercell(padding = 0.5)
    plt.show()

"""
Purpose - Writes basis states to a file 
Parameters - Basis states and the name of the file you want to write to
Returns - Nothing; adds basis states to file
"""
def write_basis_states(filename, basis_states):
    with open(filename, 'w') as f:
        for state in basis_states:
            f.write(' '.join(map(str, state)) + '\n')

"""
Purpose - Generate basis states with n/2 spin-up (1) and n/2 spin-down (-1) configurations
Parameters - The number of sites (n)
Returns - An array with all of your basis states
NOTE: For an odd number of spins - we choose one extra spin down
"""
def generate_basis_states(n):
    basis_states = []
    half_n = n // 2
    
    # Find every combination of basis states with equal numbers of spin downs and ups
    for comb in itertools.combinations(range(n), half_n):
        state = np.ones(n, dtype=int)
        state[list(comb)] = -1
        basis_states.append(state)
    
    return np.array(basis_states)

"""
Purpose - Gets the periodic neighbors of all the sites on your lattice
Parameters - Current site index, the lattice, the number of sites (N), and the supercell coefficients
Returns - A list of all the periodic neighbors
"""
def get_periodic_neighbors(site_index, lattice, N, sup_coefficient, sup_coefficient2):
    # Store the neighbors in a list
    neighbors = []

    # Loop through all of the sites (i) and check if i is a periodic neighbor to the site we were given (site_index)
    for i in range(0, N):
        if i != site_index:
            site = tuple(lattice.sites[site_index])
            site2 = tuple(lattice.sites[i])
            if lattice.periodic_distance(site, site2) == 1 and lattice.distance(site, site2) != 1:
                neighbors.append(i)
            
            # Handle the 2xN cases in a special way, such that each site ends up with four neighbors
            # no matter the supercell coefficients
            elif sup_coefficient == 2 and sup_coefficient2 == 2:
                if lattice.periodic_distance(site, site2) == 1 and lattice.distance(site, site2) == 1:
                    neighbors.append(i)
            elif sup_coefficient == 2:
                if check_rows_periodic(lattice=lattice, j=site_index, k=i, sup_coefficient=sup_coefficient) and lattice.periodic_distance(site, site2) == 1:
                    neighbors.append(i)
            elif sup_coefficient2 == 2:
                if check_columns_periodic(lattice=lattice, j=site_index, k=i, sup_coefficient2=sup_coefficient2) and lattice.periodic_distance(site, site2) == 1:
                    neighbors.append(i)
    return neighbors

"""
Purpose - Gets the direct neighbors of all the sites on your lattice
Parameters - Current site index, the lattice, and the number of sites (N)
Returns - a list of all the direct neighbors
"""
def get_direct_neighbors(site_index, lattice, N):
    # Store the neighbors in a list
    neighbors = []

    # Loop through all of the sites (i) and check if the site we were given is a direct neighbor
    for i in range(N):
        if i != site_index:
            site = tuple(lattice.sites[site_index])
            site2 = tuple(lattice.sites[i])
            if lattice.distance(site, site2) == 1:
                neighbors.append(i)
    return neighbors

"""
Purpose - Prints the number of neighbors each site has and what sites are its neighbors
Paramaters - The list of neighbors
Returns - Nothing; prints each sites neighbors
"""
def print_neighbor_counts(real_neighbors_data):
    for i, neighbors in enumerate(real_neighbors_data):
        print(f"Site {i} has {len(neighbors)} neighbors: {neighbors}")

"""
Purpose - Stores all of the neighbors in your lattice in a 2D array
Paramaters - The lattice, number of sites (N), and supercell coefficients
Returns - A list of neighbors with the direct or periodic stored depending on the type 
of neighbor they are
"""
def store_all_neighbors_with_type(lattice, N, sup_coefficient, sup_coefficient2):
    # Store the neighbors in a list of lists
    real_neighbors_data = [[] for i in range(N)]

    # Find every sites' (i) periodic and direct neighbors
    for i in range(N):
        neighbors = get_periodic_neighbors(i, lattice, N, sup_coefficient, sup_coefficient2)
        real_neighbors_data[i].extend((n, 'periodic') for n in neighbors)
        neighbors = get_direct_neighbors(i, lattice, N)
        real_neighbors_data[i].extend((n, 'direct') for n in neighbors)
    return real_neighbors_data

"""
Purpose - Stores only the direct neighbors in a 2D array
Parameters - The lattice and the number of sites (N)
Returns - A list of neighbors with only the direct neighbors stored
"""
def store_direct_neighbors(lattice, N):
    # Store the neighbors in a list of lists
    real_neighbors_data = [[] for i in range(N)]

    # Find every sites' (i) direct neighbors
    for i in range(N):
        neighbors = get_direct_neighbors(i, lattice, N)
        real_neighbors_data[i] = [(n, 'direct') for n in neighbors]
    return real_neighbors_data

"""
Purpose - Prints all sets of neighbors (without the counts)
Parameters - The list of neighbors and the number of sites (N)
Returns - Nothing; prints all neighbors
"""
def print_neighbors(real_neighbors_data, N):
    for i in range(0, N):
        print(i, "'s neighbors are: ")
        print(real_neighbors_data[i])
        print("\n")

"""
Purpose - Iterate through the basis array and populate the dictionary of basis vectors
Parameters - The number of basis states and the basis states themselves
Returns - A dictionary for fast basis state look-ups (this speeds up some of the spin-flip operations)
"""
def populate_dictionary(num_basis, basis):
    # Make a dictionary for fast look-ups of our basis states
    basis_dict = {}

    for i in range(num_basis):
        config = basis[i]
        # Convert numpy array to a string
        config_str = ''.join(map(str, config))
        basis_dict[config_str] = i
    return basis_dict

"""
Purpose - Checks if a two lattice sites are periodically connected across a boundary
Parameters - The lattice, neighbor index (j), the current site (k), supercell coefficients
Returns - A boolean that tells you if two sites are connected across a boundary
NOTE: It's easier to simply check the neighbor_type now
"""
def wraps_boundary(lattice, j, k, sup_coefficient, sup_coefficient2):
    x1, y1 = lattice.sites[j]
    x2, y2 = lattice.sites[k]

    # Check if two points are as far as can be from each other in a direction (periodic)
    if abs(x1 - x2) == sup_coefficient - 1 or abs(y1 - y2) == sup_coefficient2 - 1:
        return True
    else:
        return False

"""
Purpose - Checks if two periodic sites are in the same column
Parameters - The lattice, neighbor index (j), the current site (k), column supercell coefficient
Returns - A boolean that tells you if two periodic sites are in the same column
"""
def check_columns_periodic(lattice, j, k, sup_coefficient2):
    x1, y1 = lattice.sites[j]
    x2, y2 = lattice.sites[k]

    # Check if two points on the same column are as far as can be from each other (periodic)
    if abs(y1 - y2) == sup_coefficient2 - 1:
        return True
    else:
        return False

"""
Purpose - Checks if two periodic sites are in the same row
Parameters - The lattice, neighbor index (j), the current site (k), row supercell coefficient
Returns - A boolean that tells you if two periodic sites are in the same row
"""
def check_rows_periodic(lattice, j, k, sup_coefficient):
    x1, y1 = lattice.sites[j]
    x2, y2 = lattice.sites[k]

    # Check if two points on the same row are as far as can be from each other (periodic)
    if abs(x1 - x2) == sup_coefficient - 1:
        return True
    else:
        return False

"""
Purpose - Checks if any two sites are in the same row
Parameters - The lattice, neighbor index (j), the current site (k)
Returns - A boolean that tells you if two sites are in the same row
"""
def check_rows(lattice, j, k):
    x1, y1 = lattice.sites[j]
    x2, y2 = lattice.sites[k]

    # Check if two points are in the same row
    if y1 == y2:
        return True
    else:
        return False

"""
Purpose - Checks if any two sites are in the same column
Parameters - The lattice, neighbor index (j), the current site (k)
Returns - A boolean that tells you if two sites are in the same column
"""
def check_columns(lattice, j, k):
    x1, y1 = lattice.sites[j]
    x2, y2 = lattice.sites[k]

    # Check if two points are in the same column
    if x1 == x2:
        return True
    else:
        return False

"""
Purpose - Makes the random outer neighbors that are needed for random boundary conditions
Parameters - The number of sites (N)
Returns - A random configuration of outer spins
"""
def make_outer_neighbors(N):
    num_random = 2 * N
    half_num_random = num_random // 2
    
    # Create a list with half +1/2 and half -1/2
    config = np.concatenate([
        np.full(half_num_random, 1/2),  # Half of the elements as +1/2
        np.full(half_num_random, -1/2)  # Half of the elements as -1/2
    ])
    
    # Randomly shuffle the configuration
    np.random.shuffle(config)
    return config

"""
Purpose - Makes the random outer neighbors that are needed for random boundary conditions in either the columns or rows
Parameters - The supercell coefficients of the random row or column
Returns - A random configuration for just the row or column
"""
def make_random(sup_coefficient_rand):
    num_random = 2 * sup_coefficient_rand
    half_num_random = num_random // 2
    
    # Create a list with half +1/2 and half -1/2
    config = np.concatenate([
        np.full(half_num_random, 1/2),  # Half of the elements as +1/2
        np.full(half_num_random, -1/2)  # Half of the elements as -1/2
    ])
    
    # Randomly shuffle the configuration
    np.random.shuffle(config)
    return config

"""
Purpose - Replace neighbors in columns with random spins with Sztot = 0
Parameters - The random column configuration, list of neighbors, and the lattice
Returns - A new list of neighbors with updated random entries
"""
def replace_random_columns(rand_columns, neighbors_data, lattice):
    rand_counter = 0

    # Loop through each site (k), then each sites neighbors (j)
    for k in range(len(neighbors_data)):
        neighbors = neighbors_data[k]
        new_neighbors = []
        for j, neighbor_type in neighbors:

            # If they are a periodic column connection, replace them with a 1/2 or -1/2
            if j != 1/2 and j != -1/2:
                if neighbor_type == 'periodic' and check_columns(lattice=lattice, j=j, k=k):
                    new_neighbors.append((rand_columns[rand_counter], 'periodic'))
                    rand_counter += 1
                else:
                    new_neighbors.append((j, neighbor_type))
            else:
                new_neighbors.append((j, neighbor_type))
        neighbors_data[k] = new_neighbors  # Update the neighbors_data list
    return neighbors_data

"""
Purpose - Replace neighbors in rows with random spins with Sztot = 0
Parameters - The random row configuration, list of neighbors, and the lattice
Returns - A new list of neighbors with updated random entries
"""
def replace_random_rows(rand_rows, neighbors_data, lattice):
    rand_counter = 0

    # Loop through each site (k), then each sites neighbors (j)
    for k in range(len(neighbors_data)):
        neighbors = neighbors_data[k]
        new_neighbors = []
        for j, neighbor_type in neighbors:

            # If they are a periodic column connection, replace them with a 1/2 or -1/2
            if j != 1/2 and j != -1/2:
                if neighbor_type == 'periodic' and check_rows(lattice=lattice, j=j, k=k):
                    new_neighbors.append((rand_rows[rand_counter], 'periodic'))
                    rand_counter += 1
                else:
                    new_neighbors.append((j, neighbor_type))
            else:
                new_neighbors.append((j, neighbor_type))
        neighbors_data[k] = new_neighbors  # Update the neighbors_data list
    return neighbors_data

"""
Purpose - Adds all of the random outer neighbors to real_neighbors_data
Parameters - The lattice, the list of neighbors, the list of random outer neighbors, the number of sites (N), and the supercell coefficients
Returns - A list of neighbors with completely random boundary conditions (both columns and rows)
"""
def add_random(lattice, real_neighbors_data, outer_neighbors_data, N, sup_coefficient, sup_coefficient2):
    num_random_added = 0
    # Loop through all sites (i)
    for i in range(N):
        # Loop through all sites and only consider those that aren't equal to i
        for site_index in range(N):
            if i != site_index:
                if wraps_boundary(lattice=lattice, k=site_index, j=i, sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2):
                    if len(real_neighbors_data[i]) < 4:
                        real_neighbors_data[i].append((outer_neighbors_data[num_random_added], 'periodic'))
                        num_random_added += 1
    return real_neighbors_data

"""
Purpose - Prints all of the edges in your lattice given the neighbors' data
Parameters - The list of neighbors
Returns - Nothing; prints the interactions between all neighbor pairs
"""
def print_edges(neighbors_data):
    seen = set()

    # Loop through all sites (i), neighbors (j), and then print any edges we see
    for i, neighbors in enumerate(neighbors_data):
        for j, neighbor_type in neighbors:
            edge = (min(i, j), max(i, j)) 
            if edge not in seen:
                seen.add(edge)
                print(f"Interaction between sites {i} and {j} (type: {neighbor_type})")
    print(f"Total number of edges: {len(seen)}\n")