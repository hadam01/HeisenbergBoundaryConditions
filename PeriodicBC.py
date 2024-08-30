##
## PeriodicBC.py
## Husam Adam
## Purpose - Construct the Hamiltonian of a square or rectangular lattice
## with a given number of sites.
## NOTE: If you want to match NetKet's results for a 2xN case, you have to get rid of the 
## neighbor_type conditions on lines 23 and 36. This is because NetKet doesn't consider a 
## periodic wrapping in the "2" direction a new interaction.

# Import necessary and supplementary functions from other files
from Lattice_Utils import make_lattice, plot_lattice, write_basis_states, generate_basis_states, print_neighbor_counts, store_all_neighbors_with_type, store_direct_neighbors, print_neighbors, populate_dictionary, wraps_boundary, check_rows, check_columns, make_outer_neighbors, make_random, replace_random_columns, add_random, print_edges, check_columns_periodic, check_rows_periodic, replace_random_rows
from Hamiltonian_Utils import print_hamiltonian, find_eigen, print_eigenvalues, print_eigenvectors, store_eigenvalues, store_eigenvectors, check_hamiltonian, print_ground_state
import numpy as np


"""
Purpose - Make the Hamiltonian simulating periodic boundary conditions
Paramaters - The number of basis states, list of neighbors, basis states, basis' dictonary, and
the number of total sites (N)
Returns - The fully constructed Hamiltonian
"""
def make_hamiltonian(num_basis, neighbors_data, basis, basis_dict, N):
    # Initialize our Hamiltonian with 0's
    Hamiltonian = np.zeros((num_basis, num_basis))
    seen = []
    # Loop through the basis states (i), lattice sites (k), and then neighbors (j)
    for i in range(0, num_basis):
        for k in range(0, N):
            neighbors = neighbors_data[k]
            for j, neighbor_type in neighbors:
                # Update the Hamiltonian if we see a new interaction
                if (k, j, neighbor_type) not in seen and (j, k, neighbor_type) not in seen: 
                    # Update the diagonal term of the Hamiltonian
                    Hamiltonian[i][i] += basis[i][k] * basis[i][j]

                    # Update the off-diagonal term of the Hamiltonian
                    new_basis = basis[i].copy()
                    if new_basis[k] == -1 and new_basis[j] == 1:
                        new_basis[j] = -1
                        new_basis[k] = 1
                    elif new_basis[k] == 1 and new_basis[j] == -1:
                        new_basis[j] = 1
                        new_basis[k] = -1
                    new_basis_str = ''.join(map(str, new_basis))
                    if not(np.array_equal(new_basis, basis[i])):
                        entry = basis_dict[new_basis_str]
                        Hamiltonian[i][entry] += 2
                    
                    # Ensure we don't consider this interaction again
                    seen.append((k, j, neighbor_type))
        # Reset the site pairs we've seen once we loop through all sites
        seen.clear()
    return Hamiltonian

"""
Purpose - Runs everything you need to make the Hamiltonian and find the eigenvalues/eigenvectors.
Parameters - Supercell coefficient, second supercell coefficient
Returns - All eigenvalues and eigenvectors
"""
def run(sup_coefficient, sup_coefficient2):
    # N is the number of sites
    N = sup_coefficient*sup_coefficient2

    # Construct the lattice and generate the basis states
    lattice = make_lattice(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2)
    basis = generate_basis_states(n=N)
    num_basis = int(basis.size/N)

    # Define our list of neighbors and populate our basis dictionary 
    neighbors_data = store_all_neighbors_with_type(lattice=lattice, N=N, 
    sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2)
    basis_dict = populate_dictionary(num_basis=num_basis, basis=basis)

    # Construct the Hamiltonian and find the eigenvalues/eigenvectors
    Hamiltonian = make_hamiltonian(num_basis=num_basis, neighbors_data=neighbors_data, 
    basis=basis, basis_dict=basis_dict, N=N)
    eigenvalues, eigenvectors = find_eigen(Hamiltonian=Hamiltonian)
    print_ground_state(eigenvalues=eigenvalues)

    return eigenvalues, eigenvectors

# Example of a run command.
# eigenvalues, eigenvectors = run(sup_coefficient=4, sup_coefficient2=4)