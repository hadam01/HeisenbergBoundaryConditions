## 
## AllBC.py
## Husam Adam
## Purpose - Make a Hamiltonian that simulates periodic, antiperiodic,
## open, random, and twisted boundary conditions. This code can take
## any pair of boundary conditions (for the rows and columns respectively)
## and compute a Hamiltonians eigenvalues/eigenvectors

# Import necessary and supplementary functions from other files
from Lattice_Utils import make_lattice, plot_lattice, write_basis_states, generate_basis_states, print_neighbor_counts, store_all_neighbors_with_type, store_direct_neighbors, print_neighbors, populate_dictionary, wraps_boundary, check_rows, check_columns, make_outer_neighbors, make_random, replace_random_columns, add_random, print_edges, check_columns_periodic, check_rows_periodic, replace_random_rows
from Hamiltonian_Utils import print_hamiltonian, find_eigen, print_eigenvalues, print_eigenvectors, store_eigenvalues, store_eigenvectors, check_hamiltonian, print_ground_state
import numpy as np

"""
Purpose - Check that we were given a theta value for twisted boundary conditions
(theta for the rows - theta2 for the columns)
Parameters - Row boundary condition, column boundary condition, theta value, and theta2 value
Returns - Nothing; Raises an error if the parameters aren't whats expected
"""
def twisted_theta_check(rowBC, columnBC, theta, theta2):
    # Raise errors if the parameters aren't what's expected
    if(rowBC=='twisted' and theta==None):
        raise ValueError("No theta value received for rowBC. Double check arguments.")
    elif(columnBC=='twisted' and theta2==None):
        raise ValueError("No theta2 value received for columnBC. Double check arguments.")

"""
Purpose - Add random boundary conditions to our list of neighbors (if necessary)
Parameters - Row boundary condition, column boundary condition, supercell coefficient, 
second supercell coefficient, and list of neighbors
Returns - An updated list of neighbors including random neighbors (if necessary)
"""
def add_all_random(rowBC, columnBC, lattice, sup_coefficient, sup_coefficient2, neighbors_data):
    if(columnBC == 'random'):
        rand_columns = make_random(sup_coefficient_rand=sup_coefficient)
        neighbors_data = replace_random_columns(rand_columns=rand_columns, neighbors_data=neighbors_data, lattice=lattice)
        return neighbors_data
    if(rowBC == 'random'):
        rand_rows = make_random(sup_coefficient_rand=sup_coefficient2)
        neighbors_data = replace_random_rows(rand_rows=rand_rows, neighbors_data=neighbors_data, lattice=lattice)
        return neighbors_data
    if(rowBC != 'random' and columnBC != 'random'):
        return neighbors_data

"""
Purpose - Calculate the Hamiltonian entry for two sites that are directly or periodically connected
Parameters - The type of neighbor, current basis index (i), the site neighbor (j), the site
you're currently considering (k), site pairs you've seen, Hamiltonian, basis states,
and basis dictionary
Returns - Nothing; It simply adds the direct/periodic neighbor interaction to the Hamiltonian
"""
def direct_neighbor_entry(neighbor_type, i, j, k, seen, Hamiltonian, basis, basis_dict):
    # Update the Hamiltonian if we see a new interaction
    if (j, k, neighbor_type) not in seen and (k, j, neighbor_type) not in seen:
        # Update the diagonal of the Hamiltonian
        Hamiltonian[i][i] +=  basis[i][k] * basis[i][j]

        # Update the off-diagonal terms
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
            Hamiltonian[i][entry] +=  2

        # Ensure we don't consider this interaction again
        seen.append((k, j, neighbor_type))

"""
Purpose - Calculate the Hamiltonian entry for an antiperiodic boundary crossing
Parameters - The type of neighbor, current basis index (i), the site neighbor (j), the site
you're currently considering (k), site pairs you've seen, Hamiltonian, basis states,
and basis dictionary
Returns - Nothing; It simply adds the antiperiodic neighbor interaction to the Hamiltonian
"""
def antiperiodic_entry(neighbor_type, i, j, k, seen, Hamiltonian, basis, basis_dict):
    # Update the Hamiltonian if we see a new interaction
    if (j, k, neighbor_type) not in seen and (k, j, neighbor_type) not in seen:
        # Add the antiperiodic diagonal element
        Hamiltonian[i][i] += -1 * basis[i][k] * basis[i][j]

        # Add the antiperiodic off-diagonal element
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
            Hamiltonian[i][entry] +=  -1 * 2
        
        # Ensure we don't consider this interaction again
        seen.append((k, j, neighbor_type))

"""
Purpose - Calculate the Hamiltonian entry for a random boundary crossing
Parameters - The type of neighbor, current basis index (i), the site neighbor (j), the site
you're currently considering (k), site pairs you've seen, Hamiltonian, basis states,
and basis dictionary
Returns - Nothing; It simply adds the random neighbor interaction to the Hamiltonian
"""
def random_entry(neighbor_type, i, j, k, seen, Hamiltonian, basis):
    # Update the Hamiltonian if we see a new interaction
    if(k, j, neighbor_type) not in seen and (j, k, neighbor_type) not in seen:
        interaction_term = 0
        # Only compute diagonal terms for random spins (the spins are fixed)
        if j == 1/2:
            interaction_term = 1 * basis[i][k]
        elif j == -1/2:
            interaction_term = -1 * basis[i][k]
        
        Hamiltonian[i][i] += interaction_term

        # Ensure we don't consider this interaction again
        seen.append((k, j, neighbor_type))

"""
Purpose - Calculate the Hamiltonian entry for a twisted boundary crossing
Parameters - The type of neighbor, current basis index (i), the site neighbor (j), the site
you're currently considering (k), site pairs you've seen, Hamiltonian, basis states,
and basis dictionary
Returns - Nothing; It simply adds the twisted neighbor interaction to the Hamiltonian
"""
def twisted_entry(theta, neighbor_type, i, j, k, seen, Hamiltonian, basis, basis_dict):
    # Update the Hamiltonian if we see a new interaction
    if (j, k, neighbor_type) not in seen and (k, j, neighbor_type) not in seen:
        shift = np.exp(-1j * theta) if neighbor_type=='periodic' else 1
        shift2 = np.exp(1j * theta) if neighbor_type=='periodic' else 1
        Hamiltonian[i][i] += basis[i][k] * basis[i][j]
    
        new_basis = basis[i].copy()
        if new_basis[k] == -1 and new_basis[j] == 1:
            new_basis[j] = -1
            new_basis[k] = 1
            new_basis_str = ''.join(map(str, new_basis))
            entry = basis_dict[new_basis_str]
            Hamiltonian[i][entry] += shift * 2
        elif new_basis[k] == 1 and new_basis[j] == -1:
            new_basis[j] = 1
            new_basis[k] = -1
            new_basis_str = ''.join(map(str, new_basis))
            entry = basis_dict[new_basis_str]
            Hamiltonian[i][entry] += shift2 * 2
        seen.append((k, j, neighbor_type))

"""
Purpose - Make the actual Hamiltonian and call the necessary functions to update it given our boundary conditions
Parameters - The lattice, the number of basis states, list of neighbors, basis states, basis' dictonary
the number of total sites (N), row boundary condition, column boundary condtion, theta value, and theta2 value
Returns - The fully constructed Hamiltonian
"""
def make_hamiltonian(lattice, num_basis, neighbors_data, basis, basis_dict, N, rowBC, columnBC, theta, theta2):
    # Initialize our complex Hamiltonian with 0's
    Hamiltonian = np.zeros((num_basis, num_basis), dtype=complex)

    # Make a list where we can store the neighbors we have seen so far
    seen = []

    # Loop through the basis (i), then each site (k), and then each sites neighbors (j)
    for i in range(0, num_basis):
        for k in range(0, N):
            neighbors = neighbors_data[k]
            for j, neighbor_type in neighbors:
                # Update the Hamiltonian depending on the type of interaction you're considering
                if(neighbor_type == 'direct'):
                    direct_neighbor_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                elif(neighbor_type == 'periodic'):
                    if(j == 1/2 or j == -1/2):
                        random_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis)
                    elif(check_columns(lattice=lattice, j=j, k=k)):
                        if(columnBC == 'periodic'):
                            direct_neighbor_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        elif(columnBC == 'antiperiodic'):
                            antiperiodic_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        elif(columnBC == 'random'):
                            random_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        elif(columnBC == 'open'):
                            pass
                        elif(columnBC == 'twisted'):
                            twisted_entry(theta=theta2, neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        else:
                            # If we couldn't find a correct boundary condition, raise an error
                            raise ValueError(f"Invalid rowBC value: '{columnBC}'. Expected one of: 'periodic', 'antiperiodic', 'open', 'random', 'twisted'.")
                    elif(check_rows(lattice=lattice, j=j, k=k)):
                        if(rowBC == 'periodic'):
                            direct_neighbor_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        elif(rowBC == 'antiperiodic'):
                            antiperiodic_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        elif(rowBC == 'random'):
                            random_entry(neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis)
                        elif(rowBC == 'open'):
                            pass
                        elif(rowBC == 'twisted'):
                            twisted_entry(theta=theta, neighbor_type=neighbor_type, i=i, j=j, k=k, seen=seen, Hamiltonian=Hamiltonian, basis=basis, basis_dict=basis_dict)
                        else:
                            # If we couldn't find a correct boundary condition, raise an error
                            raise ValueError(f"Invalid rowBC value: '{rowBC}'. Expected one of: 'periodic', 'antiperiodic', 'open', 'random', 'twisted'.")
        # Clear the sites we've seen after running through all sites
        seen.clear()
    return Hamiltonian

"""
Purpose - Runs everything you need to make the Hamiltonian and find the eigenvalues/eigenvectors.
Parameters - Supercell coefficient, second supercell coefficient, row boundary condition, column
boundary condition, theta value, theta2 value
Returns - All eigenvalues and eigenvectors
"""
def run(sup_coefficient, sup_coefficient2, rowBC, columnBC, theta, theta2):
    # Check that we were given thetas for twisted boundary conditions
    twisted_theta_check(rowBC=rowBC, columnBC=columnBC, theta=theta, theta2=theta2)

    # Calculate the number of sites (N)
    N = sup_coefficient*sup_coefficient2

    # Make our lattice, basis states, and the number of basis states
    lattice = make_lattice(sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2)
    basis = generate_basis_states(n=N)
    num_basis = int(basis.size/N)

    # Store all neighbors and add random ones if necessary
    neighbors_data = store_all_neighbors_with_type(lattice=lattice, N=N, 
    sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2)
    neighbors_data = add_all_random(rowBC=rowBC, columnBC=columnBC, lattice=lattice, 
    sup_coefficient=sup_coefficient, sup_coefficient2=sup_coefficient2, neighbors_data=neighbors_data)

    # Populate the basis state dictionary
    basis_dict = populate_dictionary(num_basis=num_basis, basis=basis)

    # Make our Hamiltonian
    Hamiltonian = make_hamiltonian(lattice=lattice, 
    num_basis=num_basis, neighbors_data=neighbors_data, basis=basis,
    basis_dict=basis_dict, N=N, rowBC=rowBC, columnBC=columnBC, theta=theta, theta2=theta2)

    # Find the eigenvalues/eigenvectors and print the ground state
    eigenvalues, eigenvectors = find_eigen(Hamiltonian)
    print_ground_state(eigenvalues=eigenvalues)
    return eigenvalues, eigenvectors

# Example of a run command.
# eigenvalues, eigenvectors = run(sup_coefficient=4, sup_coefficient2=3, columnBC='random', rowBC='periodic', theta=2*np.pi, theta2=None)