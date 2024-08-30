## 
## NetKetPBC.py
## Husam Adam
## Purpose - Use NetKet to simulate periodic boundary conditions.
## NOTE: The 2xN lattice cases differ from the results from PeriodicBC.py.
## This is because I decided to view neighbors in the 2 direction as having
## two separate interactions - one periodic and one direct. This makes my
## comparisons to other boundary conditions more consistent.

# Import netket and helper libraries
import netket as nk
import numpy as np

"""
Purpose - A function that stores the energy in a given file
Parameters - The eigenvalues and filename
Return - Nothing; stores eigenvalues in the file
"""
def store_eigenvalues(eigenvalues, filename):
    # Open a file in write mode
    with open(filename, 'w') as file:
        # Iterate through the list of eigenvalues and write each one to the file
        for value in eigenvalues:
            file.write(f'{value}\n')

"""
Purpose - A function that stores all eigenvectors in a given file
Parameters - The eigenvectors and filename
Return - Nothing; store eigenvectors in the file
"""
def store_eigenvectors(eigenvectors, filename):
    # Open a file in write mode
    with open(filename, 'w') as file:
        # Store all of the eigenvectors
        for eigenvector in eigenvectors:
            for value in eigenvector:
                file.write(f"{value} ")
            file.write("\n")

"""
Purpose - A function that will write the basis states NetKet uses to a given file
Parameters - The filename and the basis states
Return - Nothing; store basis states in the file
"""
def write_basis_states(filename, basis_states):
    with open(filename, 'w') as f:
        for state in basis_states:
            f.write(' '.join(map(str, state)) + '\n')

"""
Purpose - A function that will print all of the interactions NetKet considers
Parameters - The graph from NetKet (g)
Return - Nothing; print all edges on NetKets graph
"""
def print_netket_edges(g):
    netket_edges = g.edges()
    for edge in netket_edges:
        print(f"Interaction between sites {edge[0]} and {edge[1]}")
    print(f"Total number of edges: {len(netket_edges)}\n")

"""
Purpose - Use periodic boundary conditions for a square lattice
Parameters - The supercell coefficient (L)
Returns - Eigenvalues and eigenvectors 
"""
def run_square(L):
        # Define a 2d chain
        g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

        # Define the Hilbert space based on this graph
        # We impose to have a fixed total magnetization of zero 
        hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)

        # calling the Heisenberg Hamiltonian
        ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

        # Compute eigenvalues and eigenvectors
        evals, evecs = nk.exact.lanczos_ed(ha, k=10, compute_eigenvectors=True)        
        print("Ground State Energy: ", evals[0])

        return evals, evecs

"""
Purpose - Use periodic boundary conditions for a rectangular lattice
Parameters - The supercell coefficients (L1, L2)
Returns - Eigenvalues and eigenvectors 
"""
def run_rectangle(L1, L2):
    # Define rectangular lattice
    basis_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # Basis vectors for a rectangular lattice
    extent = [L1, L2]  # Dimensions of the lattice
    pbc = True  # Periodic boundary conditions

    g = nk.graph.Lattice(basis_vectors=basis_vectors, extent=extent, pbc=pbc)

    # Define Hilbert space
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)

    # Define Heisenberg Hamiltonian
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Compute ground-state energy and eigenvectors
    evals, eigenvectors = nk.exact.lanczos_ed(ha, k=10, compute_eigenvectors=True)
    print("Ground State Energy: ", evals[0])

    return evals, eigenvectors

# Example of a run command.
# eigenvalues, eigenvectors = run_rectangle(L1=2, L2=2)