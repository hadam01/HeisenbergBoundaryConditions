##
## Hamiltonian_Utils.py
## Husam Adam
## Purpose - This file provides utility functions for dealing with different
## Hamiltonian matrices - particularly dealing with eigenvalues, eigenvectors,
## and the Hamiltoninan matrix itself.

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']=300

"""
Purpose - Prints the Hamiltonian matrix
Paramaters - The Hamiltonian
Returns - Nothing; prints the Hamiltonian
"""
def print_hamiltonian(Hamiltonian):
    H = np.copy(Hamiltonian)
    # Set imaginary parts close to zero to zero
    H[np.isclose(H.imag, 0)] = H.real[np.isclose(H.imag, 0)]

    # Print the Hamiltonian
    np.set_printoptions(precision=3, suppress=True)
    print("Hamiltonian:")
    print(np.array(H))

"""
Purpose - Finds the eigenvalues and eigenvectors
Paramaters - The Hamiltonian
Returns - The eigenvalues and eigenvectors of the given Hamiltonian
"""
def find_eigen(Hamiltonian):
    # Use numpy to find the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)
    tolerance = 1e-10
    eigenvalues[np.abs(eigenvalues) < tolerance] = 0
    return eigenvalues, eigenvectors

"""
Purpose - Prints the lowest 10 eigenvalues
Paramaters - The eigenvalues
Returns - Nothing; prints the eigenvalues
"""
def print_eigenvalues(eigenvalues):
    np.set_printoptions(threshold=np.inf, precision=10, suppress=True)
    print("Lowest Ten Eigenvalues:")
    for i in range (0, 10):
        print(eigenvalues[i])

"""
Purpose - Prints all of the eigenvectors
Paramaters - The eigenvectors
Returns - Nothing; prints the eigenvectors
"""
def print_eigenvectors(eigenvectors):
    np.set_printoptions(threshold=np.inf, precision=10, suppress=True)
    print("Eigenvectors:")
    for eigenvector in eigenvectors:
        print(eigenvector)

"""
Purpose - Store the first 10 eigenvalues in a given file
Paramaters - The eigenvalues and the name of your file
Returns - Nothing; adds the eigenvalues to your file
"""
def store_eigenvalues(eigenvalues, filename):
    # Open a file in write mode
    with open(filename, 'w') as file:
    # Store the lowest 10 eigenvalues
        for i in range(0, 10):
            file.write(f'{eigenvalues[i]}\n')

"""
Purpose - Store all eigenvectors in a given file
Paramaters - The eigenvectors and the name of your file
Returns - Nothing; adds all of the eigenvectors to your file
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
Purpose - Checks if the Hamiltonian is Hermitian and/or symmetric
Paramaters - The Hamiltonian
Returns - Nothing; prints whether or not the Hamiltonian is Hermitian and/or symmetric
NOTE: A Hamiltonian should always be Hermitian (only symmetric if real)
"""
def check_hamiltonian(Hamiltonian):
    H = np.copy(Hamiltonian)
    # Check if the Hamiltonian is symmetric
    is_symmetric = np.allclose(H, H.T)
    if is_symmetric:
        print("The matrix H is symmetric")
    else:
        print("The matrix H is not symmetric")

    # Check if the Hamiltonian is Hermitian
    is_hermitian = True
    difference = H - H.conj().T
    # Check if all off-diagonal elements of difference are essentially zero
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if i != j and not np.isclose(difference[i][j], 0, atol=1e-10):
                is_hermitian = False
                break
        if not is_hermitian:
            break
    
    if is_hermitian:
        print("Hamiltonian is Hermitian")
    else:
        print("Hamiltonian is not Hermitian")

"""
Purpose - Checks that all eigenvectors are orthogonal to one another
Paramaters - The eigenvectors
Returns - Nothing; prints whether or not your eigenvectors are orthogonal
"""
def check_orthogonality(eigenvectors):
    print("Periodic orthogonality check:")
    N = len(eigenvectors)
    orthogonal = True
    
    # Check if all eigenvectors have a dot product that is essentially zero
    for i in range(N):
        for j in range(i + 1, N):
            dot_product = np.dot(eigenvectors[i], eigenvectors[j])
            if not np.isclose(dot_product, 0, atol=1e-10):
                print(f"Eigenvectors {i} and {j} are not orthogonal. Dot product: {dot_product}")
                orthogonal = False
    
    # Print the results
    if orthogonal:
        print("Periodic is completely orthogonal")
    else:
        print("Periodic is not completely orthogonal")

"""
Purpose - Checks that all eigenvectors are properly normalized
Paramaters - The eigenvectors
Returns - Nothing; prints whether or not your eigenvectors are normalized
"""
def check_normalization(eigenvectors):
    # Check if the eigenvectors are normalized
    norms = np.linalg.norm(eigenvectors, axis=1)
    normalized = np.allclose(norms, 1, atol=1e-10)

    # Print the result
    if normalized:
        print("Periodic is normalized")
    else:
        print("Periodic is not normalized")

"""
Purpose - Prints the ground state of a Hamiltonian
Paramaters - The eigenvalues
Returns - Nothing; prints the ground state energy of your Hamiltonian
"""
def print_ground_state(eigenvalues):
    print(f"Ground State Energy: ", eigenvalues[0])