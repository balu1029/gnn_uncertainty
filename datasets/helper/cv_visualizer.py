import numpy as np
import time
import matplotlib.pyplot as plt



def calculate_dihedrals_batch(molecules, indices) -> np.array:
    """
    Calculate dihedral angles for a batch of molecules in a vectorized manner.
    
    Parameters:
    molecules (np.ndarray): Array of shape (n_molecules, n_atoms, 3) containing the coordinates of the atoms in each molecule.
    indices (list): List of 4 indices that define the dihedral angle for all molecules.
    
    Returns:
    np.ndarray: Array of dihedral angles for the batch of molecules.
    """
    n_molecules = molecules.shape[0]
    
    # Extract coordinates of the four atoms for all molecules
    p0 = molecules[:, indices[0], :]
    p1 = molecules[:, indices[1], :]
    p2 = molecules[:, indices[2], :]
    p3 = molecules[:, indices[3], :]
    
    # Compute the vectors between the points
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    
    # Normalize b1
    b1 /= np.linalg.norm(b1, axis=1)[:, np.newaxis]
    
    # Compute normal vectors to the planes
    n0 = np.cross(b0, b1)
    n1 = np.cross(b1, b2)
    
    # Normalize the normal vectors
    n0 /= np.linalg.norm(n0, axis=1)[:, np.newaxis]
    n1 /= np.linalg.norm(n1, axis=1)[:, np.newaxis]
    
    # Compute the dihedral angles
    m1 = np.cross(n0, b1)
    x = np.einsum('ij,ij->i', n0, n1)
    y = np.einsum('ij,ij->i', m1, n1)
    angles = np.arctan2(y, x)
    
    return angles




if __name__ == "__main__":
    indizes1 = [4, 6, 8, 14]
    indizes2 = [6, 8, 14, 16]
    n = 100000
    traj = np.load("datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19.npy")

    start_time = time.time()
    angles1 = calculate_dihedrals_batch(traj[0:n], indizes1)
    angles2 = calculate_dihedrals_batch(traj[0:n], indizes2)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time for calculate_dihedrals_batch: {elapsed_time} seconds")

    print(np.max(angles1), np.min(angles1))
    print(np.max(angles2), np.min(angles2)) 

    plt.hist2d(angles1, angles2, bins=50, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Angle 1')
    plt.ylabel('Angle 2')
    plt.title('2D Histogram of Angle 1 and Angle 2')
    plt.show()