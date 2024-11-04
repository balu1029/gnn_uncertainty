from openmm.app import *
from openmm import *
from openmm.unit import picosecond, femtoseconds, kelvin , kilojoules_per_mole, nanometers, angstrom
from sys import stdout
import os
import numpy as np
import multiprocessing
import torch

sys.path.append(os.path.abspath('/home/kit/iti/fq0795/gnn_uncertainty/'))
from datasets.md17_dataset import MD17Dataset

kjpmol_to_kcalpmol = 0.239006
ev_to_kjpmol = 96.485

class OpenMMEnergyCalculation:
    """
    Class for performing energy calculations using OpenMM.

    Args:
        file (str): Path to the PDB file. Default is 'datasets/files/alaninedipeptide/pdb/alaninedipeptide.pdb'.
    """

    def __init__(self, file:str='datasets/files/alaninedipeptide/pdb/alaninedipeptide.pdb') -> None:
        pdb = PDBFile('datasets/files/alaninedipeptide/pdb/alaninedipeptide.pdb')
        forcefield = ForceField('amber14-all.xml')
        system = forcefield.createSystem(pdb.topology)
        self.temperature = 300
        integrator = LangevinIntegrator(self.temperature*kelvin, 1/picosecond, 1*femtoseconds)
        self.simulation = Simulation(pdb.topology, system, integrator)
        self.simulation.context.setPositions(pdb.positions)

        atoms = [atom.name for atom in pdb.topology.atoms()]
        print(atoms)

        self.context = self.simulation.context

    def add_ml_energy(self, model, model_path:str)->None:
        """
        Add a machine learning model to the energy calculation.

        Args:
            model: Machine learning model to use.
            model_path (str): Path to the model.
        """
        self.simulation.context.setProperty('model', model)
        self.simulation.context.setProperty('model_path', model_path)

    def set_positions(self, positions:np.array)->None:
        """
        Set the positions of the atoms in the simulation.

        Args:
            positions (np.array): Array of atom positions in nm.
        """
        positions =  positions 
        positions = [Vec3(position[0], position[1], position[2]) * nanometers for position in positions]
        self.simulation.context.setPositions(positions)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
        #self.simulation.context.setVelocities([(0.0, 0.0, 0.0)] * len(positions))

    def step(self, steps:int)->None:
        """
        Perform a specified number of simulation steps.

        Args:
            steps (int): Number of steps to perform.
        """
        self.simulation.step(steps)

    def run_xyz_file(self, in_file:str, out_file:str, num_molecules:int=None)->np.array:
        """
        Run energy calculations for molecules specified in an XYZ file.

        Args:
            in_file (str): Path to the XYZ file.
            out_file (str): Path to the output file to store xyz and energies in.
            num_molecules (int): Number of molecules to calculate energy for. If 0 is provided, it iterates over all molecules in the file.
        """
        with open(in_file, 'r') as file:
            sum = 0 
            molecules = []
            atoms = []
            lines = file.readlines()
            if num_molecules is None:
                num_molecules = len(lines)

            for i in range(num_molecules):
                if sum >= len(lines):
                    num_molecules = i
                    break
                num_atoms = int(lines[sum])
                coordinates = []
                for line in lines[sum+2:sum+num_atoms+2]:
                    atom, x, y, z = line.split()
                    atoms.append(atom)
                    coordinates.append([float(x), float(y), float(z)]) * 0.1  # convert to nm
                sum += num_atoms + 2
                molecules.append(coordinates)
        molecules = np.array(molecules)
        atoms = np.array(atoms)

        energies = []
        with open(out_file, 'w') as file:
            for i in range(num_molecules):
                self.set_positions(molecules[i])
                state = self.context.getState(getEnergy=True)
                energy = state.getPotentialEnergy()
                energies.append(energy)
                file.write(f"{len(molecules[i])}\n")
                file.write(f"{energy.value_in_unit(kilojoules_per_mole)}\n")
                #file.write(f"{energy}\n")
                for j in range(len(molecules[i])):
                    file.write(f"{atoms[j]} {molecules[i][j][0]} {molecules[i][j][1]} {molecules[i][j][2]}\n")

        print(f"finished calculation for {num_molecules} molecules")

        return np.array(energies)

    def run_npy_file(self, in_file:str, out_file:str, num_molecules:int)->None:
        """
        Run energy calculations for molecules specified in a NPY file.

        Args:
            in_file (str): Path to the NPY file.
            out_file (str): Path to the output file to store xyz and energies in.
            num_molecules (int): Number of molecules to calculate energy for. If 0 is provided, it iterates
        """
        molecules = np.load(in_file)
        
        if num_molecules is None:
            num_molecules = len(molecules)

        self._generate_from_npy(molecules, out_file, num_molecules)
    

    def run_npy_file_parallel(self, in_file:str, out_file:str, num_molecules:int, num_threads:int):
        """
        Run energy calculations for molecules specified in a NPY file in parallel.

        Args:
            in_file (str): Path to the NPY file.
            out_file (str): Path to the output file to store xyz and energies in.
            num_molecules (int): Number of molecules to calculate energy for. If 0 is provided, it iterates
            num_threads (int): Number of threads to use for parallel processing.
        """
        
        molecules = np.load(in_file)
        if num_molecules is None:
            num_molecules = len(molecules)
        else:
            molecules = molecules[:num_molecules]
        

        self._generate_from_npy(molecules, out_file, num_molecules)
        return NotImplementedError
    
    def generate_validation_from_numpy(self, in_file:str, out_file:str, num_molecules:int)->None:
        """
        Generate a validation set from a NPY file.

        Args:
            in_file (str): Path to the NPY file.
            out_file (str): Path to the output file to store xyz and energies in.
            num_molecules (int): Number of molecules to calculate energy for. If 0 is provided, it iterates
        """
        molecules = np.load(in_file)
        
        if num_molecules is None:
            num_molecules = len(molecules)

        molecules = molecules[-num_molecules:]

        self._generate_from_npy(molecules, out_file, num_molecules)
    
    def generate_in_distribution_from_numpy(self, in_file:str, out_file:str, num_molecules:int, valid:bool)->None:
        molecules = np.load(in_file)
        indices1 = [4, 6, 8, 14]
        indices2 = [6, 8, 14, 16]   

        psi = self.calculate_dihedrals_batch(molecules, indices1)
        phi = self.calculate_dihedrals_batch(molecules, indices2)

        phi_indices = np.where(psi < -100)[0]
        psi_indices = np.where((phi > 50) | (phi < -150))[0]
        
        overlapping_indices = np.intersect1d(phi_indices, psi_indices)

        in_dist = molecules[overlapping_indices][-int(num_molecules/2):]
        out_dist = molecules[np.delete(np.arange(len(molecules)), overlapping_indices)][-int(num_molecules/2):]
        del(molecules)

        out_file_name = out_file.split(".xyz")[0]
        if not valid:
            self._generate_from_npy(in_dist, out_file_name + "_in_dist.xyz", num_molecules)
        else:
            self._generate_from_npy(in_dist, out_file_name + "_validation_in_dist.xyz", int(num_molecules/2))
            self._generate_from_npy(out_dist, out_file_name + "_validation_out_dist.xyz", int(num_molecules/2))

    def generate_validation_uniform(self, in_file:str, out_file:str, num_grids_per_dim:int, samples_per_grid:int)->None:
        molecules = np.load(in_file)

        indices1 = [4, 6, 8, 14]
        indices2 = [6, 8, 14, 16]

        samples = None


        psi = self.calculate_dihedrals_batch(molecules, indices1)
        phi = self.calculate_dihedrals_batch(molecules, indices2)

        grid_points = np.linspace(-180, 180, num_grids_per_dim + 1) 

        for i in range(num_grids_per_dim):
            ph = grid_points[i]
            for j in range(num_grids_per_dim):
                ps = grid_points[j]
                
                phi_indices = np.where((psi > ps) & (psi < grid_points[j+1]))[0]
                psi_indices = np.where((phi > ph) & (phi < grid_points[i+1]))[0]
                overlapping_indices = np.intersect1d(phi_indices, psi_indices)
                num_samples = min(len(overlapping_indices), samples_per_grid)
                if num_samples > 0:
                    sampled_indices = np.random.choice(overlapping_indices, size=num_samples, replace=False)
                    if samples is None:
                        samples = molecules[sampled_indices]
                    else:
                        samples = np.concatenate((samples, molecules[sampled_indices]), axis=0)
        self._generate_from_npy(samples, out_file, len(samples))

    def generate_from_dataset(self, dataset, out_file:str)->None:
        molecules = dataset.coordinates
        print(len(molecules))
        print(out_file)
        self._generate_from_npy(molecules, out_file, len(molecules))


    
    def calc_energy(self, positions:np.array)->np.array:
        """
        Calculate the energy of a molecule with the given positions.

        Args:
            positions (np.array): Array of atom positions in nm.

        Returns:
            float: Energy of the molecule in kJ/mole.
        """
        energies = []
        for i in range(len(positions)):
            self.set_positions(positions[i])
            state = self.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole) #/ ev_to_kjpmol
            energies.append(energy)
        return np.array(energies)

    def calc_forces(self, positions:np.array)->np.array:
        """
        Calculate the forces of a molecule with the given positions.

        Args:
            positions (np.array): Array of atom positions in nm.

        Returns:
            np.array: Array of forces in kJ/(mol*angstrom).
        """
        forces = []
        for i in range(len(positions)):
            self.set_positions(positions[i])
            state = self.context.getState(getForces=True)
            force = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole/angstrom)
            forces.append(force)
        return np.array(forces)

    def _generate_from_npy(self, molecules:np.array, out_file:str, num_molecules:int)->None:
        energies = []
        atoms_ala = ['H', 'C', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'H', 'H']
        with open(out_file, 'w') as file:
            for i in range(num_molecules):
                self.set_positions(molecules[i]*0.1)    # convert to nm
                state = self.context.getState(getEnergy=True, getForces=True)
                energy = state.getPotentialEnergy()
                forces = state.getForces()
                energies.append(energy)
                file.write(f"{len(molecules[i])}\n")
                file.write(f"{energy.value_in_unit(kilojoules_per_mole)}\n")
                for j in range(len(molecules[i])):
                    file.write(f"{atoms_ala[j]} {molecules[i][j][0]} {molecules[i][j][1]} {molecules[i][j][2]} {forces[j][0].value_in_unit(kilojoules_per_mole/angstrom)} {forces[j][1].value_in_unit(kilojoules_per_mole/angstrom)} {forces[j][2].value_in_unit(kilojoules_per_mole/angstrom)}\n")

    def calculate_dihedrals_batch(self, molecules, indices) -> np.array:
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
        angles = -np.degrees(np.arctan2(y, x))
        
        return angles


if __name__ == "__main__":
    energy_calculation = OpenMMEnergyCalculation()
    xyz_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj.xyz"
    out_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj_energies.xyz"
   
    #npy_in = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19.npy"
    #xyz_out = "datasets/files/ala_converged/dataset_10000.xyz"
    #energy_calculation.run_npy_file(npy_in, xyz_out,10000)
    #energy_calculation.generate_in_distribution_from_numpy(npy_in, xyz_out, 10000, valid=True)

    npy_in = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19.npy"
    xyz_out = "datasets/files/active_learning_validation/dataset2.xyz"

    dataset = "datasets/files/train_in"
    out_path = "datasets/files/train_in2/dataset.xyz"
    trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)
    
    energy_calculation.generate_from_dataset(trainset, out_path)
    #energy_calculation.generate_validation_uniform(npy_in, xyz_out, 300, 1)
