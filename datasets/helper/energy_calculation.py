from openmm.app import *
from openmm import *
from openmm.unit import picosecond, femtoseconds, kelvin , kilojoules_per_mole, nanometers, angstrom
from sys import stdout
import numpy as np
import multiprocessing

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
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 1*femtoseconds)
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


if __name__ == "__main__":
    energy_calculation = OpenMMEnergyCalculation()
    xyz_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj.xyz"
    out_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj_energies.xyz"
   
    npy_in = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19.npy"
    xyz_out = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19_energies_forces_10000.xyz"
    energy_calculation.run_npy_file(npy_in, xyz_out,10000)
