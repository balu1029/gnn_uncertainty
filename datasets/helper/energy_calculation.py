from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import numpy as np

kjpmol_to_kcalpmol = 0.239006

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

        self.context = self.simulation.context

    def set_positions(self, positions:np.array)->None:
        """
        Set the positions of the atoms in the simulation.

        Args:
            positions (np.array): Array of atom positions.
        """
        positions =  positions * 0.1
        positions = [Vec3(position[0], position[1], position[2]) for position in positions]
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
                    coordinates.append([float(x), float(y), float(z)])
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
        energies = []
        atoms_ala = ['H', 'C', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'C', 'H', 'H', 'H', 'C', 'O', 'N', 'H', 'C', 'H', 'H', 'H']
        if num_molecules is None:
            num_molecules = len(molecules)
        with open(out_file, 'w') as file:
            for i in range(num_molecules):
                self.set_positions(molecules[i])
                state = self.context.getState(getEnergy=True)
                energy = state.getPotentialEnergy()
                energies.append(energy)
                file.write(f"{len(molecules[i])}\n")
                file.write(f"{energy.value_in_unit(kilojoules_per_mole)}\n")
                for j in range(len(molecules[i])):
                    file.write(f"{atoms_ala[j]} {molecules[i][j][0]} {molecules[i][j][1]} {molecules[i][j][2]}\n")


if __name__ == "__main__":
    energy_calculation = OpenMMEnergyCalculation()
    xyz_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj.xyz"
    out_path = "datasets/files/alaninedipeptide_traj/alaninedipeptide_traj_energies.xyz"
    '''with open(xyz_path, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0])
        coordinates = []
        for line in lines[2:num_atoms+2]:
            atom, x, y, z = line.split()
            coordinates.append([float(x), float(y), float(z)])
    energy_calculation.set_positions(np.array(coordinates))
    energy_calculation.step(1)
    state = energy_calculation.context.getState(getEnergy=True)
    print(state.getPotentialEnergy() * kjpmol_to_kcalpmol)'''
    npy_in = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19.npy"
    xyz_out = "datasets/files/ala_converged/prod_positions_20-09-2023_13-10-19_energies_1,000,000.xyz"
    energy_calculation.run_npy_file(npy_in, xyz_out, 1000000)
