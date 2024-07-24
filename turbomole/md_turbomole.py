from ase import Atoms
from ase.calculators.turbomole import Turbomole
from ase.md import VelocityVerlet
from ase.io import read, write
from ase import units
import os
import time


def write_xyz(positions, elements,energy, output_file):
    """
    Write molecular positions and their corresponding energies into an XYZ file.
    
    Parameters:
    - molecular_positions: list of tuples, where each tuple contains (element, x, y, z)
    - energies: list of floats, where each float is the energy corresponding to the molecule
    - output_file: string, the name of the output XYZ file
    """
    with open(output_file, 'a') as f:
        num_atoms = len(positions)
        f.write(f"{num_atoms}\n")
        f.write(f"{energy:.6f}\n")
        for element, pos in zip(elements,positions):
            x,y,z = pos
            f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


xyz_path = 'turbomole/coords/alaninedipeptide.xyz'

# Read the starting point from an XYZ file
atoms = read(xyz_path)


os.chdir('turbomole/tmp2/')
# Set up the Turbomole calculator
params = {'multiplicity': 1}
calc = Turbomole(**params)
atoms.calc = calc
calc.initialize()

# Set up the MD simulation
dyn = VelocityVerlet(atoms, timestep=1 * units.fs)

# Perform the MD simulation
timings = []
energies = []
coords = []
numbers_to_atoms = {6: 'C', 1: 'H', 7: 'N', 8: 'O'}

for step in range(3000):  
    start = time.time()
    positions = atoms.get_positions()
    total_energy = atoms.get_potential_energy()
    molecules = [numbers_to_atoms[atom] for atom in atoms.get_atomic_numbers()]

    write_xyz(positions, molecules, total_energy, f'../{xyz_path.split("/")[-1].split(".")[0]}_traj2.xyz')

    dyn.run(1)


