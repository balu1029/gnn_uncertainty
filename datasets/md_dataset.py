import numpy as np
import torch
import os


class MDDataset(torch.utils.data.Dataset):
    def __init__(self, foldername):
        self.type_to_number = {"H" : 0,
                               "C" : 1,
                               "N" : 2,
                               "O" : 3}
        self.ground_charges  = {0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        self_energies = {0: -0.500607632585,
                         1: -37.8302333826,
                         2: -54.5680045287,
                         3: -75.0362229210}
        Eh_to_eV = 27.211399
        
        self.atom_numbers, self.coordinates = self._read_coordinates(foldername + "/xyz")
        self.energies = self._read_energies(foldername + "/energy")
        self.total_self_energy = [sum(self_energies[atom] for atom in molecule) * Eh_to_eV for molecule in self.atom_numbers] 
        self.energies = self.energies - self.total_self_energy

        self.num_atoms = self.atom_numbers.shape[1]
        self.num_snapshots = self.atom_numbers.shape[0]
        self.num_types = len(self.type_to_number)

        self.one_hot = torch.eye(self.num_types, dtype=torch.float)[torch.tensor(self.atom_numbers.flatten())].reshape(-1, self.num_atoms, self.num_types)
        self.charges = torch.tensor([self.ground_charges[atom] for atom in self.atom_numbers.flatten()]).reshape(-1, self.num_atoms)

        self.atom_mask = torch.ones((self.num_atoms), dtype=torch.float).reshape(-1, 1)
        self.edge_mask = torch.eye((self.num_atoms), dtype=torch.float).flatten().reshape(-1, 1)
        self.edge_mask = torch.where(self.edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        self.energies = torch.tensor(self.energies)
        self.coordinates = torch.tensor(self.coordinates)
        

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, idx):
        return {"one_hot" : self.one_hot[idx],
                "coordinates" : self.coordinates[idx],
                "energies" : self.energies[idx],
                "atom_mask": self.atom_mask,
                "edge_mask": self.edge_mask,
                "charges": self.charges[idx]}
    

    def _read_coordinates(self, foldername):
        all_atom_numbers = []
        all_coordinates = []
        for filename in os.listdir(foldername):
            if filename.endswith(".xyz"):
                with open(os.path.join(foldername, filename), 'r') as file:
                    while True:
                        line = file.readline()
                        if line == "":
                            break
                        num_atoms = int(line.strip())
                        


                    # Skip the comment line
                        file.readline()

                    # Initialize lists to store atom types and coordinates
                        atom_numbers = []

                        coordinates = []

                        # Read atom types and coordinates for each atom
                        for _ in range(num_atoms):
                            line = file.readline().split()
                            atom_type = line[0]
                            x, y, z = map(float, line[1:4])

                            atom_numbers.append(int(self.type_to_number[atom_type]))
                            coordinates.append([x, y, z]) 
                        all_coordinates.append(coordinates)
                        all_atom_numbers.append(atom_numbers)
        return np.array(all_atom_numbers), np.array(all_coordinates)


    def _read_energies(self, foldername):
        energies = []
        for filename in os.listdir(foldername):
            with open(os.path.join(foldername, filename)) as file:
                energies.extend(np.array([float(e) for e in file.readline()[1:-1].split(",")]))
        return np.array(energies)



if __name__ == "__main__":
    ds = MDDataset("datasets/files/alaninedipeptide")