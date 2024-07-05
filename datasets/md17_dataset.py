import numpy as np
import torch
import os


class MD17Dataset(torch.utils.data.Dataset):
    def __init__(self, foldername):
        self.type_to_number = {"H" : 0,
                               "C" : 1,
                               "N" : 2,
                               "O" : 3}
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        self_energies = {-1: 0.0,
                         0: -0.500607632585,
                         1: -37.8302333826,
                         2: -54.5680045287,
                         3: -75.0362229210}
        self.Eh_to_eV = 27.211399
        self.kcal_to_eV = 0.0433641
        
        self.atom_numbers_raw, self.coordinates_raw, self.energies = self._read_coordinates_energies(foldername)
        self.atom_numbers = self._pad_array(self.atom_numbers_raw, fill = [-1])
        self.coordinates = self._pad_array(self.coordinates_raw, fill = [[0, 0, 0]])

        self.atom_mask = torch.where(torch.tensor(self.atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
    
        self.atom_numbers = np.array(self.atom_numbers)
        self.coordinates = np.array(self.coordinates)
        self.energies = np.array(self.energies)

        self.total_self_energy = [sum(self_energies[atom] for atom in molecule) * self.Eh_to_eV for molecule in self.atom_numbers] 
        self.energies = self.energies - self.total_self_energy

        self.num_atoms = self.atom_numbers.shape[1]
        self.num_snapshots = self.atom_numbers.shape[0]
        self.num_types = len(self.type_to_number)

        self.one_hot = torch.eye(self.num_types, dtype=torch.float)[torch.tensor(self.atom_numbers.flatten())].reshape(-1, self.num_atoms, self.num_types)
        self.charges = torch.tensor([self.ground_charges[atom] for atom in self.atom_numbers.flatten()]).reshape(-1, self.num_atoms)

        #self.atom_mask = torch.ones((self.num_atoms), dtype=torch.float).reshape(-1, 1)
        self.edge_mask = torch.eye(self.num_atoms, dtype=torch.float).unsqueeze(0).expand(self.num_snapshots, -1, -1)
        self.edge_mask = torch.where(self.edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        self.edge_mask = self.edge_mask * self.atom_mask.unsqueeze(1) * self.atom_mask.unsqueeze(2)
        self.edge_mask = self.edge_mask.view(self.num_snapshots, -1)

        self.energies = torch.tensor(self.energies)
        self.coordinates = torch.tensor(self.coordinates)
        

    def __len__(self):
        return self.num_snapshots

    def __getitem__(self, idx):
        return {"one_hot" : self.one_hot[idx],
                "coordinates" : self.coordinates[idx],
                "energies" : self.energies[idx],
                "atom_mask": self.atom_mask[idx],
                "edge_mask": self.edge_mask[idx],
                "charges": self.charges[idx]}
    

    def _read_coordinates_energies(self, foldername):
        all_atom_numbers = []
        all_coordinates = []
        energies = []
        for filename in os.listdir(foldername):
            if filename.endswith(".xyz"):
                with open(os.path.join(foldername, filename), 'r') as file:
                    while True:
                        line = file.readline()
                        if line == "":
                            break
                        num_atoms = int(line.strip())
                        


                    # Skip the comment line
                        energy = float(file.readline().strip()) * self.kcal_to_eV
                        
                        energies.append(energy)

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

        return all_atom_numbers, all_coordinates, energies
    

    def _pad_array(self, arr, fill = [-1]):
        max_length = max(len(nums) for nums in arr)
        padded_arr = []
        for nums in arr:
            # Calculate the number of padding elements needed
            num_padding = max_length - len(nums)
            # Pad the atom numbers with the filling element
            padded_nums = nums + fill * num_padding
            padded_arr.append(padded_nums)

    
        return padded_arr






if __name__ == "__main__":
    ds = MD17Dataset("datasets/files/md17_double")
    
