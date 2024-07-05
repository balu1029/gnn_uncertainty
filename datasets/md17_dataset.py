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
        
        self.atom_numbers, self.coordinates, self.energies = self._read_coordinates_energies(foldername)
        self.total_self_energy = [sum(self_energies[atom] for atom in molecule) * self.Eh_to_eV for molecule in self.atom_numbers] 
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
        # Find the length of the longest element in the sequence of coordinates
        max_length_coords = max(len(coords) for coords in all_coordinates)

        # Find the length of the longest element in the sequence of atom numbers
        max_length_atom_numbers = max(len(nums) for nums in all_atom_numbers)

        # Pad all the coordinates to the same size
        padded_coordinates = []
        for coords in all_coordinates:
            # Calculate the number of padding elements needed
            num_padding = max_length_coords - len(coords)
            # Pad the coordinates with zeros
            padded_coords = coords + [[0, 0, 0]] * num_padding
            padded_coordinates.append(padded_coords)

        # Pad all the atom numbers to the same size
        padded_atom_numbers = []
        for nums in all_atom_numbers:
            # Calculate the number of padding elements needed
            num_padding = max_length_atom_numbers - len(nums)
            # Pad the atom numbers with zeros
            padded_nums = nums + [-1] * num_padding
            padded_atom_numbers.append(padded_nums)

        # Convert the padded coordinates and atom numbers back to numpy arrays
        padded_coordinates = np.array(padded_coordinates)
        padded_atom_numbers = np.array(padded_atom_numbers)

        # Update the return statement
        return padded_atom_numbers, padded_coordinates, np.array(energies)
        #return np.array(all_atom_numbers), np.array(all_coordinates), np.array(energies)





if __name__ == "__main__":
    ds = MD17Dataset("datasets/files/md17_double")
    print(ds.coordinates.shape)
    print(ds.atom_numbers.shape)
