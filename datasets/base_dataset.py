from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class Base_Dataset(ABC, Dataset):

    @abstractmethod
    def create(self):
        pass