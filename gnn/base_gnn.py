from abc import ABC, abstractmethod
import torch


class BaseGNN(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def forward(self, data):
        pass

        