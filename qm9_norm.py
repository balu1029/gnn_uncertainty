from datasets.qm9 import QM9
import torch

if __name__ == "__main__":
    qm9 = QM9()
    qm9.create(128,0)
    trainloader = qm9.dataloaders["train"]
    total = 0
    for data in trainloader:
        total += torch.sum(data["homo"])
    print(total / len(trainloader.dataset))