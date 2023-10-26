import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__) ))
from components.trainer.trainer import TrainOrchestrator
from components.dataloader.data_loader import DataLoader
import matplotlib.pyplot as plt
from PIL import Image

def main(data_path):
    train_df, valid_df, test_df = DataLoader().load_data(data_path)
    train_loader, valid_loader, test_loader = DataLoader().create_dataloaders(train_df, valid_df, test_df)
    train_orchestrator = TrainOrchestrator()
    train_orchestrator.run(train_loader, valid_loader, test_loader)

if __name__ == "__main__":
    main(data_path = "Dataset/animals/animals")