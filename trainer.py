import torch
import torch.nn as nn
from colorist import Color
import time
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, valid_loader, optimizer, scheduler, device):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_acc_list = []
        self.valid_losses = []
        self.valid_acc_list = []
        self.best_acc = 0.0
        self.model_checkpoint_dir = './model_checkpoints/'
        self.init_time = time.strftime("%Y-%m-%d_%H.%M.%S")
        if os.path.exists(f"./model_checkpoints/{self.init_time}") is False:
            os.makedirs(f"./model_checkpoints/{self.init_time}")
            print("Created new directory for model checkpoints at", f"./model_checkpoints/{self.init_time}")
    
    def train(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0

        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"{Color.MAGENTA}Epoch {epoch}{Color.OFF}")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = total_loss / len(self.train_loader.dataset)
        train_accuracy = 100. * correct / len(self.train_loader.dataset)
        self.train_losses.append(train_loss)
        self.train_acc_list.append(train_accuracy)

        print(f"Train set ===> Average Loss: {Color.RED}{train_loss:.4f}{Color.OFF} | Accuracy: {correct}/{len(self.train_loader.dataset)} ({Color.CYAN}{train_accuracy:.2f}%{Color.OFF})")

    def validate(self):
        self.model.eval()
        correct = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.valid_loader.dataset)
        valid_accuracy = 100. * correct / len(self.valid_loader.dataset)
        self.valid_acc_list.append(valid_accuracy)
        self.valid_losses.append(val_loss)
        print(f"Test set  ===> Average Loss: {Color.RED}{val_loss:.4f}{Color.OFF} | Accuracy: {correct}/{len(self.valid_loader.dataset)} ({Color.GREEN}{valid_accuracy:.2f}%{Color.OFF})")

        if valid_accuracy > self.best_acc:
            self.best_acc = valid_accuracy
            self.save_model(f'./model_checkpoints/{self.init_time}/best_model.pth')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved at {path}")
    
    def __call__(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            self.validate()
            print(f"Best Accuracy: [\033[1;32m{round(float(self.best_acc),3)}%\033[0m]")
            self.scheduler.step(epoch)