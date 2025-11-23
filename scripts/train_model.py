#!/usr/bin/env python3
"""
Train the CNN model using PyTorch and save weights for C++ inference
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import struct
import os

class SimpleCNN(nn.Module):
    """Matches the C++ architecture exactly"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv(1 -> 8 channels, 5x5 kernel)
        self.conv = nn.Conv2d(1, 8, kernel_size=5, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # FC(1152 -> 10)
        self.fc = nn.Linear(8 * 12 * 12, 10)
        
    def forward(self, x):
        x = self.conv(x)      # (N, 8, 24, 24)
        x = self.relu(x)      # (N, 8, 24, 24)
        x = self.pool(x)      # (N, 8, 12, 12)
        x = x.view(x.size(0), -1)  # Flatten to (N, 1152)
        x = self.fc(x)        # (N, 10)
        return x

def save_weights_cpp_format(model, filename):
    """Save weights in C++ readable format (binary float)"""
    with open(filename, 'wb') as f:
        # Conv weights: (8, 1, 5, 5) = 200 floats
        conv_w = model.conv.weight.data.cpu().numpy().flatten()
        for w in conv_w:
            f.write(struct.pack('f', float(w)))
        
        # Conv bias: (8,) = 8 floats
        conv_b = model.conv.bias.data.cpu().numpy().flatten()
        for b in conv_b:
            f.write(struct.pack('f', float(b)))
        
        # FC weights: (10, 1152) = 11520 floats
        fc_w = model.fc.weight.data.cpu().numpy().flatten()
        for w in fc_w:
            f.write(struct.pack('f', float(w)))
        
        # FC bias: (10,) = 10 floats
        fc_b = model.fc.bias.data.cpu().numpy().flatten()
        for b in fc_b:
            f.write(struct.pack('f', float(b)))
    
    print(f"Saved weights to {filename}")

def train_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Training
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f'Epoch {epoch+1} - Test Accuracy: {test_acc:.2f}%')
        print('-' * 50)
    
    # Save weights
    os.makedirs('weights', exist_ok=True)
    save_weights_cpp_format(model, 'weights/model_weights.bin')
    
    # Final test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    final_acc = 100. * test_correct / test_total
    print(f'\nFinal Test Accuracy: {final_acc:.2f}%')
    print(f'Weights saved to weights/model_weights.bin')

if __name__ == '__main__':
    train_model()

