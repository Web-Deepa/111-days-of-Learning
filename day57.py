import torch, torchvision, torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# 1.minimal cnn
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
loader = DataLoader(train_data, batch_size=128, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(8 * 13 * 13, 10)
        )
    def forward(self, x): return self.net(x)
#optimizer configuration
configs = {
    'SGD (No Scheduler)': {
        'opt': lambda p: torch.optim.SGD(p, lr=0.1),
        'sched': lambda o: None
    },
    'RMSprop + Cosine': {
        'opt': lambda p: torch.optim.RMSprop(p, lr=0.01),
        'sched': lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=100)
    },
    'Adam + Cosine': {
        'opt': lambda p: torch.optim.Adam(p, lr=0.01),
        'sched': lambda o: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=100)
    }
}

# table and loop
for name, cfg in configs.items():
    torch.manual_seed(42)  
    model = SimpleCNN()
    optimizer = cfg['opt'](model.parameters())
    scheduler = cfg['sched'](optimizer)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss, correct, total = 0, 0, 0
    start_lr = optimizer.param_groups[0]['lr']
    
    for i, (images, labels) in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # update scheduler
        if scheduler is not None:
            scheduler.step()
            
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i == 100: break
    end_lr = optimizer.param_groups[0]['lr']
    accuracy = 100. * correct / total
    
    print(f"{name:20} -> Loss: {total_loss/100:.4f} | Accuracy: {accuracy:.2f}% | LR: {start_lr} -> {end_lr:.5f}")
