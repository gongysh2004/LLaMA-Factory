import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x.shape [64, 1, 28, 28]  # batch_size=64, channels=1, height=28, width=28
        x = self.conv1(x)          # [64, 32, 26, 26] - Conv2d(1→32, kernel=3, stride=1): (28-3+0)/1+1=26
        x = torch.relu(x)          # [64, 32, 26, 26] - 激活函数不改变shape
        x = self.conv2(x)          # [64, 64, 24, 24] - Conv2d(32→64, kernel=3, stride=1): (26-3+0)/1+1=24
        x = torch.relu(x)          # [64, 64, 24, 24] - 激活函数不改变shape
        x = torch.max_pool2d(x, 2) # [64, 64, 12, 12] - 池化窗口2x2, stride=2: 24/2=12
        x = self.dropout1(x)       # [64, 64, 12, 12] - Dropout不改变shape
        x = torch.flatten(x, 1)    # [64, 9216]       - 展平: 64*12*12=9216
        x = self.fc1(x)            # [64, 128]        - Linear(9216→128)
        x = torch.relu(x)          # [64, 128]        - 激活函数不改变shape
        x = self.dropout2(x)       # [64, 128]        - Dropout不改变shape
        x = self.fc2(x)            # [64, 10]         - Linear(128→10)
        output = torch.log_softmax(x, dim=1)  # [64, 10] - log_softmax不改变shape
        return output


def load_checkpoint(model, optimizer):
    """Load checkpoint if exists, return starting epoch"""
    import os
    checkpoint_path = 'checkpoint.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")
    return 0


def save_checkpoint(model, optimizer, epoch):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'checkpoint.pth')


def main():
    # 设置设备（单GPU或CPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 修复cuDNN错误：设置cuDNN为确定性模式并禁用benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 清空GPU缓存
        torch.cuda.empty_cache()

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 数据加载器（减少num_workers以避免cuDNN冲突）
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    # 初始化模型、优化器、损失函数
    model = SimpleCNN().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    criterion = nn.NLLLoss()

    # 加载Checkpoint，获取起始epoch
    start_epoch = load_checkpoint(model, optimizer)
    epochs = 1  # 总训练epoch数

    # 训练循环
    for epoch in range(start_epoch, epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # print(f'Batch {batch_idx} data shape: {data.shape}, target shape: {target.shape}')
                # torch.Size([64, 1, 28, 28]), target shape: torch.Size([64])
                data, target = data.to(device), target.to(device)
                # 梯度计算核心步骤拆解：
                # 1. 清空梯度缓存：PyTorch中梯度会累积，若不清空会导致当前批次梯度与上批次叠加，影响参数更新
                optimizer.zero_grad()
                # 2. 前向传播：将输入数据传入模型，得到预测输出（log_softmax后的概率分布）
                output = model(data)
                # 3. 计算损失：通过损失函数（NLLLoss）计算预测输出与真实标签的偏差，量化模型误差
                loss = criterion(output, target)
                # 4. 反向传播：从损失值出发，沿网络反向计算每个可训练参数的梯度（链式法则），梯度值会存入参数的.grad属性
                loss.backward()
                # 5. 参数更新：优化器（Adadelta）根据计算出的梯度，调整模型参数以减小损失（核心逻辑为梯度下降）
                optimizer.step()

                # 打印训练信息
                if batch_idx % 100 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            except RuntimeError as e:
                if 'cuDNN' in str(e):
                    print(f'cuDNN error at batch {batch_idx}, clearing cache and retrying...')
                    torch.cuda.empty_cache()
                    # 跳过这个batch
                    continue
                else:
                    raise

        # 每个epoch结束后保存Checkpoint
        save_checkpoint(model, optimizer, epoch)

        # 测试阶段
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        # 打印测试信息
        print(f'\nTest set: Average loss: {test_loss:.4f}, '
              f'Accuracy: {correct}/{len(test_loader.dataset)} '
              f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    main()