import torch
import torch.nn as nn

class ChildNet(nn.Module):
    def __init__(self, f_size, k_size, d_val):
        super(ChildNet, self).__init__()
        self.conv = nn.Conv2d(1, f_size, kernel_size=k_size, padding=k_size // 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(d_val)
        self.fc = nn.Linear(f_size * 14 * 14, 11)  # Fixed out-features matching OrganAMNIST classes

    def forward(self, x):
        x = self.dropout(self.pool(torch.relu(self.conv(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)