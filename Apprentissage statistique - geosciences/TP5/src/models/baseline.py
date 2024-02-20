import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, kernel_size):
        super(BaselineModel, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size, stride=1, padding=1)
        self.conv5 = nn.Conv2d(hidden_dim*2, out_channels, kernel_size, stride=1, padding=1)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x)
        x = self.sigmoid(x)
        
        return x

if __name__ == "__main__":
    # Define the model architecture
    in_channels = 1
    out_channels = 1
    hidden_dim = 64
    kernel_size = 3

    # Create an instance of the model
    model = BaselineModel(in_channels, out_channels, hidden_dim, kernel_size)

    # Print the model architecture
    print(model)