import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.onnx
import numpy as np

## LeNet5 Model for inference
class LeNet5_Infer(nn.Module):
    def __init__(self, input_size):
        super(LeNet5_Infer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        fc1_h = int(input_size[0] / 4 - 3)
        fc1_w = int(input_size[1] / 4 - 3)
        self.fc1 = nn.Linear(fc1_h * fc1_w * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    # load trained parameters (mnist_cnn.pt) into infer model
    model = LeNet5_Infer([28,28])
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    model.eval()
    
    # Input to the model
    x = torch.randn(1, 1, 28, 28)
    torch_out = model(x)

    # Export the model as onnx (lenet5.onnx)
    torch.onnx.export(model,             # model being run
                        x,               # model input (or a tuple for multiple inputs)
                        "lenet5.onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=10,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
