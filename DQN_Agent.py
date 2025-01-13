import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_content=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_content)
    return layer

class Agent(nn.Module):
    def __init__(self, input_dim, num_actions, image_obs=False):
        r"""
        input_dim : the image shape, in case image_obs is true else number of features in observation space
        num_actions : the number of outputs in action space
        image_obs : boolean is True if observation is image else False
        """
        self.img_obs = image_obs
        super().__init__()
        if self.img_obs and len(input_dim) > 1:
            self.net = nn.Sequential(
                layer_init(nn.Conv2d(input_dim[0], out_channels=32, kernel_size=4, stride=2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)),
                nn.ReLU(),
                layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)),
                nn.ReLU(),
            )
            dummy_input = torch.zeros(1, *input_dim)
            with torch.no_grad():
                out = self.net(dummy_input)
                out = out.view(1, -1)
            flattened_size = out.shape[-1]
            print("Flattened Shape :", flattened_size)
            self.fc = nn.Linear(flattened_size, num_actions)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, num_actions)
            )  
            
    def forward(self, x):
        x = self.net(x)
        if self.img_obs: # if processing image
            x = torch.flatten(x, start_dim=1)
            x = self.fc(x)
        return x
    
    def act(self, obs):
        with torch.no_grad():
            q_val = self.forward(obs.unsqueeze(0)) # this is to make the observation shape (1, 1, img_W, img_H)
            max_q_action = torch.argmax(q_val, dim=1)[0].item()
        return max_q_action
    

def pre_process(obs, range_dim, image=False, device="cpu"):
    """
    Converts the observation to torch tensor
    obs : Observation sent from env
    range_dim : To reduce the size of the image, (start, end)
    image : Boolean is False if obs is not image else True
    device : To shift tensor to desired tensor
    """
    if image:
        assert isinstance(range_dim, tuple), f"Expected {tuple}, got {type(range_dim)}"
        assert len(range_dim) == 2, f"Expected range_dim to have 2 values (start, end)"
        return torch.tensor(obs[range_dim[0]:range_dim[1], :, 0], device=device).unsqueeze(0) / 255 # removing extra clutter, and taking only one dimension
    
    return torch.tensor(obs, device=device)
