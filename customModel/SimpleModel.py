import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size ,hidden_layer, num_class , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.sequencial = nn.Sequential(
            nn.Linear(input_size, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,  hidden_layer//2),
            nn.ReLU(),
            nn.Linear(hidden_layer//2, hidden_layer//4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_layer//4, hidden_layer//8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_layer//8, num_class)
        )
    
    def forward(self, x):
        input = x.view(x.shape[0], -1)
        out = self.sequencial(input)

        return out