import torch
import torch.nn as nn
from controllers.validity_mask import apply_action_mask

class Controller(nn.Module):
    def __init__(self, num_filters, num_kernels, num_dropouts, input_size=1, hidden_size=100):
        super(Controller, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        # Hyperparameter selection heads
        self.f_head = nn.Linear(hidden_size, num_filters)
        self.k_head = nn.Linear(hidden_size, num_kernels)
        self.d_head = nn.Linear(hidden_size, num_dropouts)

    def forward(self, apply_mask=False, mask_filter_idx=3):
        # Input tensor initialization on CPU for stability
        x = torch.zeros(1, 1, 1)
        out, _ = self.lstm(x)
        out = out.squeeze(0)
        
        f_logits = self.f_head(out)
        k_logits = self.k_head(out)
        d_logits = self.d_head(out)
        
        if apply_mask:
            f_logits = apply_action_mask(f_logits, mask_filter_idx)
            
        return f_logits, k_logits, d_logits