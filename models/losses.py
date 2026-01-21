import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        """
        inputs: predicted probabilities (after sigmoid), shape [batch_size]
        targets: true labels (0 or 1), shape [batch_size]
        """
        # Clamp to prevent log(0)
        inputs = torch.clamp(inputs, min=1e-7, max=1-1e-7)
        
        # Calculate focal loss
        bce_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # Modulating factor
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha balancing
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_weight * focal_weight * bce_loss
        
        return focal_loss.mean()
