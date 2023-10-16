import torch

import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    def __init__(self, phos_w=1, phoc_w=0.1, phoc_class_weights=None):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

        self.mse_loss = nn.MSELoss()

        if phoc_class_weights is not None:
            phoc_class_weights = torch.tensor(phoc_class_weights).float()
        self.ce_loss = nn.CrossEntropyLoss(weight=phoc_class_weights)

    def forward(self, y: dict, targets: torch.Tensor):
        phos_size = 165
        phoc_size = 604
        # Extracting PHOS and PHOC outputs and targets

        y_phos, y_phoc = y["phos"], y["phoc"]

        t_phos, t_phoc = targets[:, :phos_size], targets[:, phos_size:]

        phos_loss = self.mse_loss(y_phos, t_phos)

        # Assuming t_phoc contains class indices and not one-hot encoded vectors
        phoc_loss = self.ce_loss(y_phoc, torch.argmax(t_phoc, dim=1))

        # Weighting the losses
        loss = self.phos_w * phos_loss + self.phoc_w * phoc_loss
        
        return loss
