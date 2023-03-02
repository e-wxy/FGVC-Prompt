import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLoss(nn.Module):
    def __init__(self, sim_gw=0.5, loss_fct=nn.CrossEntropyLoss()):
        """ Loss Function for TokenFlow

        Args:
            sim_gw (float): weights for global feature similarity
            loss_fct (nn.Module)
        """
        super().__init__()
        self.sim_gw = sim_gw
        self.loss_fct = loss_fct

    def forward(self, sim_g, sim_v, sim_t):
        loss_v = self.loss_fct(sim_v, torch.arange(sim_v.shape[-1]))
        loss_t = self.loss_fct(sim_t, torch.arange(sim_t.shape[-1]))
        loss_g = self.loss_fct(sim_g, torch.arange(sim_v.shape[-1])) + self.loss_fct(sim_g.t(), torch.arange(sim_t.shape[-1]))

        loss = loss_g * self.sim_gw + (loss_v + loss_t) * (1 - self.sim_gw)

        return loss