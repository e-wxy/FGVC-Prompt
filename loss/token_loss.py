import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLoss(nn.Module):
    def __init__(self, sim_gw=0.5, t_global=0.07, t_part=0.07, loss_fct=nn.CrossEntropyLoss()):
        """ Loss Function for TokenFlow

        Args:
            sim_gw (float): weights for global feature similarity
            t_global (float): temperature for global features
            t_part (float): temperature for patch/word features
            loss_fct (nn.Module): Defaults to InfoNCE
        """
        super().__init__()
        self.sim_gw = sim_gw
        self.loss_fct = loss_fct
        self.t_g = t_global     # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.t_p = t_part

    def forward(self, sim_g, sim_v, sim_t):
        # logit scale
        sim_g = sim_g / self.t_g
        sim_v /= self.t_p
        sim_t /= self.t_p
        # cal loss
        loss_v = self.loss_fct(sim_v, torch.arange(sim_v.shape[-1]).to(sim_v.device))
        loss_t = self.loss_fct(sim_t, torch.arange(sim_t.shape[-1]).to(sim_t.device))
        loss_g = (self.loss_fct(sim_g, torch.arange(sim_v.shape[-1]).to(sim_g.device)) + \
                  self.loss_fct(sim_g.t(), torch.arange(sim_t.shape[-1]).to(sim_g.device))) / 2

        loss = loss_g * self.sim_gw + (loss_v + loss_t) * (1 - self.sim_gw) / 2

        return loss
    

class ContrastiveLoss(nn.Module):
    def __init__(self, sim_gw=0.5, t_global=0.07, t_part=0.07, loss_fct=nn.CrossEntropyLoss()):
        """ Loss Function for TokenFlow

        Args:
            sim_gw (float): weights for global feature similarity
            loss_fct (nn.Module)
        """
        super().__init__()
        self.sim_gw = sim_gw
        self.loss_fct = loss_fct
        self.t_g = t_global     # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, sim_g, sim_v, sim_t):
        # logit scale
        sim_g = sim_g / self.t_g
        loss_g = (self.loss_fct(sim_g, torch.arange(sim_g.shape[-1]).to(sim_g.device)) + \
                  self.loss_fct(sim_g.t(), torch.arange(sim_g.shape[-2]).to(sim_g.device))) / 2

        return loss_g