import torch
from torch.nn import Module, MSELoss, L1Loss
from .common_losses import SmoothnessLoss


class SupervisedL1Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL1Loss, self).__init__()
        self.l1_loss = L1Loss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l1_loss(pred_flow, gt_flow)


class SupervisedL2Loss(Module):
    def __init__(self, **kwargs):
        super(SupervisedL2Loss, self).__init__()
        self.l2_loss = MSELoss()

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        return self.l2_loss(pred_flow, gt_flow)


class SupervisedL1RegLoss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, **kwargs):
        super(SupervisedL1RegLoss, self).__init__()
        self.data_loss = L1Loss()
        self.smoothness_loss = SmoothnessLoss(**smoothness_loss_params)
        self.w_data = w_data
        self.w_smoothness = w_smoothness

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor, i=0) -> torch.Tensor:
        if len(self.w_data) == 1:
            w_data = self.w_data[0]
            w_smoothness = self.w_smoothness[0]
        else:
            w_data = self.w_data[i]
            w_smoothness = self.w_smoothness[i]

        loss = (w_data * self.data_loss(pred_flow, gt_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
        return loss