import torch
from torch.nn import Module, MSELoss, L1Loss
from .common_losses import SmoothnessLoss, ChamferLoss


class UnSupervisedL1Loss(Module):
    def __init__(self, w_data, w_smoothness, smoothness_loss_params, chamfer_loss_params, **kwargs):
        super(UnSupervisedL1Loss, self).__init__()
        self.data_loss = ChamferLoss(**chamfer_loss_params)
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
        loss = (w_data * self.data_loss(pc_source, pc_target, pred_flow)) + (w_smoothness * self.smoothness_loss(pc_source, pred_flow))
        return loss