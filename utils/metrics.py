import torch
from pytorch_lightning.metrics import TensorMetric
from typing import Any, Optional
from losses.supervised_losses import *
from losses.unsupervised_losses import *
from losses.common_losses import *


class EPE3D(TensorMetric):
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        epe3d = torch.norm(pred_flow - gt_flow, dim=2).mean()
        return epe3d

class Acc3DR(TensorMetric):
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_relax = (torch.logical_or(l2_norm < 0.1, relative_err < 0.1)).float().mean()
        return acc3d_relax

class Acc3DS(TensorMetric):
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        acc3d_strict = (torch.logical_or(l2_norm < 0.05, relative_err < 0.05)).float().mean()
        return acc3d_strict

class EPE3DOutliers(TensorMetric):
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(pred_flow - gt_flow, dim=2)
        sf_norm = torch.norm(gt_flow, dim=2)
        relative_err = l2_norm / (sf_norm + 1e-4)
        epe3d_outliers = (torch.logical_or(l2_norm > 0.3, relative_err > 0.1)).float().mean()
        return epe3d_outliers

class SupervisedL1LossMetric(TensorMetric):
    def __init__(self, name: str, reduce_op: Optional[Any] = None):
        super(SupervisedL1LossMetric, self).__init__(name=name, reduce_op=reduce_op)
        self.loss = SupervisedL1Loss()
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow, gt_flow)
        return loss_metric


class SmoothnessLossMetric(TensorMetric):
    def __init__(self, smoothness_loss_params, name: str, reduce_op: Optional[Any] = None):
        super(SmoothnessLossMetric, self).__init__(name=name, reduce_op=reduce_op)
        self.loss = SmoothnessLoss(**smoothness_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pred_flow)
        return loss_metric

class ChamferLossMetric(TensorMetric):
    def __init__(self, chamfer_loss_params, name: str, reduce_op: Optional[Any] = None):
        super(ChamferLossMetric, self).__init__(name=name, reduce_op=reduce_op)
        self.loss = ChamferLoss(**chamfer_loss_params)
    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor, gt_flow: torch.Tensor) -> torch.Tensor:
        loss_metric = self.loss(pc_source, pc_target, pred_flow)
        return loss_metric


class SceneFlowMetrics():
    """
    An object of relevant metrics for scene flow.
    """

    def __init__(self, split: str, loss_params: dict, reduce_op: Optional[Any] = None):
        """
        Initializes a dictionary of metrics for scene flow
        keep reduction as 'none' to allow metrics computation per sample.

        Arguments:
            split : a string with split type, should be used to allow logging of same metrics for different aplits
            loss_params: loss configuration dictionary
            reduce_op: the operation to perform during reduction within DDP (only needed for DDP training).
                        Defaults to sum.
        """

        self.metrics = {
            split + '_epe3d': EPE3D(name='epe3d', reduce_op=reduce_op),

        }
        if loss_params['loss_type'] == 'sv_l1_reg':
            self.metrics[f'{split}_data_loss'] = SupervisedL1LossMetric(name=f'{split}_data_loss', reduce_op=reduce_op)
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'], name=f'{split}_smoothness_loss', reduce_op=reduce_op)
        if loss_params['loss_type'] == 'unsup_l1':
            self.metrics[f'{split}_chamfer_loss'] = ChamferLossMetric(loss_params['chamfer_loss_params'], name=f'{split}_chamfer_loss', reduce_op=reduce_op)
            self.metrics[f'{split}_smoothness_loss'] = SmoothnessLossMetric(loss_params['smoothness_loss_params'], name=f'{split}_smoothness_loss', reduce_op=reduce_op)

        if split in ['test', 'val']:
            self.metrics[f'{split}_acc3dr'] = Acc3DR(name='acc3dr', reduce_op=reduce_op)
            self.metrics[f'{split}_acc3ds'] = Acc3DS(name='acc3ds', reduce_op=reduce_op)
            self.metrics[f'{split}_epe3d_outliers'] = EPE3DOutliers(name='epe3d_outliers', reduce_op=reduce_op)

    def __call__(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flows: list, gt_flow: torch.Tensor) -> dict:
        """
        Compute and scale the resulting metrics

        Arguments:
            pc_source : a tensor containing source point cloud
            pc_target : a tensor containing target point cloud
            pred_flows : list of tensors containing model's predictions
            gt_flow : a tensor containing ground truth labels

        Return:
            A dictionary of copmuted metrics
        """

        result = {}
        for key, metric in self.metrics.items():
            for i, pred_flow in enumerate(pred_flows):
                val = metric(pc_source, pc_target, pred_flow, gt_flow)
                result.update({f'{key}_i#{i}': val})

        return result
