import torch
from torch.nn import Module, MSELoss, L1Loss
from lib.pointnet2 import pointnet2_utils as pointutils


class KnnLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(KnnLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        dist, idx = pointutils.knn(self.k, pc_source, pc_source)
        tmp_idx = idx[:, :, 0].unsqueeze(2).repeat(1, 1, self.k).to(idx.device)
        idx[dist > self.radius] = tmp_idx[dist > self.radius]
        nn_flow = pointutils.grouping_operation(flow, idx.detach())
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class BallQLoss(Module):
    def __init__(self, k, radius, loss_norm, **kwargs):
        super(BallQLoss, self).__init__()
        self.k = k
        self.radius = radius
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        flow = pred_flow.permute(0, 2, 1).contiguous()
        idx = pointutils.ball_query(self.radius, self.k, pc_source, pc_source)
        nn_flow = pointutils.grouping_operation(flow, idx.detach())  # retrieve flow of nn
        loss = (flow.unsqueeze(3) - nn_flow).norm(p=self.loss_norm, dim=1).mean( dim=-1)
        return loss.mean()


class SmoothnessLoss(Module):
    def __init__(self, w_knn, w_ball_q, knn_loss_params, ball_q_loss_params, **kwargs):
        super(SmoothnessLoss, self).__init__()
        self.knn_loss = KnnLoss(**knn_loss_params)
        self.ball_q_loss = BallQLoss(**ball_q_loss_params)
        self.w_knn = w_knn
        self.w_ball_q = w_ball_q

    def forward(self, pc_source: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        loss = (self.w_knn * self.knn_loss(pc_source, pred_flow)) + (self.w_ball_q * self.ball_q_loss(pc_source, pred_flow))
        return loss


class ChamferLoss(Module):
    def __init__(self, k, loss_norm, **kwargs):
        super(ChamferLoss, self).__init__()
        self.k = k
        self.loss_norm = loss_norm

    def forward(self, pc_source: torch.Tensor, pc_target: torch.Tensor, pred_flow: torch.Tensor) -> torch.Tensor:
        pc_target = pc_target.contiguous()
        pc_target_t = pc_target.permute(0, 2, 1).contiguous()
        pc_pred = (pc_source + pred_flow).contiguous()
        pc_pred_t = pc_pred.permute(0, 2, 1).contiguous()

        _, idx = pointutils.knn(self.k, pc_pred, pc_target)
        nn1 = pointutils.grouping_operation(pc_target_t, idx.detach())
        dist1 = (pc_pred_t.unsqueeze(3) - nn1).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        _, idx = pointutils.knn(self.k, pc_target, pc_pred)
        nn2 = pointutils.grouping_operation(pc_pred_t, idx.detach())
        dist2 = (pc_target_t.unsqueeze(3) - nn2).norm(p=self.loss_norm, dim=1).mean( dim=-1)  # nn flow consistency
        ch_dist = (dist1 + dist2)
        return ch_dist.mean()