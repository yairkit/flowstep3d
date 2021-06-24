import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from utils.metrics import SceneFlowMetrics
from utils.utils import get_num_workers
from utils.exp_utils import get_datasets
from losses import *


class SceneFlowExp(pl.LightningModule):
    """
    Class for Scene Flow experiment.
    """

    def __init__(self, model, hparams):
        """
        Initializes an experiment.

        Arguments:
            model : a torch.nn.Module object, model to be tested
            hparams : a dictionary of hyper parameters
        """

        super(SceneFlowExp, self).__init__()
        self.model = model
        self.hparams = hparams
        self.loss = losses_dict[hparams['loss']['loss_type']](**hparams['loss'])
        self.train_metrics = SceneFlowMetrics(split='train', loss_params=hparams['loss'], reduce_op='mean')
        self.val_metrics = SceneFlowMetrics(split='val', loss_params=hparams['loss'], reduce_op='mean')

    def forward(self, pos1, pos2, feat1, feat2, iters):
        """
        A forward call
        """
        return self.model(pos1, pos2, feat1, feat2, iters)

    def sequence_loss(self, pos1, pos2, flows_pred, flow_gt):
        if 'loss_iters_w' in self.hparams:
            assert (len(self.hparams['loss_iters_w']) == len(flows_pred))
            loss = torch.zeros(1).cuda()
            for i, w in enumerate(self.hparams['loss_iters_w']):
                loss += w * self.loss(pos1, pos2, flows_pred[i], flow_gt, i)
        else:
            loss = self.loss(pos1, pos2, flows_pred[-1], flow_gt)
        return loss
        
    def training_step(self, batch, batch_idx):
        """
        Executes a single training step
        """

        pos1, pos2, feat1, feat2, flow_gt, fnames = batch
        flows_pred = self(pos1, pos2, feat1, feat2, self.hparams['train_iters'])
        loss = self.sequence_loss(pos1, pos2, flows_pred, flow_gt)
        metrics = self.train_metrics(pos1, pos2, flows_pred, flow_gt)

        train_results = pl.TrainResult(loss)
        train_results.log('train_loss', loss, sync_dist=True, on_step=False, on_epoch=True, logger=True, prog_bar=False, reduce_fx=torch.mean)
        train_results.log_dict(metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False, reduce_fx=torch.mean)  # No need to sync_dist since metrics are already synced     
        
        return train_results

    def _test_val_step(self, batch, batch_idx, split):
        pos1, pos2, feat1, feat2, flow_gt, fnames = batch
        flows_pred = self(pos1, pos2, feat1, feat2, self.hparams[f'{split}_iters'])
        loss = self.sequence_loss(pos1, pos2, flows_pred, flow_gt)
        metrics = self.val_metrics(pos1, pos2, flows_pred, flow_gt)
        
        i_last = self.hparams[f'{split}_iters'] - 1
        val_results = pl.EvalResult(checkpoint_on=metrics[f'val_epe3d_i#{i_last}'])
        val_results.log('val_loss', loss, sync_dist=True, on_step=False, on_epoch=True, logger=True, prog_bar=False, reduce_fx=torch.mean)
        val_results.log_dict(metrics, on_step=False, on_epoch=True, logger=True, prog_bar=False, reduce_fx=torch.mean)  # No need to sync_dist since metrics are already synced

        return val_results

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step.
        """
        return self._test_val_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        """
        Executes a single test step.
        """
        return self._test_val_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams['optimizer']['lr'])
        lr_scheduler = MultiStepLR(optimizer, milestones=self.hparams['scheduler']['milestones'], gamma=self.hparams['scheduler']['gamma'])

        return {'optimizer': optimizer,
                'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        """
        Returns a train set Dataloader object 
        """
        return self._dataloader(split_type='train')
        
    def val_dataloader(self):
        """
        Returns a validation set Dataloader object 
        """
        return self._dataloader(split_type='val')

    def test_dataloader(self):
        """
        Returns a validation set Dataloader object
        """
        return self._dataloader(split_type='test')

    # def prepare_data(self):
    #     """
    #     Prepare data, executed before loading to GPUs (globally or per node).
    #     """

    def setup(self, stage: str):
        """
        Load datasets
        """
        train_dataset, val_dataset, test_dataset = get_datasets(self.hparams['data'])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def _dataloader(self, split_type: str):
        """
        Arguments:
            split_type : (str) shouild be one of ['train', 'val', 'test']
        Return:
            A DataLoader object of the correspondig split
        """
        
        split_dict = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
            }
        is_train = (split_type == 'train')
        num_workers = get_num_workers(self.hparams['num_workers'])
        if split_dict[split_type] is None:
            loader = None
        else:
            loader = DataLoader(split_dict[split_type],
                                batch_size=self.hparams['batch_size'],
                                num_workers=num_workers,
                                pin_memory=True,
                                shuffle=True if is_train else False)
        return loader

