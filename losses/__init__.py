from .supervised_losses import SupervisedL1Loss, SupervisedL2Loss, SupervisedL1RegLoss
from .unsupervised_losses import UnSupervisedL1Loss


losses_dict = {
    'sv_l1': SupervisedL1Loss,
    'sv_l2': SupervisedL2Loss,
    'sv_l1_reg': SupervisedL1RegLoss,
    'unsup_l1': UnSupervisedL1Loss
               }
