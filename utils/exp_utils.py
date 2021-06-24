from data.loaders.kitti import KITTI
from data.loaders.flyingthings3d_subset import FlyingThings3DSubset
from data.transforms import transforms


def get_datasets(data_params):
    train_transform = transforms.Augmentation(data_params['data_augmentation']['aug_together'],
                                              data_params['data_augmentation']['aug_pc2'],
                                              data_params['data_process'],
                                              data_params['num_points'])

    test_transform = transforms.ProcessData(data_params['data_process'],
                                            data_params['data_augmentation']['aug_pc2'],
                                            data_params['num_points'],
                                            data_params['allow_less_points'])

    if data_params['train_dataset'] is not None:
        if data_params['train_dataset'] == 'flyingthings3d':
            if data_params['overfit_samples'] is None:
                train_dataset = FlyingThings3DSubset(train=True,
                                                     transform=train_transform,
                                                     num_points=data_params['num_points'],
                                                     data_root=data_params['train_data_root'],
                                                     overfit_samples=data_params['overfit_samples'],
                                                     full=data_params['full'])
            else:
                train_dataset = FlyingThings3DSubset(train=False,
                                                     transform=test_transform,
                                                     num_points=data_params['num_points'],
                                                     data_root=data_params['train_data_root'],
                                                     overfit_samples=data_params['overfit_samples'])

            val_dataset = FlyingThings3DSubset(train=False,
                                               transform=test_transform,
                                               num_points=data_params['num_points'],
                                               data_root=data_params['train_data_root'],
                                               overfit_samples=data_params['overfit_samples'])
        else:
            raise ValueError('Undefined dataset')
    else:
        train_dataset = None
        val_dataset = None

    if data_params['test_dataset'] is not None:
        if data_params['test_dataset'] == 'kitti':
            test_dataset = KITTI(train=False,
                                 transform=test_transform,
                                 num_points=data_params['num_points'],
                                 data_root=data_params['test_data_root'])
        elif data_params['test_dataset'] == 'flyingthings3d':
            test_dataset = FlyingThings3DSubset(False,
                                                transform=test_transform,
                                                num_points=data_params['num_points'],
                                                data_root=data_params['test_data_root'])
        else:
            raise ValueError('Undefined test dataset')
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset
