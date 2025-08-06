from flemme.dataset import ImgSegDataset
from flemme.utils import DataForm
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose
from flemme.augment import get_transforms, select_label_transforms
from flemme.logger import get_logger
import os

logger = get_logger('auxtar_dataset_io')
def create_auxtar_dataset(auxtar_dataset_config):
    aux_path = auxtar_dataset_config.get('auxiliary_path')
    aux_names = auxtar_dataset_config.get('auxiliary_dataset_names')
    aux_path_list = [os.path.join(aux_path, name) for name in aux_names]
    tar_path_list = auxtar_dataset_config.get('target_path_list')
    dataset_config = auxtar_dataset_config.get('dataset')
    aux_datasets = []
    tar_datasets = []

    ### read transforms
    trans_config_list = auxtar_dataset_config.pop('data_transforms', [])
    data_transforms = get_transforms(trans_config_list, DataForm.IMG, img_dim=3)
    data_transforms = Compose(data_transforms)
    dataset_config['data_transform'] = data_transforms
    ### generate label transform config
    label_trans_config_list = auxtar_dataset_config.pop('label_transforms', None)
    logger.warning('Note that all transforms that would deform the image or change the number or order of points should be applied in front of other transforms that introduce random operations.')
    if label_trans_config_list is None:
        logger.info('There is no specified transforms for labels, we would generate the necessary transforms for labels automatically.')
        label_trans_config_list = \
            select_label_transforms(trans_config_list, DataForm.IMG)
    
    label_transforms = get_transforms(label_trans_config_list, DataForm.IMG, img_dim=3)
    label_transforms = Compose(label_transforms)
    dataset_config['label_transform'] = label_transforms

    for ap in aux_path_list:
        dataset_config['data_path'] = ap
        aux_datasets.append(ImgSegDataset(**dataset_config))

    for tp in tar_path_list:
        dataset_config['data_path'] = tp
        tar_datasets.append(ImgSegDataset(**dataset_config))

    return aux_datasets, tar_datasets

def create_auxtar_loader(datasets, loader_config):
    ### return loader
    if not type(datasets) == list:
        datasets = [datasets] 
    logger.info(f"Createing loader from following {len(datasets)} datasets: ")
    for d in datasets:
        logger.info(f'dataset: {d.data_path}')
    # logger.info(f'loader info: {loader_config}')
    loader = DataLoader(ConcatDataset(datasets), **loader_config)
    return loader


if __name__ == '__main__':
    from flemme.utils import load_config
    config = load_config()
    loader_config = config['loader']
    auxtar_dataset_config = config['auxtar_dataset']
    aux_datasets, tar_datasets = create_auxtar_dataset(auxtar_dataset_config)
    selected_idx = 0
    loader = create_auxtar_loader([aux_datasets[selected_idx],tar_datasets], loader_config)
    print(aux_datasets, tar_datasets)
    print(loader)
