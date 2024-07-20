import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode

import dataset.utils as utils
from dataset.srs_dataset import srs_dataset
from dataset.randaugment import RandomAugment

def create_dataset(dataset, config):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset=='srs':
        train_dataset = srs_dataset(config['train_file'], train_transform, augment, config['image_root'], mode='train')
        val_dataset = srs_dataset(config['val_file'], test_transform, augment, config['image_root'], mode='eval')
        test_dataset = srs_dataset(config['test_file'], test_transform, augment, config['image_root'], mode='eval')
        return train_dataset, val_dataset, test_dataset

    else:
        raise NotImplementedError


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
