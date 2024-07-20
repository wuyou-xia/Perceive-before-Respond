import pickle
import os
import random
import re
from glob import glob
import json
from os.path import join
import numpy as np

from cnsenti import Emotion

import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class srs_dataset(Dataset):
    def __init__(self, anno_root, transform, augment, img_root, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_root = img_root
        self.sticker_candidates = 10
        self.transform = transform
        self.augment = augment
        self.all_stickers = self.loading_stickers()

        if self.mode=='train':
            with open(anno_root, 'rb') as f:
                self.annos = pickle.load(f)
        else:
            annos = []
            cnt = 0
            with open(anno_root, 'r', encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    anno = json.loads(line)
                    # with open(join(img_root, str(anno['current']['sticker_set_id']), 'emoji_mapping.txt'), 'r') as f:
                    #     emoji_txt = f.readlines()
                    # if len(emoji_txt)<self.sticker_candidates:
                    #     cnt += 1
                    #     continue  # 丢弃少于sticker_candidates张图像的sticker set
                    annos.append(anno)
            # print(f'有{cnt}个sample不足{self.sticker_candidates}张图像')
            self.annos = annos
        

    def __len__(self):
        return len(self.annos)


    def __getitem__(self, index):
        anno = self.annos[index]
        ## Read Context
        context_text = []
        pos, neg = 0, 0
        for i in range(len(anno['context'])):
            c = anno['context'][i]
            context_text.append((c['id'], c['text']))
            if self.mode=='train':
                pos, neg = pos+anno['sentiment'][i]['pos'], neg+anno['sentiment'][i]['neg']
        sticker_set_id = anno['current']['sticker_set_id']
        sticker_id = anno['current']['sticker_id']
        ## Construct Example
        text, ori_sticker_pix, temp_negative_sticker_pixs = self.process(context_text, sticker_set_id, sticker_id)
        gray_image = np.ones_like(ori_sticker_pix)*127
        ori_sticker_pix = Image.fromarray(ori_sticker_pix)
        gray_image = Image.fromarray(gray_image)
        temp_negative_sticker_pixs = [Image.fromarray(image) for image in temp_negative_sticker_pixs]
        ## 判断负样本个数
        if len(temp_negative_sticker_pixs)==0:
            temp_negative_sticker_pixs = [gray_image for _ in range(9)]
        elif len(temp_negative_sticker_pixs)<self.sticker_candidates-1:
            padding_num = self.sticker_candidates-1-len(temp_negative_sticker_pixs)
            temp_negative_sticker_pixs = temp_negative_sticker_pixs+[gray_image for _ in range(padding_num)]

        ## transform处理图像
        negative_sticker_pixs = []
        sticker_pix = self.transform(ori_sticker_pix)
        for image in temp_negative_sticker_pixs:
            image = self.transform(image)
            negative_sticker_pixs.append(image.unsqueeze(0))
        ## used for SimCLR
        neg_aug1 = []
        neg_aug2 = []
        pos_aug1 = self.augment(ori_sticker_pix)
        pos_aug2 = self.augment(ori_sticker_pix)
        for image in temp_negative_sticker_pixs:
            image = self.augment(image)
            neg_aug1.append(image.unsqueeze(0))
        for image in temp_negative_sticker_pixs:
            image = self.augment(image)
            neg_aug2.append(image.unsqueeze(0))
        aug1 = torch.cat([pos_aug1.unsqueeze(0)]+neg_aug1, dim=0)
        aug2 = torch.cat([pos_aug2.unsqueeze(0)]+neg_aug2, dim=0)
        

        # index为0时是正确的图像, 即sticker_selection_label=0
        image = torch.cat([sticker_pix.unsqueeze(0)]+negative_sticker_pixs, dim=0)
        idx = 0
        if self.mode=='train':
            return (
                    image,     # [bs, sticker_candidates(10), 3, 128, 128], sticker候选集合, 其中第0张是正确的图像, 后9张是同一个set内按顺序抽取的图像
                    aug1,      # [bs, 3, 128, 128], SimCLR
                    aug2,      # [bs, 3, 128, 128], SimCLR
                    text,      # [bs] str, 当前图像对应的conversation
                    idx,       # [bs], 用于compute loss
                    pos,       # [bs], 对话的positive得分
                    neg,       # [bs], 对话的negative得分
                    )
        else:
            return (
                    image,     # [bs, sticker_candidates(10), 3, 128, 128], sticker候选集合, 其中第0张是正确的图像, 后9张是同一个set内按顺序抽取的图像
                    text,      # [bs] str, 当前图像对应的conversation
                    idx,       # [bs], 用于compute loss
                    )


    def loading_stickers(self):
        pattern = re.compile('stickers/(\d+)')
        all_stickers = {}
        for sset in glob(join(self.img_root, '*')):
            r = pattern.findall(sset)
            set_id = r[0]
            f = open(join(sset, 'emoji_mapping.txt'), encoding='utf8')
            emojis = {}
            for l in f:
                ee = l.strip().split('\t')
                emojis[ee[0]] = ee[1]
            f.close()
            all_stickers[set_id] = list(emojis.keys())

            # img_list = os.listdir(sset)
            # img_list = [x for x in img_list if '.npy' in x]
            # all_stickers[set_id] = img_list

        return all_stickers  # 3516个key, 对应3516个表情包集合, value为每个集合内每张图片对应的emoji


    """Class representing a train/val/test example for text summarization."""
    def process(self, context_text, sticker_set_id, sticker_id):
        # 读取文本
        context_input = ''
        for c in context_text:
            c_text = c[1].strip().replace(' ', '')
            if c_text == '':
                continue
            else:
                context_input = context_input+c_text+'[SEP]'
        # 读取图像
        sticker_pix = np.load(join(self.img_root, str(sticker_set_id), str(sticker_id)+'.npy'))
        sticker_id = sticker_id
        sticker_set_id = sticker_set_id
        # 构建图像负样本
        negative_sticker_ids = self.all_stickers[str(sticker_set_id)][:]  # 负样本来自同一个表情包图像集合
        if str(sticker_id)+'.npy' in negative_sticker_ids:  # emoji_mapping.txt可能没有正样本图像名
            negative_sticker_ids.remove(str(sticker_id)+'.npy')
        negative_sticker_ids = negative_sticker_ids[:self.sticker_candidates-1]
        # if 0<len(negative_sticker_ids)<self.sticker_candidates-1:
        #     negative_sticker_ids = np.random.choice(negative_sticker_ids, self.sticker_candidates-1)
        # elif len(negative_sticker_ids)>=self.sticker_candidates-1:
        #     negative_sticker_ids = np.random.choice(negative_sticker_ids, self.sticker_candidates-1, replace=False)
        # len等于0的情况交给getitem处理
        negative_sticker_pixs = [
            np.load(join(self.img_root, str(sticker_set_id), str(i)+'.npy')) for i in negative_sticker_ids
        ]
        return context_input, sticker_pix, negative_sticker_pixs

if __name__=='__main__':
    import torch
    import dataset.utils as utils
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.3, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    config = {}
    config['train_file'] = '/home/ubuntu/lsz/MM2022_PAMI/CH-ALBEF/data/release_train_senti.pkl'
    config['image_root'] = '/home/ubuntu/lsz/MM2022_PAMI/stickerchat/npy_stickers'
    train_dataset = srs_dataset(config['train_file'], train_transform, augment, config['image_root'], mode='train')
    train_loader = DataLoader(
                            train_dataset,
                            batch_size=128,
                            num_workers=16,
                            pin_memory=False,
                            shuffle=True,
                        )
    for (image, aug1, aug2, text, idx, pos, neg) in tqdm(train_loader):
        pass