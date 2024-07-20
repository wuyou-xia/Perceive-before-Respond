import os

import utils
import numpy as np
from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder = None,
                 config = None,
                 ):
        super().__init__()
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        ssl_mlp_dim = config['ssl_mlp_dim']
        ssl_emb_dim = config['ssl_emb_dim']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.is_ssl_loss = False
        if config['is_ssl']!=0:
            self.is_ssl_loss = True
            self.image_mlp = self._build_mlp(in_dim=vision_width, mlp_dim=ssl_mlp_dim, out_dim=ssl_emb_dim)
            self.intra_ssl_loss = Intra_SIMCLRLOSS(temperature=config['ssl_temp'])
            self.inter_ssl_loss = Inter_SIMCLRLoss(temperature=config['ssl_temp'])

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.itm_head = nn.Linear(text_width, 2)

        self.is_emo_loss = False
        if config['cluster_num']>0:
            self.cluster_num = config['cluster_num']
            print(f'==>load {self.cluster_num} emotion cluster')
            self.is_emo_loss = True
            self.emo_proj = nn.Linear(vision_width, 7*self.cluster_num)
            self.register_buffer('emotion_cluster', torch.from_numpy(np.load(config['emotion_cluster'])))


    def forward(self, image, aug1, aug2, text, idx, emo_feat=None, text_pos=None, text_neg=None):
        #### image-text alignment ####
        image = rearrange(image, 'b k c h w -> (b k) c h w')
        image_embeds = self.visual_encoder(image)
        pos_image_embeds = rearrange(image_embeds, '(b k) p c -> b k p c', k=10)[:, 0]
        
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        pos_image_atts = torch.ones(pos_image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        image_feat = rearrange(image_feat, '(b k) c-> b k c', k=10)
        pos_image_feat = image_feat[:, 0, :]  # get the grount truth image
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                        return_dict = True, mode = 'text')
        text_embeds = text_output.last_hidden_state  # [bs, seq, 768]
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)  # [bs, 256]

        sim_t2i_targets = F.one_hot(idx, num_classes=10).to(image.device)
        sim_i2t_targets = torch.eye(pos_image_feat.shape[0]).to(image.device)

        sim_i2t = pos_image_feat @ text_feat.t() / self.temp  # [bs, bs]
        sim_t2i = text_feat.unsqueeze(1) @ image_feat.permute(0, 2, 1) / self.temp  # [bs, 10]
        sim_t2i = sim_t2i.squeeze(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i)/2

        #### SimCLR ####
        intra_loss_ssl, inter_loss_ssl, acc_ssl = 0, 0, 0
        if self.is_ssl_loss:
            aug1 = rearrange(aug1, 'b k c h w -> (b k) c h w')
            aug2 = rearrange(aug2, 'b k c h w -> (b k) c h w')
            aug1_embed = self.image_mlp(self.visual_encoder(aug1)[:,0,:])  # [bs*10, 256]
            aug2_embed = self.image_mlp(self.visual_encoder(aug2)[:,0,:])  # [bs*10, 256]
            inter_ssl = self.inter_ssl_loss(aug1_embed, aug2_embed)
            intra_loss_ssl = self.intra_ssl_loss(aug1_embed, aug2_embed)
            inter_loss_ssl, acc_ssl = inter_ssl['loss'], inter_ssl['acc']

        #### Emotion Anchor ####
        loss_emo = 0
        if self.is_emo_loss:
            emo_target = emo_feat @ self.emotion_cluster.t()
            pred_emo = self.emo_proj(image_embeds[:,0,:])
            loss_emo = F.kl_div(F.log_softmax(pred_emo, dim=1), F.softmax(emo_target, dim=1), reduction='batchmean')

        #### image-text matching ####
        output_pos = self.text_encoder(encoder_embeds = text_embeds,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = pos_image_embeds,
                                        encoder_attention_mask = pos_image_atts,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )
        #### Hard Example Mining
        text_senti = torch.cat([text_pos.unsqueeze(1), text_neg.unsqueeze(1)], dim=1)
        text_senti = text_senti/(text_senti.sum(dim=1)+1e-4).unsqueeze(1)  # [bs, 2]
        # emotions = {'0':'Surprise',
        #         '1':'Happiness',
        #         '2':'Disgust',
        #         '3':'Fear',
        #         '4':'Sadness',
        #         '5':'Anger',
        #         '6':'Neutral'
        #         }
        image_senti = rearrange(pred_emo, 'b (e k)->b e k', k=self.cluster_num).mean(dim=2)  # [bs*10, 7], mining02:emo_target, mining03:
        image_senti = torch.cat([image_senti[:,:2].max(dim=-1).values.unsqueeze(1), image_senti[:,2:6].max(dim=-1).values.unsqueeze(1)], dim=1).softmax(dim=1)  # [bs*10, 2]
        # image_senti = rearrange(image_senti, '(b k) s->b k s', k=10)
        pos_image_senti = rearrange(image_senti, '(b k) s->b k s', k=10)[:,0]
        
        senti_t2i = torch.abs(text_senti.unsqueeze(1)-image_senti.unsqueeze(0)).sum(dim=-1)
        senti_t2i = (2-senti_t2i+1e-4).softmax(dim=1)  # [bs, bs*10], 将极性差值调整为概率, 极性差距越大被选择的概率越小. 2表示pos和neg极性差值最大=2
        senti_i2t = torch.abs(pos_image_senti.unsqueeze(1)-text_senti.unsqueeze(0)).sum(dim=-1)  # [bs, bs]
        senti_i2t = (2-senti_i2t+1e-4).softmax(dim=1)  # [bs, bs]

        with torch.no_grad():
            bs = pos_image_embeds.size(0)
            image_feat = rearrange(image_feat, 'b k c->(b k) c', k=10)
            sim_t2i_global = text_feat @ image_feat.permute(1, 0)
            sim_t2i_global_targets = torch.zeros([bs, bs*10]).to(image.device)
            for i in range(bs): sim_t2i_global_targets[i, i*10] = 1

            weights_i2t = F.softmax(sim_i2t[:,:]+senti_i2t[:,:]+1e-4,dim=1)
            weights_t2i = F.softmax(sim_t2i_global[:,:]+senti_t2i[:,:]+1e-4,dim=1)

            mask_i2t = sim_i2t_targets.bool()
            weights_i2t.masked_fill_(mask_i2t, 0)
            mask_t2i = sim_t2i_global_targets.bool()
            weights_t2i.masked_fill_(mask_t2i, 0)

        # select a negative image for each text
        image_embeds_neg = []
        # image_embeds = rearrange(image_embeds, '(b k) p c -> b k p c', k=10)
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()  # 按照weights的权重采样1个元素, value=0处不会被选中(即Ground truth)
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)  # [bs, 65, 768]

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)  # [bs, 512, 768]
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      # [bs, 512]

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)  # [2*bs, 512, 768]
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)  # # [2*bs, 512]

        image_embeds_all = torch.cat([image_embeds_neg, pos_image_embeds],dim=0)  # [2*bs, 65, 768]
        image_atts_all = torch.cat([pos_image_atts, pos_image_atts],dim=0)  # [2*bs, 65]

        output_neg = self.text_encoder(encoder_embeds = text_embeds_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)  # [3*bs, 768]
        vl_output = self.itm_head(vl_embeddings)  # [3*bs, 2]

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)  # [3*bs], 前bs=1, 后2*bs=0
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm, intra_loss_ssl, inter_loss_ssl, acc_ssl, loss_emo

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        if not utils.is_dist_avail_and_initialized():
            return nn.Sequential(OrderedDict([
                ("layer1", nn.Linear(in_dim, mlp_dim)),
                ("bn1", nn.BatchNorm1d(mlp_dim)),
                ("relu1", nn.ReLU(inplace=True)),
                # ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                # ("bn2", nn.BatchNorm1d(mlp_dim)),
                # ("relu2", nn.ReLU(inplace=True)),
                ("layer3", nn.Linear(mlp_dim, out_dim)),
            ]))
        else:
            return nn.Sequential(OrderedDict([
                ("layer1", nn.Linear(in_dim, mlp_dim)),
                ("bn1", nn.SyncBatchNorm(mlp_dim)),
                ("relu1", nn.ReLU(inplace=True)),
                # ("layer2", nn.Linear(mlp_dim, mlp_dim)),
                # ("bn2", nn.SyncBatchNorm(mlp_dim)),
                # ("relu2", nn.ReLU(inplace=True)),
                ("layer3", nn.Linear(mlp_dim, out_dim)),
            ]))


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class Intra_SIMCLRLOSS(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, q_a, q_b):

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)
        q_a = rearrange(q_a, '(b k) c-> b k c', k=10)
        q_b = rearrange(q_b, '(b k) c-> b k c', k=10)

        local_batch_size = q_a.size(0)
        candidate_size = q_a.size(1)

        # 由于仅在同一个topic下计算负样本，因此不需要gather全部GPU的tensor
        # k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = torch.arange(candidate_size, device=q_a.device)
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, candidate_size) * 1e9
            self.labels = repeat(self.labels, 'k -> (b k)', b=local_batch_size)
            self.masks = self.masks.unsqueeze(0)
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, q_a.transpose(1, 2)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, q_b.transpose(1, 2)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, q_b.transpose(1, 2)) / self.tau
        logits_ba = torch.matmul(q_b, q_a.transpose(1, 2)) / self.tau
        logits_aa = rearrange(logits_aa, 'b k c-> (b k) c', k=10)
        logits_bb = rearrange(logits_bb, 'b k c-> (b k) c', k=10)
        logits_ab = rearrange(logits_ab, 'b k c-> (b k) c', k=10)
        logits_ba = rearrange(logits_ba, 'b k c-> (b k) c', k=10)

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
        
        return loss


class Inter_SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, q_a, q_b):

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)
        # q_a = rearrange(q_a, '(b k) c-> b k c', k=10)
        # q_b = rearrange(q_b, '(b k) c-> b k c', k=10)

        local_batch_size = q_a.size(0)
        # candidate_size = q_a.size(1)

        # 由于仅在同一个topic下计算负样本，因此不需要gather全部GPU的tensor
        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            #----used for topic ssl----#
            # self.labels = torch.arange(candidate_size, device=q_a.device)
            # self.labels = repeat(self.labels, 'k -> (b k)', b=local_batch_size)
            # self.masks = F.one_hot(self.labels, candidate_size) * 1e9
            # self.masks = self.masks.unsqueeze(0)
            #----used for total ssl----#
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau
        #----used for topic ssl----#
        # logits_aa = rearrange(logits_aa, 'b k c-> (b k) c', k=10)
        # logits_bb = rearrange(logits_bb, 'b k c-> (b k) c', k=10)
        # logits_ab = rearrange(logits_ab, 'b k c-> (b k) c', k=10)
        # logits_ba = rearrange(logits_ba, 'b k c-> (b k) c', k=10)

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
        
        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'acc': acc}

        
if __name__=='__main__':
    import argparse
    from tqdm import tqdm
    import ruamel.yaml as yaml
    from models.tokenization_bert import BertTokenizer
    from dataset import create_dataset, create_sampler, create_loader

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/SRS.yaml')
    parser.add_argument('--output_dir', default='output/SRS')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    train_dataset, val_dataset, test_dataset = create_dataset('srs', config)
    samplers = [None, None, None]
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[4]+[8]+[8],
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None,None,None])
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    model = ALBEF(config=config, text_encoder=args.text_encoder)
    # model = model.cuda()
    max_word = 0
    for (image, aug1, aug2, text, idx) in tqdm(train_loader):
        # image, idx = image.cuda(), idx.cuda()
        # aug1, aug2 = aug1.cuda(), aug2.cuda()
        text = tokenizer(text, max_length=512, padding='longest', truncation=True, return_tensors="pt")
        text = text.to(image.device)
        loss_ita, loss_itm = model(image, aug1, aug2, text, idx)
        break