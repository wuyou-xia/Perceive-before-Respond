import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
from tqdm import tqdm
import datetime
import json
import shutil
from pathlib import Path

import timm
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_srs import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from evaluation import evaluation
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, test_loader, model_without_ddp, res50=None):
    global cnt_step, best, best_epoch
    # train
    model.train()
    if res50 is not None:
        res50.eval()
        
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('intra_loss_ssl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('inter_loss_ssl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('acc_ssl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_emo', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    val_freq = len(data_loader)//config['val_freq']
    print(f"val_freq: {len(data_loader)}//{config['val_freq']}={val_freq}")
    warmup_iterations = warmup_steps*step_size

    for i,(image, aug1, aug2, text, idx, pos, neg) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        cnt_step += 1
        aug1 = aug1.to(device, non_blocking=True)
        aug2 = aug2.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        neg = neg.to(device, non_blocking=True)
        text_input = tokenizer(text, max_length=512, padding='longest', truncation=True, return_tensors="pt").to(device)
        
        ## Compute emotion feat
        emo_feat = None
        if res50 is not None:
            emo_feat = res50(rearrange(image, 'b k c h w -> (b k) c h w'))
            emo_feat = emo_feat.detach()

        loss_ita, loss_itm, intra_loss_ssl, inter_loss_ssl, acc_ssl, loss_emo = model(image, aug1, aug2, text_input, idx=idx, emo_feat=emo_feat, text_pos=pos, text_neg=neg)
        loss = loss_ita + loss_itm + 0.5*intra_loss_ssl + 0.5*inter_loss_ssl + 0.5*loss_emo

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(intra_loss_ssl=intra_loss_ssl.item())
        metric_logger.update(inter_loss_ssl=inter_loss_ssl.item())
        metric_logger.update(acc_ssl=acc_ssl.item())
        metric_logger.update(loss_emo=loss_emo.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch==0 and i%step_size==0 and i<=warmup_iterations:
            scheduler.step(i//step_size)
        if cnt_step%val_freq==0:
            pred_ita_test, pred_itm_test, gt_label_test = eval(model_without_ddp, test_loader, tokenizer, device, config)
            model.train()
            if utils.is_main_process():
                # val_result = evaluation(pred_itm_val, gt_label_test)
                # print(val_result)
                test_result = evaluation(pred_itm_test, gt_label_test)
                print(test_result)

                if args.evaluate:
                    log_stats = {#**{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},
                                'epoch': epoch,
                                }
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")
                else:
                    log_stats = {#**{f'train_{k}': v for k, v in train_stats.items()},
                                #**{f'val_{k}': v for k, v in val_result.items()},
                                **{f'test_{k}': v for k, v in test_result.items()},
                                'epoch': epoch,
                                }
                    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                        f.write(json.dumps(log_stats) + "\n")

                    # save checkpoint
                    save_obj = {
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'best': max(best, test_result['MAP']),
                        }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint.pth'))
                    if test_result['MAP']>best:
                        shutil.copyfile(os.path.join(args.output_dir, 'checkpoint.pth'), os.path.join(args.output_dir, 'checkpoint_best.pth'))
                        best = test_result['MAP']
                        best_epoch = epoch

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def eval(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    pred_ita = []
    pred_itm = []
    gt_label = []
    for image, text, idx in tqdm(data_loader):
        image = image.to(device)
        label = F.one_hot(idx, num_classes=10)
        bs = image.shape[0]
        image = rearrange(image, 'b k c h w -> (b k) c h w')
        #### ita ####
        image_feat = model.visual_encoder(image)  # [bs*10, 65, 768]
        image_atts = torch.ones(image_feat.size()[:-1],dtype=torch.long).to(image.device)
        image_embed = model.vision_proj(image_feat[:,0,:])  # [bs*10, 256]
        image_embed = F.normalize(image_embed,dim=-1)
        # image_feat = rearrange(image_feat, '(b k) p c -> b k p c', k=10)  # [bs, 10, 65, 768]
        image_embed = rearrange(image_embed, '(b k) c-> b k c', k=10)  # [bs, 10, 256]

        text_input = tokenizer(text, padding='longest', truncation=True, 
                               max_length=512, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, 
                                         mode='text')
        text_feat = text_output.last_hidden_state  # [bs, 512, 768]
        text_embed = F.normalize(model.text_proj(text_feat[:,0,:]))  # [bs, 256]
        sim_t2i = text_embed.unsqueeze(1) @ image_embed.permute(0, 2, 1)
        sim_t2i = sim_t2i.squeeze(1)  # [bs, 10] text->image alignment score
        #### itm ####
        repeat_text_embeds = repeat(text_feat, 'b w c->(b k) w c', k=10)
        repeat_attention_mask = repeat(text_input.attention_mask, 'b w->(b k) w', k=10)
        output = model.text_encoder(encoder_embeds = repeat_text_embeds,
                                    attention_mask = repeat_attention_mask,
                                    encoder_hidden_states = image_feat,
                                    encoder_attention_mask = image_atts,
                                    return_dict = True,
                                    mode = 'fusion'
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score = rearrange(score, '(b k)->b k', k=10)
        #### save ####
        score = score.view(-1).cpu().numpy()
        sim_t2i = sim_t2i.view(-1).cpu().numpy()
        label = label.view(-1).cpu().numpy()
        pred_ita.extend(list(sim_t2i))
        pred_itm.extend(list(score))
        gt_label.extend(list(label))
    print(f"len pred_ita: {len(pred_ita)}, len gt_label: {len(gt_label)}")


    if args.distributed:
        dist.barrier()
        # torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return pred_ita, pred_itm, gt_label


def main(args, config):
    global cnt_step, best, best_epoch
    cnt_step = 0

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating sticker response selection dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('srs', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None,None,None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder)

    if args.vision_checkpoint:
        checkpoint = torch.load(args.vision_checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if 'visual_encoder_m' not in key and 'visual_encoder' in key:
                new_state_dict[key.replace('visual_encoder.', '')] = state_dict[key]
        msg = model.visual_encoder.load_state_dict(new_state_dict,strict=False)
        print('load checkpoint from %s'%args.vision_checkpoint)
        print(msg)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict,strict=False)
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    res50 = None
    if args.res_checkpoint:
        res50 = timm.create_model('resnet50', pretrained=False, num_classes=0)
        res50 = res50.to(device)
        # 加载模型
        msg = res50.load_state_dict(torch.load(args.res_checkpoint), strict=False)
        print(msg)

    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        if args.res_checkpoint:
            res50 = torch.nn.parallel.DistributedDataParallel(res50, device_ids=[args.gpu])

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0
    start_epoch = 0
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        best = checkpoint['best']

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config,
                                test_loader, model_without_ddp, res50)

        # pred_ita_val, pred_itm_val, gt_label_val = eval(model_without_ddp, val_loader, tokenizer, device, config)
        pred_ita_test, pred_itm_test, gt_label_test = eval(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            # val_result = evaluation(pred_itm_val, gt_label_test)
            # print(val_result)
            test_result = evaluation(pred_itm_test, gt_label_test)
            print(test_result)

            if args.evaluate:
                log_stats = {#**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             #**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if test_result['MAP']>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = test_result['MAP']
                    best_epoch = epoch

        if args.evaluate:
            break

        lr_scheduler.step(epoch+warmup_steps+1)
        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/SRS.yaml')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--res_checkpoint', default='')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--vision_checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)