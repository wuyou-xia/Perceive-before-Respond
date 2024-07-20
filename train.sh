python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
--config ./configs/SRS.yaml \
--output_dir output/ \
--res_checkpoint weights/128_res50/model_58_9149_6456.pth \
--vision_checkpoint weights/ALBEF.pth