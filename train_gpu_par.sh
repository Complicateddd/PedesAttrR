CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=1 --master_port=1233 PARTrain.py --cfg ./configs/pedes_baseline/parbaseline.yaml