import argparse
import os
import pprint

import logging
import timeit
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import torch.optim as optim
import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/root/autodl-tmp/DSNet/configs/cityscapes/dsnet_cityscapes.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument("--local_rank", type=int, default=-1)       

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()
    # torch.autograd.set_detect_anomaly(True)
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)        

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'dsnet_m')

    logger.info(pprint.pformat(args))
    logger.info(config)
    print(tb_log_dir)
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED


    gpus = list(config.GPUS)


    print(gpus)
    if torch.cuda.device_count() != len(gpus):
        print(len(gpus))
        print(torch.cuda.device_count())
        print("The gpu numbers do not match!")


    distributed = args.local_rank >= 0
    if distributed:
        print("---------------devices:", args.local_rank)
        device = torch.device('cuda:{}'.format(args.local_rank))    
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )  
        # return 0

#     imgnet = 'imagenet' in config.MODEL.PRETRAINED
    model = models.dsnet.get_seg_model(config, imgnet_pretrained=True)
    # print(model)
    # model = models.dsnet.get_pred_model("", num_classes=config.DATASET.NUM_CLASSES)

    print(final_output_dir)
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        if os.path.exists(models_dst_dir):
            shutil.rmtree(models_dst_dir)
        shutil.copytree(os.path.join(this_dir, '../models'), models_dst_dir)
        print(os.path.join(this_dir, '../models'))

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)   

    # batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)
    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TRAIN_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TRAIN.MULTI_SCALE,
                        flip=config.TRAIN.FLIP,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TRAIN.BASE_SIZE,
                        crop_size=crop_size,
                        scale_factor=config.TRAIN.SCALE_FACTOR)

    train_sampler = get_sampler(train_dataset)

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE and train_sampler is None,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True,
        sampler = train_sampler)


    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=config.TEST.MULTI_SCALE,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size)
    test_sampler = get_sampler(test_dataset)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True,
        sampler=test_sampler)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                        thres=config.LOSS.OHEMTHRES,
                                        min_kept=config.LOSS.OHEMKEEP,
                                        weight=train_dataset.class_weights)
        ce_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                    weight=train_dataset.class_weights)


    model = FullModel(model, sem_criterion, ce_criterion)

    if distributed:
        model = model.to(device)
#         if SyncBatchNorm
        # 使用 SyncBatchNorm 
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # 将模型封装为 DistributedDataParallel
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
    )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                lr=config.TRAIN.LR,
                                momentum=config.TRAIN.MOMENTUM,
                                weight_decay=config.TRAIN.WD,
                                nesterov=config.TRAIN.NESTEROV,
                                )
    else:
        
        raise ValueError('Only Support SGD optimizer')
    # 定义Adam优化器，并传入模型参数和学习率
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, eps=1e-8)


    epoch_iters = np.int64(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
    


    best_mIoU = 0
    mean_IoU = 0
    last_epoch = config.TRAIN.BEGIN_EPOCH
    valid_loss = 0
    flag_rm = config.TRAIN.RESUME
    
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(config.MODEL.PRETRAINED)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']
            
            model.module.model.load_state_dict({k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    # last_epoch = 0
    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # real_end = 120+1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    real_end = end_epoch
    base_lr = config.TRAIN.LR
    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)



        train(config, epoch, config.TRAIN.END_EPOCH, 
                  epoch_iters, base_lr, num_iters,
                  trainloader, optimizer, model, writer_dict)

        if flag_rm == 1 or (epoch % 2 == 0 and epoch <= 150)  or (epoch % 20 == 0 and epoch > 50 and epoch <= 380)  or (epoch>380 and epoch % 2 == 0) and (epoch>420): 
            valid_loss, mean_IoU, IoU_array = validate(config, 
                        testloader, model, writer_dict)

        if flag_rm == 1:
            flag_rm = 0
        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint_dhs_base_bdd.pth.tar'))
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir,'checkpoint_dhs_base_bdd.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dhs_base_bdd.pth'))
                torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dhs_base_bdd.pt'))
                if best_mIoU > 0.620:
                    torch.save(model.module.state_dict(),
                        os.path.join(final_output_dir, 'best_dhs_base_bdd_{: .6f}.pth'.format(best_mIoU)))

                    torch.save({
                    'epoch': epoch+1,
                    'best_mIoU': best_mIoU,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(final_output_dir,'best_dhs_base_bdd_{: .6f}.pth.tar'.format(best_mIoU)))
                else :
                    torch.save({
                    'epoch': epoch+1,
                    'best_mIoU': best_mIoU,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, os.path.join(final_output_dir,'best_dhs_base_bdd.pth.tar'))
            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                        valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)




    if args.local_rank <= 0:

        torch.save(model.module.state_dict(),
                os.path.join(final_output_dir, 'final_dhs_base_bdd.pt'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % int((end-start)/3600))
        logger.info('Done')

if __name__ == '__main__':
    main()
