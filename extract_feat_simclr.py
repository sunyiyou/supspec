#!/usr/bin/env python
import argparse
import builtins
import os
import shutil
import copy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.models as models

import sys
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from ylib.ytool import cluster_acc

from train_linear import get_linear_acc

cudnn.benchmark = True

sys.path.extend(['..', '.'])
from datasets.dataset_tinyimagenet import load_train, load_val_loader, num_classes_dict
from tools.store import ExperimentLogWriter
import models.builder as model_builder
import utils

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
model_names += ['resnet18_cifar_variant1']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', choices=['cifar100'], default='cifar100',
                    help='Which dataset to evaluate on.')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_cifar_variant1')
parser.add_argument('-n', '--name', default='resnet18_simclr_ncd')
parser.add_argument('--dir', type=str, default='/home/sunyiyou/workspace/orca/results/simclr-ncd-cifar100/snapshot',
                    help='Directory where all of the runs are located.')
parser.add_argument('--specific_ckpts', default='199.pth', nargs='*', help='filenames of specific checkpoints to evaluate')
# parser.add_argument('-n', '--name', default='resnet18_simclr')
# parser.add_argument('--dir', type=str, default='/home/sunyiyou/workspace/orca/pretrained/',
#                     help='Directory where all of the runs are located.')
# parser.add_argument('--specific_ckpts', default='simclr_cifar_100.pth.tar', nargs='*', help='filenames of specific checkpoints to evaluate')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--opt', type=str, default='sgd', choices=['sgd', 'adam'],
                    help='Directory where all of the runs are located.')

parser.add_argument('--num_per_class', type=int, default=int(1e10),
                    help='Number of images per class for getting a subset of Imagenet')
parser.add_argument('--val_every', type=int, default=5, help='How often to evaluate lincls')
parser.add_argument('--latest_only', action='store_true', help='if set, only evaluate the latest_ checkpoints')
parser.add_argument('--mpd', action='store_true', help='short hand for multi-gpu training')
parser.add_argument('--dist_url_add', default=0, type=int, help='to avoid collisions of tcp')
parser.add_argument('--use_random_labels', action='store_true', help='whether to evaluate using the random labels')
parser.add_argument('--normalize', action='store_true', help='whether to evaluate using the random labels')
parser.add_argument('--nomlp', default=True, type=str, help='only use the backbone without the last mlp layer')
parser.add_argument('--aug', type=str, default='standard', choices=['standard', 'mocov2', 'not-standard'],
                    help='Directory where all of the runs are located.')

best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.mpd:
        args.multiprocessing_distributed = True
        args.world_size = 1
        args.rank = 0
        args.dist_url = 'tcp://127.0.0.1:' + str(10001 + args.dist_url_add)
    utils.spawn_processes(main_worker, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        utils.init_proc_group(args, ngpus_per_node)
    
    logger = ExperimentLogWriter(args.dir)

    # loop through checkpoints and set pre-trained
    ckpt_dir = os.path.join(args.dir)
    for fname in sorted(os.listdir(ckpt_dir)):
        if args.specific_ckpts is not None and fname not in args.specific_ckpts: continue
        args.pretrained = os.path.join(ckpt_dir, fname)
        lineval_dir = os.path.join(args.dir, 'lin_eval_ckpt')
        if os.path.exists(lineval_dir):
            print('linear evaluation dir exists at {}, may overwrite...'.format(lineval_dir))
        eval_ckpt(
            copy.deepcopy(args), # because args.batch_size and args.workers are changed
            ngpus_per_node,
            fname,
            logger)



def eval_ckpt(args, ngpus_per_node, ptrain_fname, logger):
    # create model
    pretrained_id = ptrain_fname.split('.')[0]
    dict_id = pretrained_id + '_lineval'
    dict_id += '_{}_lr:{}_wd:{}_{}eps'.format(args.opt, args.lr, args.weight_decay, args.epochs)
    if args.nomlp:
        dict_id = dict_id + '_nomlp'
    dict_id += '_aug:' + args.aug
    dict_id += '_random_labels' if args.use_random_labels else ''
    ckpt_dir = os.path.join(args.dir, 'lin_eval_ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ptrain_fname += '_random_labels' if args.use_random_labels else ''
    lin_eval_loc = os.path.join(ckpt_dir, ptrain_fname)

    logger.create_data_dict(
        ['epoch', 'train_acc', 'val_acc','train_loss', 'val_loss', 'train5', 'val5'],
        dict_id=dict_id)

    from models.backbones.resnet_s import resnet18
    model = resnet18(num_classes=100)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        param.requires_grad = False

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            state_dict = torch.load(args.pretrained, map_location='cpu')
            if args.specific_ckpts not in {'simclr_cifar_100.pth.tar'}:
                state_dict = state_dict['state_dict']
                state_dict = {k.replace("model.", ""):v for k,v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # model = utils.init_data_parallel(args, model, ngpus_per_node)
    model = model.cuda()

    from train_linear import LinearClassifier
    model_linear = LinearClassifier(512, 100).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model_linear.parameters()))
    assert len(parameters) == 2

    if args.opt=='sgd':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.opt=='adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)


    # Data loading code
    if args.use_random_labels:
        random_labels = torch.load(os.path.join(args.dir, 'saved_tensors', 'random_labels.pth')).numpy()
    else:
        random_labels = None
    train_sampler, train_loader = load_train(args.dataset, args.num_per_class, args.distributed,
                                             args.batch_size, args.workers, data_aug=args.aug, random_labels=random_labels)
    val_loader = load_val_loader(args.dataset, args.batch_size, args.workers)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    feat_log_val, label_log_val = extract('val', val_loader, model, args, FORCE_RUN=False)
    feat_log_train, label_log_train = extract('train', train_loader, model, args, FORCE_RUN=False)


    # get_linear_acc(feat_log_train, label_log_train, feat_log_val, label_log_val, 100)  #61.34



    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

    # # 100 classes
    # print("100 classes")
    # alg = KMeans(init="k-means++", n_clusters=100, n_init=5, random_state=0)
    # if True:
    #     estimator = alg.fit(feat_log_train)
    #     # ltrain_novel_pred = estimator.labels_
    #     feat_log_train_pred = estimator.predict(feat_log_train)
    #     cluster_acc(feat_log_train_pred, label_log_train.astype(np.int64), print_ret=True)
    #
    #     ltest_novel_pred = alg.predict(feat_log_val)
    #     cluster_acc(ltest_novel_pred, label_log_val.astype(np.int64), print_ret=True)

    # 50 classes
    print("50 classes")
    ftrain = normalizer(feat_log_train[label_log_train >= 50])
    ltrain = label_log_train[label_log_train >= 50]
    ftest = normalizer(feat_log_val[label_log_val >= 50])
    ltest = label_log_val[label_log_val >= 50]
    alg = KMeans(init="k-means++", n_clusters=50, n_init=5, random_state=0)
    if True:
        estimator = alg.fit(ftrain)
        # ltrain_novel_pred = estimator.labels_
        feat_log_train_pred = estimator.predict(ftrain)
        cluster_acc(feat_log_train_pred, ltrain.astype(np.int64), print_ret=True)

        ltest_novel_pred = alg.predict(ftest)
        cluster_acc(ltest_novel_pred, ltest.astype(np.int64), print_ret=True)

    1/0



    from ylib.ytool import ArrayDataset
    trainset = ArrayDataset(feat_log_train, labels=label_log_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    valset = ArrayDataset(feat_log_val, labels=label_log_val)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)


    best_acc1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        top1, top5, losses = train(train_loader, model_linear, criterion, optimizer, epoch, args)

        # always test after 1 epoch of linear evaluation
        if epoch == 0 or (epoch + 1) % args.val_every == 0:
            # evaluate on validation set
            acc1, acc5, val_losses = validate(val_loader, model_linear, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                logger.update_data_dict(
                    {
                    'epoch' : int(epoch) + 1,
                    'train_acc' : top1.item(), 
                    'val_acc' : acc1.item(),
                    'train_loss' : losses,
                    'val_loss' : val_losses,
                    'train5' : top5.item(),
                    'val5' : acc5.item()
                    }, dict_id=dict_id)
                logger.save_data_dict(dict_id=dict_id)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_linear.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, filename=lin_eval_loc, best_file=os.path.join(ckpt_dir, 'model_linear_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True).float()
            target = target.cuda(args.gpu, non_blocking=True).long()
        else:
            images = images.cuda(non_blocking=True).float()
            target = target.cuda(non_blocking=True).long()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % args.print_freq == 0:
        #     progress.display(i)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(
        len(val_loader),
        [losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True).float()
            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # if i % args.print_freq == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # is the above todo done??
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def extract(split, loader, model, args, FORCE_RUN=False):
    model.eval()

    cache_name = f"cache/{args.dataset}_{split}_{args.name}_in.npy"
    if FORCE_RUN or not os.path.exists(cache_name):
        feat_log = np.zeros((len(loader.dataset), 512))
        label_log = np.zeros(len(loader.dataset))
        batch_size = args.batch_size
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(loader.dataset))

            out = model.features(inputs)
            # score = net(inputs)
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            print(batch_idx)
        np.save(cache_name, (feat_log.T, label_log))
    else:
        feat_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log = feat_log.T

    return feat_log, label_log.astype(np.int64)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
