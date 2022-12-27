import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from itertools import cycle
import numpy as np
import argparse
from arguments import set_deterministic, Namespace, csv, shutil, yaml
from augmentations import get_aug
from models import get_model
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from datetime import date
from utils import plot_cluster
from sklearn.cluster import KMeans, DBSCAN
from ylib.ytool import cluster_acc

from train_linear import get_linear_acc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main(log_writer, log_file, device, args):
    iter_count = 0


    import open_world_cifar as datasets
    dataroot = args.data_dir
    if args.dataset.name == 'cifar10':
        train_label_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs))
        train_unlabel_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        train_eval_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.num_classes = 10
    elif args.dataset.name == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        train_eval_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_label_set.unlabeled_idxs)
        args.num_classes = 100


    labeled_len = len(train_label_set)
    unlabeled_len = len(train_unlabel_set)
    labeled_batch_size = int(args.train.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_label_set, batch_size=labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_unlabel_set, batch_size=args.train.batch_size - labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_set, batch_size=100, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    # define model
    model = get_model(args.model, args).to(device)
    # model = torch.nn.DataParallel(model)

    # define optimizer
    optimizer = get_optimizer(
        args.train.optimizer.name, model, 
        lr=args.train.base_lr*args.train.batch_size/256, 
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/256, 
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/256, args.train.final_lr*args.train.batch_size/256, 
        len(train_label_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(0, args.train.stop_at_epoch):
        #######################  Train #######################
        model.train()
        loss_list = []
        print("number of iters this epoch: {}".format(len(train_label_loader)))
        unlabel_loader_iter = cycle(train_unlabel_loader)
        for idx, ((x1, x2), target) in enumerate(train_label_loader):
            ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            x1, x2, ux1, ux2, target, target_unlabeled = x1.to(device), x2.to(device), ux1.to(device), ux2.to(device), target.to(device), target_unlabeled.to(device)

            iter_count += 1
            model.zero_grad()
            data_dict = model.forward_ncd(x1, x2, ux1, ux2, target)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            loss_list.append(loss.item())

            if (idx + 1) % args.print_freq == 0:
                # print('Train: [{0}][{1}/{2}]\t Loss_all {3}'.format(
                #     epoch, idx + 1, len(train_label_loader), loss.item()
                # ))
                loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"]["loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), data_dict["d_dict"]["loss5"].item()
                print('Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} c1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}\t'.format(
                    epoch, idx + 1, len(train_label_loader), loss.item(), loss1, loss2, loss3, loss4, loss5
                ))


        #######################  Evaluation #######################
        def feat_extract(loader):
            targets = np.array([])
            features = []
            for idx, (x1, labels) in enumerate(loader):
                feat = model.backbone.features(x1.to(device, non_blocking=True))
                targets = np.append(targets, labels.cpu().numpy())
                features.append(feat.data.cpu().numpy())

            return np.concatenate(features), targets.astype(int)


        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            features_train, targets_train = feat_extract(train_eval_loader)
            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

            #######################  K-Means #######################
            ftrain = normalizer(features_train[targets_train >= 50])
            ltrain = targets_train[targets_train >= 50]
            alg = KMeans(init="k-means++", n_clusters=50, n_init=5, random_state=0)
            estimator = alg.fit(ftrain)
            feat_log_train_pred = estimator.predict(ftrain)
            kmeans_acc = cluster_acc(feat_log_train_pred, ltrain.astype(np.int64))

            #######################  Linear Probe #######################
            features_test, targets_test = feat_extract(test_loader)   # originally it is 100 classes
            features_test = normalizer(features_test[targets_test >= 50])
            targets_test = targets_test[targets_test >= 50]
            lp_acc, _ = get_linear_acc(features_train, targets_train, features_test, targets_test, 50, print_ret=False)  # 61.34

            write_dict = {
                'epoch': epoch,
                'lr': lr_scheduler.get_lr(),
                'kmeans_acc': kmeans_acc,
                'lp_acc': lp_acc,
            }
            print(f"K-Means Acc: {kmeans_acc:.4f}\t linear Probe Acc: {lp_acc:.4f}")
            log_writer.writerow(write_dict)
            log_file.flush()

        #######################  Visualization #######################
        if (epoch + 1) % args.vis_freq == 0:
            features, targets = feat_extract(train_eval_loader)
            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
            sampling_size = int(0.15 * len(features))
            rand_idx = np.random.choice(range(len(features)), sampling_size, replace=False)
            fvis = normalizer(features)[rand_idx]
            tvis = targets[rand_idx].astype(int)

            os.makedirs(os.path.join(args.log_dir, 'vis'), exist_ok=True)
            plot_cluster(fvis, tvis,
                         figsize=(9, 9),
                         sampling_ratio=1,
                         path=os.path.join(args.log_dir, 'vis', f"cluster_{epoch}.jpg"),
                         colors=None,
                         )

        #######################  Save Epoch #######################
        if (epoch + 1) % args.log_freq == 0:
            model_path = os.path.join(ckpt_dir, f"{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict()
            }, model_path)
            print(f"Model saved to {model_path}")

    # Save checkpoint
    model_path = os.path.join(ckpt_dir, f"latest_{epoch+1}.pth")
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    with open(os.path.join(args.log_dir, "checkpoints", f"checkpoint_path.txt"), 'w+') as f:
        f.write(f'{model_path}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar100_lr003_mu1.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=80)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/home/sunyiyou/workspace/orca/datasets')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--ckpt_dir', type=str, default='~/.cache/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=1, type=float)
    # parser.add_argument('--c1', default=0.0002, type=float)
    # parser.add_argument('--c2', default=1.0, type=float)
    # parser.add_argument('--c3', default=1e-8, type=float)
    # parser.add_argument('--c4', default=5e-5, type=float)
    # parser.add_argument('--c5', default=0.25, type=float)
    parser.add_argument('--gamma_l', default=0.2, type=float)
    parser.add_argument('--c3_rate', default=1, type=float)
    parser.add_argument('--c4_rate', default=1, type=float)
    parser.add_argument('--c5_rate', default=2, type=float)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if args.debug:
        if args.train:
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval:
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    gamma_l = 0.35
    gamma_u = 0.5
    scale = 2
    args.c1, args.c2 = 2 * gamma_l ** 2 * scale, 2 * gamma_u * scale
    args.c3, args.c4, args.c5 = gamma_l ** 4 * scale * args.c3_rate, \
                 gamma_l ** 2 * gamma_u * scale * args.c4_rate, \
                 gamma_u ** 2 * scale * args.c5_rate

    disc = f"c1-{args.c1}-c2-{args.c2}-c3-{args.c3}-c4-{args.c4}-c5-{args.c5}-gamma_l-{args.gamma_l}-r345-{args.c3_rate}-{args.c4_rate}-{args.c5_rate}"
    args.log_dir = os.path.join(args.log_dir, 'in-progress-'+'{}'.format(date.today())+args.name+'-{}'.format(disc))

    os.makedirs(args.log_dir, exist_ok=True)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)
    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'name': args.model.name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    log_file = open(os.path.join(args.log_dir, 'log.csv'), mode='w')
    fieldnames = ['epoch', 'lr', 'kmeans_acc', 'lp_acc']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')