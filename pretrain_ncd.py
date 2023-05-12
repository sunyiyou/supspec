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
from utils import plot_cluster, accuracy
from sklearn.cluster import KMeans
from ylib.ytool import cluster_acc

from train_linear import get_linear_acc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class kmeans2classsifier:
    def __init__(self, estimator):
        classifier_novel = estimator.cluster_centers_
        self.w = 2 * classifier_novel  # Nu * fdim
        self.b = -np.linalg.norm(classifier_novel, 2, 1) ** 2  # Nu
    def predict(self, input):
        out = np.matmul(input, self.w.T) + self.b
        return out.argmax(1), out

class emsemble:
    def __init__(self, w1, w2, b1, b2, scale2=1, bias2=0):
        self.w = np.concatenate([w1, scale2 * w2], axis=0)
        self.b = np.concatenate([b1, bias2 * b2], axis=0)
    def predict(self, input):
        out = np.matmul(input, self.w.T) + self.b
        return out.argmax(1), out

def main(log_writer, log_file, device, args):
    iter_count = 0

    import open_world_cifar as datasets
    dataroot = args.data_dir
    if args.dataset.name == 'cifar10':
        train_set_known = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs))
        train_set_novel = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), unlabeled_idxs=train_set_known.unlabeled_idxs)
        train_set_known_eval = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        train_set_novel_eval = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), unlabeled_idxs=train_set_known.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR10(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        test_set_known = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets < args.labeled_num])
        test_set_novel = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets >= args.labeled_num])
        args.num_classes = 10
    elif args.dataset.name == 'cifar100':

        known_class_division_1 = [
            "beaver", "dolphin", "otter", "seal", "whale", "aquarium_fish", "flatfish", "ray", "shark", "trout",
            "orchid", "poppy", "rose", "sunflower", "tulip", "bottle", "bowl", "can", "cup", "plate",
            "apple", "mushroom", "orange", "pear", "sweet_pepper", "clock", "keyboard", "lamp",
            "telephone", "television", "bed", "chair", "couch", "table", "wardrobe", "bee", "beetle", "butterfly",
            "caterpillar", "cockroach", "bear", "leopard", "lion", "tiger", "wolf", "bridge", "castle", "house", "road",
            "skyscraper"
        ]

        known_class_division_2 = [
            "beaver", "dolphin", "otter", "aquarium_fish", "flatfish", "orchid", "poppy", "rose", "bottle",
            "bowl", "apple", "mushroom", "orange", "clock", "keyboard", "bed", "chair", "couch", "bee", "beetle",
            "bear", "leopard", "lion", "bridge", "castle", "cloud", "forest", "mountain", "camel", "cattle", "fox",
            "porcupine", "possum", "crab", "lobster", "baby", "boy", "girl", "crocodile", "dinosaur", "hamster",
            "mouse", "rabbit", "maple_tree", "oak_tree", "bicycle", "bus", "motorcycle", "lawn_mower", "rocket"
        ]

        class_list = known_class_division_2

        train_set_known = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), class_list=class_list)
        train_set_novel = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=True, **args.aug_kwargs), class_list=class_list, unlabeled_idxs=train_set_known.unlabeled_idxs)
        train_set_known_eval = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=True, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list)
        train_set_novel_eval = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list, unlabeled_idxs=train_set_known.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root=dataroot, labeled=False, train=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs), class_list=class_list)
        test_set_known = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets < args.labeled_num])
        test_set_novel = datasets.data.Subset(test_set, np.arange(len(test_set))[test_set.targets >= args.labeled_num])
        args.num_classes = 100


    labeled_len = len(train_set_known)
    unlabeled_len = len(train_set_novel)
    labeled_batch_size = int(args.train.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    train_label_loader = torch.utils.data.DataLoader(train_set_known, batch_size=labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_unlabel_loader = torch.utils.data.DataLoader(train_set_novel, batch_size=args.train.batch_size - labeled_batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader_known_eval = torch.utils.data.DataLoader(train_set_known_eval, batch_size=100, shuffle=True, num_workers=2)
    train_loader_novel_eval = torch.utils.data.DataLoader(train_set_novel_eval, batch_size=100, shuffle=True, num_workers=2)
    test_loader_known = torch.utils.data.DataLoader(test_set_known, batch_size=100, shuffle=False, num_workers=2)
    test_loader_novel = torch.utils.data.DataLoader(test_set_novel, batch_size=100, shuffle=False, num_workers=2)

    # define model
    model = get_model(args.model, args).to(device)
    # model = torch.nn.DataParallel(model)

    # if args.dataset.name == 'cifar10':
    #     # state_dict = torch.load('./pretrained/spectral_cifar_10.pth.tar')['state_dict']
    #     path = "/media/sunyiyou/ubuntu-hdd1/Downloads/spectral_contrastive_learning/log/awslog/completed-2023-05-09spectral-resnet18-mlp1000-norelu-cifar10-lr003-mu1-labelnum-5-c1-0.04-c2-4.0-c3-5.1e-04-c4-9.0e-02-c5-4.0e+00-gamma_l-0.15-gamma_u-2.00-r345-1-2-1-fdim-1000-went0.0-mm0.95-lr0.03-seed6/checkpoints/1200.pth"
    #     state_dict = torch.load(path)['state_dict']
    #     state_dict.pop('label_stat', None)
    # elif args.dataset.name == 'cifar100':
    #     state_dict = torch.load('./pretrained/spectral_cifar_100.pth.tar')['state_dict']
    # state_dict = {k: v for k, v in state_dict.items() if 'encoder' not in k}
    # model.load_state_dict(state_dict, strict=False)
    # model.featdim = 1000
    #
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
        model.reset_stat()

        #######################  Train #######################
        model.train()
        print("number of iters this epoch: {}".format(len(train_label_loader)))
        unlabel_loader_iter = cycle(train_unlabel_loader)
        for idx, ((x1, x2), target) in enumerate(train_label_loader):
            ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            x1, x2, ux1, ux2, target, target_unlabeled = x1.to(device), x2.to(device), ux1.to(device), ux2.to(device), target.to(device), target_unlabeled.to(device)

            model.zero_grad()
            data_dict = model.forward_ncd(x1, x2, ux1, ux2, target)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            model.sync_prototype()

            if (idx + 1) % args.print_freq == 0:
                prob_msg = "\t".join([f"{val * 100:.0f}" for val in
                                      list((model.label_stat / (1e-6 + model.label_stat.sum())).data.cpu().numpy())])

                if args.model.name == 'spectral':
                    loss1, loss2, loss3, loss4, loss5 = 0, data_dict["d_dict"]["loss2"].item(), 0, 0, data_dict["d_dict"]["loss5"].item()
                else:
                    loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"]["loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), data_dict["d_dict"]["loss5"].item()

                print('Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} \tc1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}\t{9}'.format(
                    epoch, idx + 1, len(train_label_loader), loss.item(), loss1, loss2, loss3, loss4, loss5, prob_msg
                ))


        #######################  Evaluation #######################
        model.eval()

        def feat_extract(loader, proto_type):
            targets = np.array([])
            features = []
            preds = np.array([])
            for idx, (x, labels) in enumerate(loader):
                # feat = model.backbone.features(x.to(device, non_blocking=True))
                ret_dict = model.forward_eval(x.to(device, non_blocking=True), proto_type)
                pred = ret_dict['label_pseudo']
                feat = ret_dict['features']
                preds = np.append(preds, pred.cpu().numpy())
                targets = np.append(targets, labels.cpu().numpy())
                features.append(feat.data.cpu().numpy())
            return np.concatenate(features), targets.astype(int), preds


        if (epoch + 1) % args.deep_eval_freq == 0:

            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

            features_train_k, ltrain_k, _ = feat_extract(train_loader_known_eval, proto_type='known')
            features_train_n, ltrain_n, _ = feat_extract(train_loader_novel_eval, proto_type='novel')
            features_test_k, ltest_k, _ = feat_extract(test_loader_known, proto_type='known')
            features_test_n, ltest_n, _ = feat_extract(test_loader_novel, proto_type='novel')

            ftrain_k = normalizer(features_train_k)
            ftrain_n = normalizer(features_train_n)
            ftest_k = normalizer(features_test_k)
            ftest_n = normalizer(features_test_n)

            #######################  Linear Probe #######################
            # lp_acc, _ = get_linear_acc(ftrain, ltrain, ftest, ltest_n, args.labeled_num, print_ret=False)
            lp_acc, (clf_known, _, _, lp_preds_k) = get_linear_acc(ftrain_k, ltrain_k, ftest_k, ltest_k, args.labeled_num, print_ret=False)


            #######################  K-Means #######################
            alg = KMeans(init="k-means++", n_clusters=args.num_classes - args.labeled_num, n_init=5, random_state=0)
            estimator = alg.fit(ftrain_n)
            kmeans_acc_train = cluster_acc(estimator.predict(ftrain_n), ltrain_n.astype(np.int64))
            kmeans_preds_test_n = estimator.predict(ftest_n)
            kmeans_acc_test = cluster_acc(kmeans_preds_test_n, ltest_n.astype(np.int64))

            kmeans_preds_all = np.concatenate([lp_preds_k.astype(np.int32), kmeans_preds_test_n + args.labeled_num])
            targets_all = np.concatenate([ltest_k, ltest_n])
            kmeans_overall_acc = cluster_acc(kmeans_preds_all, targets_all)


            #######################  Task Agnostic #######################
            centers_novel = estimator.cluster_centers_
            centroids = np.zeros((args.num_classes, ftrain_k.shape[1]))
            for li in range(args.num_classes):
                if li < args.labeled_num:
                    centroids[li, :] = ftrain_k[ltrain_k == li].mean(0)
                else:
                    centroids[li, :] = centers_novel[li - args.labeled_num]
            # centroids = normalizer(centroids)
            preds_k = ((ftest_k - centroids[:, None, :]) ** 2).sum(2).argmin(0)
            preds_n = ((ftest_n - centroids[:, None, :]) ** 2).sum(2).argmin(0)
            seen_acc = (preds_k == ltest_k).sum() / len(ltest_k)
            unseen_acc = cluster_acc(preds_n, ltest_n)
            overall_acc = cluster_acc(np.concatenate([preds_k, preds_n]), targets_all)

            # clf_novel = kmeans2classsifier(estimator)
            # # verify # # kmeans_preds_test_n = ((ftest_n - classifier_novel[:, None, :]) ** 2).sum(2).argmin(0)
            # # verify # # (kmeans_preds_test_n == clf_novel.predict(ftest_n)[0]).sum()
            ### emsembling ###
            # centers = estimator.cluster_centers_
            # w_n = 2 * centers  # Nu * fdim
            # b_n = -np.linalg.norm(centers, 2, 1) ** 2  # Nu
            # w_k = clf_known.fc.weight.data.cpu().numpy()
            # b_k = clf_known.fc.bias.data.cpu().numpy()
            # w_k = w_k / np.linalg.norm(w_k, 2, 1).mean()
            # b_k = b_k / np.linalg.norm(w_k, 2, 1).mean()
            # clf_all = emsemble(w_k, w_n, b_k, b_n, scale2=1, bias2=0)
            # preds_all, pred_logit = clf_all.predict(np.concatenate([ftest_k, ftest_n]))

            write_dict = {
                'epoch': epoch,
                'lr': lr_scheduler.get_lr(),
                'overall_acc': overall_acc,
                'seen_acc': seen_acc,
                'unseen_acc': unseen_acc,
                'kmeans_acc_train': kmeans_acc_train,
                'kmeans_acc_test': kmeans_acc_test,
                'kmeans_overall_acc': kmeans_overall_acc,
                'lp_acc': lp_acc
            }

            print(f"K-Means Train Acc: {kmeans_acc_train:.4f}\t K-Means Test ACC: {kmeans_acc_test:.4f}\t K-Means All Acc: {kmeans_overall_acc:.4f}\t linear Probe Acc: {lp_acc:.4f}")
            print(f"Seen Acc: {seen_acc:.4f}\t Unseen ACC: {unseen_acc:.4f}\t Overall Acc: {overall_acc:.4f}")

            log_writer.writerow(write_dict)
            log_file.flush()

            #######################  Visualization #######################
            if (epoch + 1) % args.vis_freq == 0:
                sampling_size = int(0.15 * len(ftrain_n))
                rand_idx = np.random.choice(range(len(ftrain_n)), sampling_size, replace=False)
                fvis = normalizer(ftrain_n)[rand_idx]
                tvis = ltrain_n[rand_idx].astype(int)

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
    # parser.add_argument('-c', '--config-file', default='configs/supspectral_resnet_mlp1000_norelu_cifar100.yaml', type=str)
    parser.add_argument('-c', '--config-file', default='configs/spectral_resnet_mlp1000_norelu_cifar10_lr003_mu1.yaml', type=str)
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
    parser.add_argument('--vis_freq', type=int, default=2000)
    parser.add_argument('--deep_eval_freq', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=5, type=int)
    parser.add_argument('--labeled-ratio', default=1, type=float)
    # parser.add_argument('--c1', default=0.0002, type=float)
    # parser.add_argument('--c2', default=1.0, type=float)
    # parser.add_argument('--c3', default=1e-8, type=float)
    # parser.add_argument('--c4', default=5e-5, type=float)
    # parser.add_argument('--c5', default=0.25, type=float)
    parser.add_argument('--gamma_l', default=0.15, type=float)
    parser.add_argument('--gamma_u', default=2, type=float)
    parser.add_argument('--c3_rate', default=1, type=float)
    parser.add_argument('--c4_rate', default=2, type=float)
    parser.add_argument('--c5_rate', default=1, type=float)
    parser.add_argument('--proj_feat_dim', default=1000, type=int)
    parser.add_argument('--went', default=0.0, type=float)
    parser.add_argument('--momentum_proto', default=0.95, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--base_lr', default=0.05, type=float)

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            if key not in vars(args):
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

    gamma_l = args.gamma_l
    gamma_u = args.gamma_u
    scale = 1
    args.c1, args.c2 = 2 * gamma_l ** 2 * scale, 2 * gamma_u * scale
    args.c3, args.c4, args.c5 = gamma_l ** 4 * scale * args.c3_rate, \
                 gamma_l ** 2 * gamma_u * scale * args.c4_rate, \
                 gamma_u ** 2 * scale * args.c5_rate

    disc = f"labelnum-{args.labeled_num}-c1-{args.c1:.2f}-c2-{args.c2:.1f}-c3-{args.c3:.1e}-c4-{args.c4:.1e}-c5-{args.c5:.1e}-gamma_l-{args.gamma_l:.2f}-gamma_u-{args.gamma_u:.2f}-r345-{args.c3_rate}-{args.c4_rate}-{args.c5_rate}"+ \
           f"-fdim-{args.proj_feat_dim}-went{args.went}-mm{args.momentum_proto}-lr{args.base_lr}-seed{args.seed}"
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
    fieldnames = ['epoch', 'lr', 'unseen_acc', 'seen_acc', 'overall_acc', 'kmeans_acc_train', 'kmeans_acc_test', 'kmeans_overall_acc', 'lp_acc']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')