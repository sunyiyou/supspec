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
from utils import plot_cluster, accuracy, TransformTwice, cluster_acc, torch_l2_dis_batch, GaussianBlur
from sklearn.cluster import KMeans
from ylib.ytool import cluster_acc
import torchvision.transforms as transforms

from train_linear import get_linear_acc
from PIL import ImageFile
from sklearn import metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.backends.cudnn.benchmark=True

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

# sudo mount -o size=1000000000000 -o nr_inodes=1000000 -o noatime, nodiratime -o remount /dev/shm
def main(log_writer, log_file, device, args):
    import open_world_imagenet as datasets

    # if args.aug == 'ez':
    #     transform_train = transforms.Compose([
    #                 transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #             ])
    # else:
    print('use hard aug')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    train_label_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet{}_label_{}_{:.2f}.txt'.format(args.dataset.numclasses, args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    train_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet{}_unlabel_{}_{:.2f}.txt'.format(args.dataset.numclasses, args.labeled_num, args.labeled_ratio), transform=TransformTwice(transform_train))
    concat_set = datasets.ConcatDataset((train_label_set, train_unlabel_set))
    labeled_idxs = range(len(train_label_set))
    unlabeled_idxs = range(len(train_label_set), len(train_label_set)+len(train_unlabel_set))
    batch_sampler = datasets.TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.train.batch_size, args.train.batch_size // 4 * 3)
    # print(int(args.train.batch_size * len(train_unlabel_set) / (len(train_label_set) + len(train_unlabel_set))))
    test_label_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet{}_label_{}_{:.2f}.txt'.format(args.dataset.numclasses, args.labeled_num, args.labeled_ratio), transform=transform_test)
    test_unlabel_set = datasets.ImageNetDataset(root=args.dataset_root, anno_file='./data/ImageNet{}_unlabel_{}_{:.2f}.txt'.format(args.dataset.numclasses, args.labeled_num, args.labeled_ratio), transform=transform_test)

    num_workers = 8
    train_loader = torch.utils.data.DataLoader(concat_set, batch_sampler=batch_sampler, num_workers=num_workers)
    test_label_loader = torch.utils.data.DataLoader(test_label_set, batch_size=args.train.batch_size, shuffle=False, num_workers=num_workers)
    test_unlabel_loader = torch.utils.data.DataLoader(test_unlabel_set, batch_size=args.train.batch_size, shuffle=False, num_workers=num_workers)

    # train_set_l  train_set_u  eval_set_l   eval_set_u
    # train_set_l = train_label_set
    # train_set_u = train_unlabel_set
    # eval_set_l = test_label_set
    # eval_set_u = test_unlabel_set
    #
    # labeled_len = len(train_set_l)
    # unlabeled_len = len(train_set_u)
    # labeled_batch_size = int(args.train.batch_size * labeled_len / (labeled_len + unlabeled_len))

    # Initialize the splits
    # train_label_loader = torch.utils.data.DataLoader(train_set_l, batch_size=labeled_batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    # train_unlabel_loader = torch.utils.data.DataLoader(train_set_u, batch_size=args.train.batch_size - labeled_batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=False)
    # test_label_loader = torch.utils.data.DataLoader(eval_set_l, batch_size=100, shuffle=True, num_workers=num_workers)
    # test_unlabel_loader = torch.utils.data.DataLoader(eval_set_u, batch_size=100, shuffle=True, num_workers=num_workers)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  TOBE Replaced !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    args.penul_feat_dim = 2048
    model = get_model(args.model, args).to(device)
    from torchvision.models.utils import load_state_dict_from_url
    # from torchvision.models.resnet import model_urls
    # state_dict = load_state_dict_from_url(model_urls['resnet50'])
    # state_dict = load_state_dict_from_url(model_urls['resnet18'])  ######## !!!!!!!!!!!!!!!!!  change 512 to 2048     1000 to 8192 !!!!!!!!!!!!!!!!!

    # state_dict = torch.load('./pretrained/simclr_imagenet_100.pth.tar')

    # ckpt = torch.load('./pretrained/supcon_imagenet1k.pth')
    # state_dict = {k.replace('module.encoder.', ''): v for k, v in ckpt['model'].items()}

    ckpt = torch.load('./pretrained/spectral_imagenet_100.pth')
    state_dict = {k.replace('module.backbone.', ''): v for k, v in ckpt['state_dict'].items()}
    model.backbone.load_state_dict(state_dict, strict=False)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!  Check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for name, param in model.named_parameters():
        if 'proj' not in name and 'layer4' not in name:
            param.requires_grad = False


    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")
    model = model.to(device)
    if args.multigpu:
        model = torch.nn.DataParallel(model)

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
        len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(0, args.train.stop_at_epoch):

        if args.multigpu:
            module = model.module
        else:
            module = model

        module.reset_stat()

        model.train()
        #######################  Train #######################
        print("number of iters this epoch: {}".format(len(train_loader)))
        # unlabel_loader_iter = cycle(train_unlabel_loader)
        # for idx, ((x1, x2), target) in enumerate(train_label_loader):

            # ((ux1, ux2), target_unlabeled) = next(unlabel_loader_iter)
            # x1, x2, ux1, ux2, target, target_unlabeled = x1.to(device), x2.to(device), ux1.to(device), ux2.to(
            #     device), target.to(device), target_unlabeled.to(device)

        for idx, ((view1, view2), combined_target, index) in enumerate(train_loader):
            labeled_idx =  batch_sampler.primary_batch_size
            # target = index[:labeled_idx]
            target = combined_target[:labeled_idx]
            target_unlabeled = combined_target[labeled_idx:].to(device)
            view1, view2, target = view1.to(device), view2.to(device), target.to(device)
            x1 = view1[:labeled_idx]
            ux1 = view1[labeled_idx:]
            x2 = view2[:labeled_idx]
            ux2 = view2[labeled_idx:]

            model.zero_grad()
            data_dict = model(x1, x2, ux1, ux2, target, mu=args.model.mu)
            loss = data_dict['loss'].mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})
            module.sync_prototype()

            # torch.cuda.empty_cache()
            if (idx + 1) % args.print_freq == 0:
                prob_msg = "\t".join([f"{val * 100:.0f}" for val in
                                      list((module.label_stat / (1e-6 + module.label_stat.sum())).data.cpu().numpy())])
                if args.multigpu:
                    if args.model.name == 'spectral':
                        loss1, loss2, loss3, loss4, loss5 = 0, data_dict["d_dict"]["loss2"].mean().item(), 0, 0, data_dict["d_dict"]["loss5"].mean().item()
                    else:
                        loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].mean().item(), data_dict["d_dict"]["loss2"].mean().item(), data_dict["d_dict"]["loss3"].mean().item(), data_dict["d_dict"]["loss4"].mean().item(), data_dict["d_dict"]["loss5"].mean().item()
                else:
                    if args.model.name == 'spectral':
                        loss1, loss2, loss3, loss4, loss5 = 0, data_dict["d_dict"]["loss2"].item(), 0, 0, \
                        data_dict["d_dict"]["loss5"].item()
                    else:
                        loss1, loss2, loss3, loss4, loss5 = data_dict["d_dict"]["loss1"].item(), data_dict["d_dict"][
                            "loss2"].item(), data_dict["d_dict"]["loss3"].item(), data_dict["d_dict"]["loss4"].item(), \
                        data_dict["d_dict"]["loss5"].item()

                print('Train: [{0}][{1}/{2}]\t Loss_all {3:.3f} \tc1:{4:.2e}\tc2:{5:.3f}\tc3:{6:.2e}\tc4:{7:.2e}\tc5:{8:.3f}\t{9}'.format(
                    epoch, idx + 1, len(train_loader), loss.item(), loss1, loss2, loss3, loss4, loss5, prob_msg
                ))

        #######################  Evaluation #######################
        model.eval()
        if args.multigpu:
            module = model.module
        else:
            module = model

        def feat_extract(loader, proto_type, layer='penul'):
            targets = np.array([])
            features = []
            preds = np.array([])
            with torch.no_grad():
                for idx, (x, labels, index) in enumerate(loader):
                    # feat = model.backbone.features(x.to(device, non_blocking=True))
                    ret_dict = module.forward_eval(x.to(device, non_blocking=True), proto_type, layer=layer)
                    pred = ret_dict['label_pseudo']
                    feat = ret_dict['features']
                    preds = np.append(preds, pred.cpu().numpy())
                    targets = np.append(targets, labels.cpu().numpy())
                    features.append(feat.data.cpu().numpy())
                    # if idx > 20:
                    #     break
            return np.concatenate(features), targets.astype(int), preds


        # if (epoch + 1) % 1 == 0:
        if (epoch + 1) % args.deep_eval_freq == 0:

            normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

            features_test_u, ltest_u, preds_u = feat_extract(test_unlabel_loader, proto_type='all', layer=args.layer)
            features_test_l, ltest_l, preds_l = feat_extract(test_label_loader, proto_type='all', layer=args.layer)

            ftest_l = normalizer(features_test_l)
            ftest_u = normalizer(features_test_u)

            seen_mask = ltest_u < args.labeled_num
            unseen_mask = ~seen_mask
            proto_overall_acc = cluster_acc(preds_u, ltest_u)
            proto_seen_acc = accuracy(preds_u[seen_mask], ltest_u[seen_mask])
            proto_unseen_acc = cluster_acc(preds_u[unseen_mask], ltest_u[unseen_mask])

            #######################  Linear Probe #######################
            # lp_acc, _ = get_linear_acc(ftrain, ltrain, ftest, ltest_n, args.labeled_num, print_ret=False)
            # lp_acc, (clf_known, _, _, lp_preds_k) = get_linear_acc(ftrain_k, ltrain_k, ftest_k, ltest_k, args.labeled_num, print_ret=False)

            #######################  K-Means #######################

            if args.layer == 'penul':
                featdim = args.penul_feat_dim
            else:
                featdim = args.proj_feat_dim

            X = ftest_u
            centroids_size = args.dataset.numclasses
            centroids = np.zeros((centroids_size, featdim))
            for li in range(args.labeled_num):
                centroids[li, :] = ftest_l[ltest_l == li].mean(0)

            import numpy.linalg as linalg
            from copy import deepcopy

            X = torch.Tensor(X).cuda()
            centroids = torch.Tensor(centroids).cuda()
            # K-Means++ initialization
            for icls in range(args.labeled_num, centroids_size):
                # distances = torch.norm(X - centroids[:icls, None, :], dim=2)
                distances = torch_l2_dis_batch(X, centroids[:icls, None, :])
                idx_min = torch.argmin(distances, dim=0)
                dist_min = distances[tuple(np.stack((idx_min.data.cpu().numpy(), np.arange(len(idx_min)))))]
                imax = torch.argmax(dist_min) % len(X)
                centroids[icls, :] = X[imax.item(), :]

            print(f"init done")
            ndata = X.shape[0]
            nfeature = X.shape[1]
            centers_init = centroids
            # Store new centers
            centers = deepcopy(centers_init)

            preds_kmeans = np.zeros(ndata)
            for iter in range(100):
                # print(iter)
                # distances = torch.norm(X - centers[:, None, :], dim=2)
                distances = torch_l2_dis_batch(X, centers[:, None, :])
                preds_kmeans = torch.argmin(distances, dim=0)

                for icls in range(args.labeled_num, args.dataset.numclasses):
                    if (preds_kmeans == icls).sum().item() > 0:
                        centers[icls, :] = torch.mean(X[preds_kmeans == icls], dim=0)
                # centers[icls, :] = torch.mean(torch.cat([X[preds_kmeans == icls], torch.Tensor(ftest_l)[ltest_l == icls].to(device)]), dim=0)

            preds = preds_kmeans.data.cpu().numpy().astype(int)
            overall_acc = cluster_acc(preds, ltest_u)
            seen_acc = accuracy(preds[seen_mask], ltest_u[seen_mask])
            unseen_acc = cluster_acc(preds[unseen_mask], ltest_u[unseen_mask])
            unseen_nmi = metrics.normalized_mutual_info_score(ltest_u[unseen_mask], preds[unseen_mask])

            print(f"K-means: Seen Acc: {seen_acc:.4f}\t Unseen ACC: {unseen_acc:.4f}\t Overall Acc: {overall_acc:.4f}")
            print(f"Proto:   Seen Acc: {proto_seen_acc:.4f}\t Unseen ACC: {proto_unseen_acc:.4f}\t Overall Acc: {proto_overall_acc:.4f}")
            torch.cuda.empty_cache()

            write_dict = {
                'epoch': epoch,
                'lr': lr_scheduler.get_lr(),
                'proto_overall_acc': proto_overall_acc,
                'proto_seen_acc': proto_seen_acc,
                'proto_unseen_acc': proto_unseen_acc,
                'overall_acc': overall_acc,
                'seen_acc': seen_acc,
                'unseen_acc': unseen_acc,
                'unseen_nmi': unseen_nmi,
                # 'kmeans_acc_train': kmeans_acc_train,
                # 'kmeans_acc_test': kmeans_acc_test,
                # 'kmeans_overall_acc': kmeans_overall_acc,
                # 'lp_acc': lp_acc
            }

            # print(f"K-Means Train Acc: {kmeans_acc_train:.4f}\t K-Means Test ACC: {kmeans_acc_test:.4f}\t K-Means All Acc: {kmeans_overall_acc:.4f}\t linear Probe Acc: {lp_acc:.4f}")

            log_writer.writerow(write_dict)
            log_file.flush()

            #######################  Visualization #######################
            if (epoch + 1) % args.vis_freq == 0:
                sampling_size = int(0.15 * len(ftest_u))
                rand_idx = np.random.choice(range(len(ftest_u)), sampling_size, replace=False)
                fvis = normalizer(ftest_u)[rand_idx]
                tvis = ftest_u[rand_idx].astype(int)

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
                'state_dict': module.state_dict()
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
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config-file', default='configs_openncd/supspectral_resnet18_imagenet.yaml', type=str)
    parser.add_argument('-c', '--config-file', default='configs_openncd/supspectral_resnet50_mlp8192_norelu_imagenet100.yaml', type=str)
    # parser.add_argument('-c', '--config-file', default='configs_openncd/spectral_resnet50_mlp8192_imagenet.yaml', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test_bs', type=int, default=80)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--dataset_root', default='/home/sunyiyou/dataset/imagenet/train', type=str)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--log_dir', type=str, default='./log/')
    parser.add_argument('--ckpt_dir', type=str, default='~/.cache/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--vis_freq', type=int, default=200)
    parser.add_argument('--deep_eval_freq', type=int, default=5)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
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
    parser.add_argument('--proj_feat_dim', default=8192, type=int)
    parser.add_argument('--layer', default='proj', type=str)

    parser.add_argument('--went', default=0.0, type=float)
    parser.add_argument('--momentum_proto', default=0.95, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--base_lr', default=0.05, type=float)
    parser.add_argument('--aug', default='hard', type=str)
    parser.add_argument('--batch_size', default=384, type=int)
    parser.add_argument('--multigpu', default=False, type=str2bool)
    parser.add_argument('--epochs', default=101, type=float)
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

    args.train.batch_size = args.batch_size
    args.train.stop_at_epoch = args.epochs
    args.train.num_epochs = args.epochs
    args.train.base_lr = args.base_lr

    assert not None in [args.log_dir, args.dataset_root, args.ckpt_dir, args.name]

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
        'dataset_root': args.dataset_root,
        'download':args.download,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': False,
        'num_workers': args.dataset.num_workers,
    }

    log_file = open(os.path.join(args.log_dir, 'log.csv'), mode='w')
    fieldnames = ['epoch', 'lr', 'unseen_acc', 'seen_acc', 'overall_acc', 'proto_unseen_acc', 'proto_seen_acc', 'proto_overall_acc', 'unseen_nmi',]    # 'kmeans_acc_train', 'kmeans_acc_test', 'kmeans_overall_acc', 'lp_acc']
    log_writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    log_writer.writeheader()

    return args, log_file, log_writer


if __name__ == "__main__":
    args, log_file, log_writer = get_args()

    main(log_writer, log_file, device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')