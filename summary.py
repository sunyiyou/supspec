import os
import pandas as pd
import re

root = '/home/sunyiyou/Downloads/log'
dirs = os.listdir(root)

head = ["arch", "dataset", "labelnum", "c1", "c2", "c3", "c4", "c5", "gamma_l", "gamma_u","r3", "r4", "r5", "featdim", "went", "mm", "lr", "seed"] + \
       ["epoch", "lr", "unseen_acc", "seen_acc", "overall_acc", "kmeans_acc_train", "kmeans_acc_test", "kmeans_overall_acc", "lp_acc"]
print("\t".join(head))

for dirname in dirs:
    train_match = re.findall("supspectral-(.+)-(.+)-labelnum-(\d+)-c1-(\d\.\d+)-c2-(\d\.\d+)-c3-(\d\.\d+e[-|+]\d+)-c4-(\d\.\d+e[-|+]\d+)-c5-(\d\.\d+e[-|+]\d+)-gamma_l-(\d\.\d+)-gamma_u-(\d\.\d+)-r345-(\d\.\d+)-(\d\.\d+)-(\d\.\d+)-fdim-(\d+)-went(\d\.\d+)-mm(\d\.\d+)-lr(\d\.\d+)-seed(\d+)", dirname.strip())

    if len(train_match) > 0:
        # arch, dataset, c1, c2, c3, c4, c5, gamma_l, r3, r4, r5, featdim, went, mm, seed = train_match[0]
        try:
            df = pd.read_csv(os.path.join(root, dirname, 'log.csv'))
            # data = df.iloc[df['kmeans_acc_test'].argmax()].to_dict()
            data = df.iloc[df['overall_acc'].argmax()].to_dict()
            datamsg = [f"{i}" for i in train_match[0]] + [f"{v}" for k, v in data.items()]
            print("\t".join(datamsg))
        except pd.errors.EmptyDataError:
            print("\t".join([f"{i}" for i in train_match[0]]))
    else:
        print("unmatched: ", dirname)
