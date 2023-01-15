#
# import numpy as np
# from copy import deepcopy
# def semi_kmeans(ftrain_k, ltrain_k, ftrain_n, ltrain_n, labeled_num=50, num_classes=100):
#
#     centroids_size = num_classes
#     centroids = np.zeros((centroids_size, ftrain_k.shape[1]))
#     for li in range(labeled_num):
#         centroids[li, :] = ftrain_k[ltrain_k == li].mean(0)
#
#     import numpy.linalg as linalg
#
#     # K-Means++ initialization
#     for icls in range(labeled_num, centroids_size):
#         print(f"init: {icls}")
#         distances = linalg.norm(X - centroids[:icls, None, :], axis=2)
#         idx_min = np.argmin(distances, axis=0)
#         dist_min = distances[tuple(np.stack((idx_min, np.arange(len(idx_min)))))]
#         imax = np.argmax(dist_min) % len(X)
#         centroids[icls, :] = X[imax, :]
#
#
#     ndata = X.shape[0]
#     nfeature = X.shape[1]
#
#     centers_init = centroids
#     # Store new centers
#     centers = deepcopy(centers_init)
#
#     X = ftrain_n
#     preds_kmeans = np.zeros(ndata)
#     # When, after an update, the estimate of that center stays the same, exit loop
#     for iter in range(300):
#         if iter % 10 == 0:
#             print("iter: ", iter)
#
#         distances = linalg.norm(X - centroids[:, None, :], axis=2)
#
#         # Assign all training data to closest center
#         preds_kmeans = np.argmin(distances, axis=0)
#
#         for icls in range(num_classes):
#             if (preds_kmeans == icls).any():
#                 if icls > labeled_num:
#                     centers[icls, :] = np.mean(X[preds_kmeans == icls], axis=0)
#                 else:
#                     centers[icls, :] = np.mean(np.concatenate([X[preds_kmeans == icls], features_labeled[targets_labeled == icls]]), axis=0)
#                     # centers[icls, :] = np.mean(X[preds_kmeans == icls], axis=0)
#
#     preds = preds.astype(int)
#     overall_acc = cluster_acc(preds, targets)
#     seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
#     unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
#     unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
#     mean_uncert = 1 - np.mean(confs)
#     print('PROTOTYPE Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, nmi {:.4f}'.format(overall_acc, seen_acc, unseen_acc, unseen_nmi))
#
#     preds = preds_kmeans
#     overall_acc = cluster_acc(preds, targets)
#     seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
#     unseen_acc = cluster_acc(preds[unseen_mask], targets[unseen_mask])
#     unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
#     mean_uncert = 1 - np.mean(confs)
#     print('KMEANS Test overall acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, nmi {:.4f}'.format(overall_acc, seen_acc, unseen_acc, unseen_nmi))
