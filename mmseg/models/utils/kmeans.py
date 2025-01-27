import torch
import numpy as np
import torch
import numpy as np


def find_distances(points, centroids):
    # Compute distances between points and centroids
    distances = ((points - centroids[:, None, :]) ** 2).sum(axis=-1)
    return distances

def find_distances_efficient(points, centroids):
    # points: (N, d)
    # centroids: (k, d)
    # Result: distances (k, N)
    
    dists = torch.cdist(points, centroids, p=2) ** 2  # 使用 PyTorch 的 cdist
    return dists.T

def get_new_centroids(points, indexes_for_clusters, n, k,device):
    # Initialize tensors to accumulate sum and count of points for each centroid
    centroids = torch.zeros((k, points.shape[1]), device=device, dtype=torch.float)
    sum_points = torch.zeros((k, points.shape[1]), device=device, dtype=torch.float)
    counts = torch.zeros(k, device=device, dtype=torch.float) + 1e-6

    for i in range(k):
        mask = (indexes_for_clusters == i)
        sum_points[i] = points[mask].sum(dim=0)
        counts[i] = mask.sum()

    # Compute new centroids
    centroids = sum_points / counts.unsqueeze(1)

    return centroids

'''
points: 2D torch.tensor of points in (x, y) format (ex. [[1, 2], [4,5], ..])
k: number of clusters in dataset
max_iter: maximum number of times the function checks for convergance.
'''
def kmeans(points, k, max_iter=50):
    device = points.device
    points = points
    n = points.shape[0]
    indexes = torch.linspace(0, n-1, steps=k, device=device).long()
    centroids = points[indexes]

    for _ in range(max_iter):
        distances = find_distances_efficient(points, centroids)
        indexes_for_clusters = distances.argmin(axis=0)
        new_centroids = get_new_centroids(points, indexes_for_clusters, n, k,device)

        # Check for convergence
        if (((new_centroids-centroids) ** 2).sum())==0:
          break
        else:
          centroids = new_centroids

    # print(f"Converged in {iteration+1} iterations")
    return new_centroids,indexes_for_clusters


def recompute_cluster_centers(x, pesudo_labels_batch, num_cluster):
    # x: 原始特征点 (batch_size, n, dims)
    # pesudo_labels_batch: 聚类结果 (batch_size, n)
    # num_cluster: 类别数
    batch_size, _, dims = x.shape

    # 初始化类别中心和计数器
    cluster_centers_batch = torch.zeros((batch_size, num_cluster, dims), device=x.device, dtype=torch.float)
    cluster_counts_batch = torch.zeros((batch_size, num_cluster), device=x.device, dtype=torch.float)

    # 遍历每个样本
    for b in range(batch_size):
        # 获取当前样本的特征点和聚类标签
        points = x[b].view(-1,dims)  # (h*w, dims)
        labels = pesudo_labels_batch[b].view(-1)  # (h*w,)

        # 遍历每个类别
        for k in range(num_cluster):
            mask = (labels == k)  # 找到属于类别 k 的点
            if mask.sum() > 0:  # 如果类别 k 有点
                cluster_centers_batch[b, k] = points[mask].mean(dim=0)  # 更新类别中心
                cluster_counts_batch[b, k] = mask.sum()  # 记录点的数量

    return cluster_centers_batch



# dims = 256
# batch_size = 4
# num_cluster = 200
# h,w = 240,60
# x = np.random.rand(batch_size, dims, h, w)

# x = torch.from_numpy(x)
# x = x.to("cuda:2")
# cluster_centers_batch = []
# import time
# t1 = time.time()
# cluster_centers_batch = []
# pesudo_labels_batch = []
# for i in range(batch_size):
#     b_cluster = x[i].view(dims, -1).t()
#     cluster_centers,indexes_for_clusters = kmeans(b_cluster, num_cluster)
#     pesudo_labels_batch.append(indexes_for_clusters.view(1,-1)) # (1, h*w)
#     cluster_centers_batch.append(cluster_centers.view(1, num_cluster, dims))
# cluster_centers_batch = torch.cat(cluster_centers_batch, dim=0)
# pesudo_labels_batch = torch.cat(pesudo_labels_batch, dim=0) # (batch_size, h*w)

# x = x.flatten(2).permute(0, 2, 1)  # (batch_size, h*w, dims)
# cluster_centers_batch2 = recompute_cluster_centers(x, pesudo_labels_batch, num_cluster)
# print(cluster_centers_batch2.shape)
# print(torch.equal(cluster_centers_batch, cluster_centers_batch2))

# diff = torch.abs(cluster_centers_batch - cluster_centers_batch2) # 误差引起的
# print("最大差异:", diff.max())
# print("最小差异:", diff.min())
# t2 = time.time()
# print("times(s)",t2-t1)
# print(cluster_centers_batch.shape)




# import numpy as np
# import faiss
# import sys
# import time
# import warnings

# if not sys.warnoptions:
#     # suppress pesky PIL EXIF warnings
#     warnings.simplefilter("once")
#     warnings.filterwarnings("ignore", message="(Possibly )?corrupt EXIF data.*")
#     warnings.filterwarnings("ignore", message="numpy.dtype size changed.*")
#     warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*")


# def preprocess_features(x, d=256):
#     """
#     Calculate PCA + Whitening + L2 normalization for each vector

#     Args:
#         x (ndarray): N x D, where N is number of vectors, D - dimensionality
#         d (int): number of output dimensions (how many principal components to use).
#     Returns:
#         transformed [N x d] matrix xt .
#     """
#     n, orig_d = x.shape
#     pcaw = faiss.PCAMatrix(d_in=orig_d, d_out=d, eigen_power=-0.5, random_rotation=False)
#     pcaw.train(x)
#     assert pcaw.is_trained
#     print( 'Performing PCA + whitening')
#     x = pcaw.apply_py(x)
#     print( 'x.shape after PCA + whitening:', x.shape)
#     l2normalization = faiss.NormalizationTransform(d, 2.0)
#     print( 'Performing L2 normalization')
#     x = l2normalization.apply_py(x)
#     return x


# def train_kmeans(x, num_clusters=1000, num_gpus=1):
#     """
#     Runs k-means clustering on one or several GPUs
#     """
#     d = x.shape[1]
#     kmeans = faiss.Clustering(d, num_clusters)
#     kmeans.verbose = True
#     kmeans.niter = 20

#     # otherwise the kmeans implementation sub-samples the training set
#     kmeans.max_points_per_centroid = 1000000

#     res = [faiss.StandardGpuResources() for i in range(num_gpus)]

#     flat_config = []
#     for i in range(num_gpus):
#         cfg = faiss.GpuIndexFlatConfig()
#         cfg.useFloat16 = False
#         cfg.device = i
#         flat_config.append(cfg)

#     if num_gpus == 1:
#         index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
#     else:
#         indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
#                    for i in range(num_gpus)]
#         index = faiss.IndexProxy()
#         for sub_index in indexes:
#             index.addIndex(sub_index)

#     # perform the training
#     kmeans.train(x, index)
#     print( 'Total number of indexed vectors (after kmeans.train()):', index.ntotal)
#     centroids = faiss.vector_float_to_array(kmeans.centroids)

#     # objective = faiss.vector_float_to_array(kmeans.obj)
#     stats = kmeans.iteration_stats
#     losses = np.array([stats.at(i).obj for i in range(stats.size())])
#     print( 'Objective values per iter:', losses)
#     print( "Final objective: %.4g" % losses[-1])

#     # TODO: return cluster assignment

#     return centroids.reshape(num_clusters, d)


# def compute_cluster_assignment(centroids, x):
#     assert centroids is not None, "should train before assigning"
#     d = centroids.shape[1]
#     index = faiss.IndexFlatL2(d)
#     index.add(centroids)
#     distances, labels = index.search(x, 1)
#     return labels.ravel()


# def do_clustering(features, num_clusters, num_gpus=None):
#     if num_gpus is None:
#         num_gpus = faiss.get_num_gpus()
#     print ('FAISS: using {} GPUs').format(num_gpus)
#     features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
#     features = preprocess_features(features)

#     print ('Run FAISS clustering...')
#     t0 = time.time()
#     centroids = train_kmeans(features, num_clusters, num_gpus)
#     print( 'Compute cluster assignment')
#     labels = compute_cluster_assignment(centroids, features)
#     print( 'centroids.shape:', centroids.shape)
#     print( 'labels.shape:', labels.shape)
#     t1 = time.time()
#     print( "Total elapsed time: %.3f m" % ((t1 - t0) / 60.0))
#     return labels


# def example():
#     k = 1000
#     ngpu = 1

#     x = np.random.rand(3600*4, 256)
#     print( "reshape")
#     x = x.reshape(x.shape[0], -1).astype('float32')
#     # x = preprocess_features(x)

#     print( "run")
#     t0 = time.time()
#     centroids = train_kmeans(x, k, ngpu)
#     print( 'compute_cluster_assignment')
#     labels = compute_cluster_assignment(centroids, x)
#     print( 'centroids.shape:', centroids.shape)
#     print( 'labels.shape:', labels.shape)
#     print(type(labels))
#     print(labels[:10])
#     t1 = time.time()

#     print( "total runtime: %.3f s" % (t1 - t0))


# if __name__ == '__main__':
#     example()
    