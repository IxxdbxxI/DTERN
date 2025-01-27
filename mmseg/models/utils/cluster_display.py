import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# # 假设你的数据是一个 (T, N, C) 的张量
# T = 3  # 3 个时间步
# N = 5  # 每个时间步有 5 个聚类
# C = 20  # 每个聚类的特征维度为 20
# data = torch.randn(T, N, C).to('cuda')  # 随机生成数据

def Cluster_plots(data,iters=1):
    # data tensor[T,N,C]
    T,N,C = data.shape
    # 将数据转换为 NumPy 数组
    data_np = data.detach().cpu().numpy()  # 形状是 (T, N, C)
    # 将 (T, N, C) 的数据展平成 (T * N, C) 的形状
    data_reshaped = data_np.reshape(-1, C)

    # 使用 t-SNE 将数据降到二维
    tsne = TSNE(n_components=2, perplexity=5)
    reduced_data = tsne.fit_transform(data_reshaped)

    # 为不同的聚类分配不同的颜色
    # 这里我们为每个聚类分配不同的颜色
    colors = plt.cm.get_cmap('tab10', N)  # 获取一个包含 N 种颜色的颜色图

    # 绘制每个聚类的分布图，使用不同颜色表示不同的聚类
    plt.figure(figsize=(8, 6))

    for t in range(T):  # 对每个时间步的数据进行处理
        start_idx = t * N
        end_idx = (t + 1) * N
        for n in range(N):  # 对每个聚类点进行绘制
            plt.scatter(reduced_data[start_idx + n, 0], reduced_data[start_idx + n, 1],
                        color=colors(n), label=f"Cluster {n+1}" if t == 0 else "")

    # 设置图标题和标签
    plt.title("t-SNE - 2D visualization of clustered data (T={}, N={})".format(T, N))
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    # 去除重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # 保存图片
    plt.savefig('/root/workspace/XU/Code/VSS-MRCFA-main/cluster_imgs/tsne_cluster_plot_{}.png'.format(iters), bbox_inches='tight')  # 保存为 PNG 格式plots(data)

