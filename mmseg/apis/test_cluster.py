import os
import os.path as osp
import pickle
import shutil
import tempfile
from PIL import Image
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from IPython import embed
from mmseg.ops import resize
import matplotlib.pyplot as plt
import torch.nn.functional as F
def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', prefix='/kaggle/working/outs/',delete=False).name
    # np.save(temp_file_name, array)
    image = Image.fromarray(array.astype(np.uint8))
    image.save(temp_file_name)
    return temp_file_name

def overlay_mask(image, mask, alpha=0.5):
    """
    将掩码叠加到原始图像上
    :param image: 原始图像 (PIL Image)
    :param mask: 掩码 (numpy array, shape=(H, W, 3))
    :param alpha: 透明度
    :return: 叠加后的图像 (PIL Image)
    """
    mask_pil = Image.fromarray(mask)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    overlay = Image.blend(mask_pil, mask_pil, alpha=alpha)
    return overlay


def single_gpu_test(model,
                    data_loader,
                    show=True,
                    out_dir=None,
                    efficient_test=False,lists=['50_9mZFBNGzmok']):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    # print("single_gpu_test")

    for i, data in enumerate(data_loader):
        file_name = data['img_metas'][0].data[0][0]['filename']
        cd_name = file_name.split('/')[-3]

        if lists is not None and cd_name in lists: 
            pre_file_name = file_name.split('/')[-3]
            dir_pred = '/kaggle/working/results/'+pre_file_name # lsk3  iter_15600
            # print("process file:",pre_file_name)
            if not os.path.exists(dir_pred):
                os.mkdir(dir_pred)

            last_file_name = file_name.split('/')[-1].replace('.jpg', '.npy')
            last_file_name = last_file_name.split('/')[-1].replace('.png', '.npy')
            temp_file_name = os.path.join(dir_pred, last_file_name)
            temp_file_name = temp_file_name.replace('.npy','.png')
            pre_file = last_file_name.split('/')[-1].replace('.npy', '')
            with torch.no_grad():
                result,assgined_result = model(return_loss=False, **data)
            
            if assgined_result is None:
                continue
            # print("resulting:")
            img_tensor = data['img'][0][0]
            # print("img_tensor:",img_tensor.shape)
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]

                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                # print("out_file:",out_file)
                # print("img_show:",img_show.shape)
                # print('assgined_result',assgined_result.shape)
                # print('assgined_result',result.shape)

                b,t,num_clusters,_ = assgined_result.shape #[b,t,num_clusters,n]
                target_assgined_result = assgined_result[:,-1,:,:]
                assgined_result_i = target_assgined_result
                assgined_result_maps = torch.sigmoid(assgined_result_i)
                # print("assgined_result_maps:",assgined_result_maps)
                H_m = assgined_result_maps.view(b,num_clusters,60,108)
                mean_scores = H_m.mean(dim=(2, 3), keepdim=True)  # [b,num_clusters,1,1] softmax是沿着空间维度的
                
                binary_masks = (H_m > mean_scores).float()  # 生成二值掩码 
                # print(binary_masks)
                upsampled_masks = F.interpolate(binary_masks, size=(ori_h, ori_w), mode='nearest').cpu()
                # # 根据mean_scores 按照从大到小对channel进行排序
                # sorted_mean_scores, sorted_indices = torch.sort(mean_scores, dim=1, descending=True)
                # # sorted_binary_masks = binary_masks.gather(1, sorted_indices)
                # upsampled_masks = upsampled_masks.gather(1, sorted_indices).cpu()

                # 可视化结果
                # num_shows = num_clusters
                width = 15
                # length =  num_clusters // width + 1
                length = 15
                
                img_show = np.array(img_show)
                # print("array_out",img_show.shape)
                fig, axes = plt.subplots(length, width, figsize=(50, 50))
                axes[0,0].imshow(img_show)
                axes[0,0].set_title(f"Original Image")
                axes[0,0].axis('off')
                # 可视化每个cluster的结果
                for i in range(num_clusters//width+1):
                    if i == 0:
                        j=1
                    else:
                        j=0

                    while j < width and i*width+j < num_clusters:
                        visualization = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                        visualization[:, :, 0] = (upsampled_masks[0, i*width+j].numpy() * 255).astype(np.uint8)  # Red channel for important regions
                        visualization[:, :, 2] = ((1 - upsampled_masks[0, i*width+j].numpy()) * 255).astype(np.uint8)  # Blue channel for less important regions
                        visualization = overlay_mask(img_show, visualization, alpha=0.5)
                        visualization.save(os.path.join(dir_pred, pre_file+'_'+str(i*width+j-1)+'.png'))
                        axes[i % length][j].imshow(np.array(visualization))
                        # axes[i % length][j].set_title(f"scores:{mean_scores[0, i*width+j].item():.2f}")
                        axes[i % length][j].axis('off')
                        j += 1

                plt.savefig(os.path.join(dir_pred, pre_file+'_'+'.png'))
                plt.show()


                # for m in range(1,num_shows):
                #     # 可视化二值掩码
                #     # 可视化重要区域（红色）和不重要区域（蓝色）
                #     visualization = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
                #     visualization[:, :, 0] = (upsampled_masks[0, m].numpy() * 255).astype(np.uint8)  # Red channel for important regions
                #     visualization[:, :, 2] = ((1 - upsampled_masks[0, m].numpy()) * 255).astype(np.uint8)  # Blue channel for less important regions
                #     axes[m].imshow(visualization)
                #     axes[m].set_title(f"{m},scores:{mean_scores[0, m].item():.2f}")
                #     axes[m].axis('off')
                # plt.savefig(temp_file_name)
                # plt.show()


        # if not isinstance(data['img'][0], list):
        #     batch_size = data['img'][0].size(0)
        # else:
        #     batch_size = data['img'][0][0].size(0)
        # # print(data['img'][0].shape)
        # for _ in range(batch_size):
        #     prog_bar.update()
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        # shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
