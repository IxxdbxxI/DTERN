import numpy as np
import os
from PIL import Image
#from utils import Evaluator
import sys

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def beforeval(self):
        isval = np.sum(self.confusion_matrix,axis=1)>0
        self.confusion_matrix = self.confusion_matrix*isval

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc


    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        isval = np.sum(self.confusion_matrix,axis=1)>0
        MIoU = np.nansum(MIoU*isval)/isval.sum()
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        #print(mask)
        #print(gt_image.shape)
        #print(gt_image[mask])
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#        print(label.shape)
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



def get_common(list_,predlist,clip_num,h,w,evaluator_VI8):
    accs = []
    for i in range(len(list_)-clip_num):
        global_common = np.ones((h,w))
        predglobal_common = np.ones((h,w))
        for j in range(1,clip_num): #连续预测
            common = (list_[i] == list_[i+j])
            global_common = np.logical_and(global_common,common)
            pred_common = (predlist[i]==predlist[i+j])
            
            predglobal_common = np.logical_and(predglobal_common,pred_common)
        


        #处理应该连续但是不连续的地方被错误置0
        no_pred = np.logical_and(predglobal_common == 0,global_common==1)
        # 修改成预测错误的地方
        predlist[i][no_pred] = (list_[i][no_pred] + 1) % 124
        predglobal_common = np.logical_or(predglobal_common,global_common)
        assert  np.all(np.equal(np.logical_or(predglobal_common,global_common), np.logical_or(no_pred,predglobal_common)))
        pred_union = predlist[i] * predglobal_common

        list_[i][list_[i]==0] = 128
        gt_union = list_[i] * global_common
        gt_union[gt_union == 0] = 255
        gt_union[gt_union==128] = 0
        evaluator_VI8.add_batch(gt_union[None,:], pred_union[None,:])
        # 计算iou
    return evaluator_VI8.Mean_Intersection_over_Union()

        

# VC16 score: 0.3098934444442716 on val.txt set
# VC8 score: 0.2922997535167649 on val.txt set
# **********
# Acc, Acc_class, mIoU, FWIoU:  [0.7469011060594057, 0.5105729649370202, 0.4059576727293892, 0.6162617761615689]
# good_video:  ['2097_HVti7xTm2ow', '1678__qxxSOgqMpc', '1047_CjUyIs7-IXQ', '1163_ayuSarPEafI', '1272_kDLzAZhFEVY', '151_BNFMbfOYTKg', '184_qOLd2_WB7WA', '258_xINh_0D2h_0', '512_5YBRu5JFHmw', '536__FiNAxDdKw8', '540_EF3qZgsyw50', '649_jhvbwbfVQLg', '444_NlIgZLfdpQQ', '748_vzhAXolf72w', '1029_OPfjoAQxvcQ', '516_5bqIhLCjTzE', '60_dpztQl4KzTw', '1265_y-nJktexuuM', '113_QsyPECwiolw', '633_nDt-MoHTzN4', '364_X9zXbEKIF08', '402_s2nluBdo9_U', '2194_---rrlgSY48', '2200_0cRmsADZX80', '2218_3xpH6BQyWwk', '2234_9Yb0WRZGR5E', '2257_GKNG_Ymf_dk', '2263_HR8j5T6ILF8', '2265_I2EYxtMWbj0', '2266_ILJgMfgxDnA', '2269_IpOBwbwg7-s', '2278__h5Riifwwks', '2307_iiBJ5rqlZ58', '2326_UWmbg16ywD8', '2353_s_rXQJoGc1U', '2354_t-vxanB58SQ', '2385_zhpgbTT2w8c', '2387_3502lHaZQIo', '2393_Jtm51h9jGDo', '2369_xGeYj3Kr3Iw']
# mIoU8, mIoU16:  [0.29091934068078584, 0.3107117883558238]

DIR='data/vspw//VSPW_480p'

Pred='/root/workspace/XU/Code/VSS-MRCFA-main/results/Cluster_Segdeformer_124_capa_vit_cluster_block2_srps2_mit_b1'
split = 'val.txt'

with open(os.path.join(DIR,split),'r') as f:
    lines = f.readlines()
    for line in lines:
        videolist = [line[:-1] for line in lines]
total_acc=[]
total_acc_8=[]

clip_num=16
clip_num_8=8

num_class=124    # change this when necessary
evaluator = Evaluator(num_class)
evaluator.reset()
evaluator_video = Evaluator(num_class)
evaluator_video.reset()

evaluator_VI8 = Evaluator(num_class)
evaluator_VI8.reset()

evaluator_VI16 = Evaluator(num_class)
evaluator_VI16.reset()

good_video=[]
no_np_list = []
for video in videolist:
    evaluator_video.reset()
    if video[0]=='.':
        continue
    imglist = []
    predlist = []
    print("video: ", video)
    images = sorted(os.listdir(os.path.join(DIR,'data',video,'mask')))

    if len(images)<=clip_num:
        print("here: ", video)
        continue
    for imgname in images:
        if imgname[0]=='.':
            continue
        img = Image.open(os.path.join(DIR,'data',video,'mask',imgname))
        w,h = img.size
        img = np.array(img)
        ## added by guolei
        img[img==0]=255
        img = img-1
        img[img==254]=255

        # pred = Image.open(os.path.join(Pred,video,imgname))

        # pred = np.array(pred)
        pred_np = os.path.join(Pred,video,imgname.replace('.png','.npy'))
        
        # 这里是bug
        if not os.path.exists(pred_np):
            print("no np: ", pred_np)
            
            no_np_list.append(pred_np)
            continue

        imglist.append(img)

        pred = np.load(pred_np)
        predlist.append(pred)
        evaluator.add_batch(img[None,:], pred[None,:])
        evaluator_video.add_batch(img[None,:], pred[None,:])
        # print(img[None,:].shape, pred[None,:].shape)
    
    if evaluator_video.Mean_Intersection_over_Union()>0.7:
        good_video.append(video)
    accs = get_common(imglist,predlist,clip_num,h,w,evaluator_VI16)
    print(accs)
    accs_8 = get_common(imglist,predlist,clip_num_8,h,w,evaluator_VI8)
    print(accs_8)
    total_acc.append(accs)
    total_acc_8.append(accs_8)
print("no_np_list: ", no_np_list)
Acc = np.array(total_acc)
Acc = np.nanmean(Acc)
Acc_8 = np.array(total_acc_8)
Acc_8 = np.nanmean(Acc_8)
print(Pred)
print('*'*10)
print('VC{} score: {} on {} set'.format(clip_num,Acc,split))
print('VC{} score: {} on {} set'.format(clip_num_8,Acc_8,split))
print('*'*10)

Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
print("Acc, Acc_class, mIoU, FWIoU: ", [Acc, Acc_class, mIoU, FWIoU])


print("good_video: ", good_video)

mIoU8 = evaluator_VI8.Mean_Intersection_over_Union()
mIoU16 = evaluator_VI16.Mean_Intersection_over_Union()
print("mIoU8, mIoU16: ", [mIoU8, mIoU16])