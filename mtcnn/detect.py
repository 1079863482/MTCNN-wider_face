import cv2
import time
import numpy as np
import torch
from torch.autograd.variable import Variable
from mtcnn.models import PNet,RNet,ONet
import mtcnn.utils as utils
import mtcnn.image_tools as image_tools


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):
    """
    模型加载，默认使用cpu，正常使用GPU

    """

    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if(use_cuda):
            print('p_model_path:{0}'.format(p_model_path))
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()
        else:
            pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            print('r_model_path:{0}'.format(r_model_path))
            rnet.load_state_dict(torch.load(r_model_path))
            rnet.cuda()
        else:
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            print('o_model_path:{0}'.format(o_model_path))
            onet.load_state_dict(torch.load(o_model_path))
            onet.cuda()
        else:
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

    return pnet,rnet,onet




class MtcnnDetector(object):
    """
        P,R,O net face detection and landmarks align
        P,R,O 网络人脸检测和关键点识别
    """
    def  __init__(self,
                 pnet = None,
                 rnet = None,
                 onet = None,
                 min_face_size=12,       #可缩小的最短边长
                 stride=2,        #步长
                 threshold=[0.6, 0.7, 0.7],     #各个网络阈值
                 scale_factor=0.709       #缩放比例
                 ):

        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride=stride
        self.thresh = threshold
        self.scale_factor = scale_factor


    def square_bbox(self, bbox):
        """
            convert bbox to square
            将识别框转化为正方形

        """
        square_bbox = bbox.copy()         #复制

        if bbox.size == 0 :
            return np.array([[]])

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h,w)

        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - l*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - l*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1

        return square_bbox


    def generate_bounding_box(self, cls, off, scale, threshold):
        """
            选出置信度大于阈值的框并进行反算

            返回建议框的四个坐标+偏移量+一个置信度
        """
        stride = 2         # 步长
        cellsize = 12       # 建议框大小

        t_index = np.where(cls > threshold)        #取出大于阈值的索引

        if t_index[0].size == 0:                #如果索引数量为0，返回空数组
            return np.array([])


        # 选择大于阈值的边框
        dx1, dy1, dx2, dy2 = [off[0, t_index[0], t_index[1], i] for i in range(4)]

        off = np.array([dx1, dy1, dx2, dy2])            #得到边框，加入数组中

        score = cls[t_index[0], t_index[1], 0]          #取出大于阈值的置信度

        # hence t_index[1] means column, t_index[1] is the value of x
        # hence t_index[0] means row, t_index[0] is the value of y
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),            # x1 of prediction box in original image 建议框x
                                 np.round((stride * t_index[0]) / scale),            # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale), # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale), # y2 of prediction box in original image
                                                                                     # reconstruct the box in original image
                                 score,
                                 off
                                 ])
        return boundingbox.T


    def resize_image(self, img, scale):
        """"
        缩放图像
        """
        height, width, channels = img.shape
        new_height = int(height * scale)     # resized new height
        new_width = int(width * scale)       # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
        return img_resized


    def pad(self, bboxes, w, h):
        """
        填充
        传入：p网络框，原图的w，h
        """
        #p网络输出的框四个坐标值
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        #对坐标越界的框进行重新定义
        tmp_index = np.where(x2 > w-1)           #选出右下角x坐标大于边长的框索引
        x2[tmp_index] = w - 1         #x2坐标变为最大边长w的坐标

        tmp_index = np.where(y2 > h-1)          #选出右下角y坐标大于边长的框索引
        y2[tmp_index] = h - 1       #y2坐标变为最大边长h的坐标

        tmp_index = np.where(x1 < 0)          #选出左上角x坐标小于0的框索引
        x1[tmp_index] = 0

        tmp_index = np.where(y1 < 0)            #选出左上角y坐标小于0的框索引
        y1[tmp_index] = 0

        return_list = [y1, y2, x1, x2]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list


    def detect_pnet(self, im):
        """
        p网络侦测
        :param im: 传入侦测图片
        :return: 返回有目标区域的框坐标
        """
        net_size = 12          #可缩小的最短边长

        current_scale = float(net_size) / self.min_face_size    # 自定义缩放比例
        print('imgshape:{0}, current_scale:{1}'.format(im.shape, current_scale))
        im_resized = self.resize_image(im, current_scale) # 按照自定义的初始缩放比例进行缩放
        current_height, current_width, _ = im_resized.shape     #获取图片边长

        all_boxes = list()                   #创建一个列表，用来存放框的坐标
        while min(current_height, current_width) > net_size:         #当图片小于最短边时停止

            image_tensor = image_tools.convert_image_to_tensor(im_resized).unsqueeze(0)     #转化为tersor
            feed_imgs = Variable(image_tensor)         #转化为变量

            # if self.pnet_detector.use_cuda:          #是否使用cuda
            #     feed_imgs = feed_imgs.cuda()

            with torch.no_grad():
                cls_map, reg = self.pnet_detector(feed_imgs)         #前向计算

            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())     #换轴 0，2，3，1
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg.cpu())          #换轴 0，2，3，1

            #返回反算后的框（四个建议框的坐标，一个置信度，四个偏移量）
            boxes = self.generate_bounding_box(cls_map_np[ 0, :, :], reg_np, current_scale, self.thresh[0])
            # 生成图像金字塔
            current_scale *= self.scale_factor # 乘以缩放比例 0.709
            im_resized = self.resize_image(im, current_scale)     #缩放图片
            current_height, current_width, _ = im_resized.shape      #得到新的的宽高

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')       #返回nms后的框索引
            boxes = boxes[keep]              #取出框

            all_boxes.append(boxes)          #将nms后的框加入列表中
            # i+=1

        if len(all_boxes) == 0:
            return np.array([[]])

        all_boxes = np.vstack(all_boxes)              #将所有的框的信息整合

        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')          #再做一次nms
        all_boxes = all_boxes[keep]                #取出第二次nms后的框

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1          # W
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1          # H


        #使用建议框的坐标个偏移量进行反算
        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([align_topx,align_topy,align_bottomx,align_bottomy,all_boxes[:, 4],])
        # boxes_align = boxes
        boxes_align = boxes_align.T

        return boxes_align

    def detect_rnet(self, im, p_boxes):
        """
        R网络侦测
        :param im: 侦测原图
        :param dets: P网络输出的框
        :return: R网络侦测后保留的框
        """
        h, w, c = im.shape             #获取原图的大小

        if p_boxes.size == 0:             #判断是否为空
            return np.array([]),np.array([])

        p_boxes = self.square_bbox(p_boxes)     #将p网络输出的框转化为正方形框
        # rounds
        p_boxes[:, 0:4] = np.round(p_boxes[:, 0:4])        #转化为数组

        [y1, y2, x1, x2] = self.pad(p_boxes, w, h)         #对超出原图的框进行处理
        num_boxes = p_boxes.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            p_img = im[y1[i]:y2[i]+1, x1[i]:x2[i]+1]                #在原图上裁剪出来
            crop_im = cv2.resize(p_img, (24, 24))                 #resize成24*24
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)           #转化为张量

            cropped_ims_tensors.append(crop_im_tensor)                      #加入列表中
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        # if self.rnet_detector.use_cuda:         #是否使用cuda
        #     feed_imgs = feed_imgs.cuda()

        with torch.no_grad():
            cls_map, off = self.rnet_detector(feed_imgs)              #r网络前向计算

        cls_map = cls_map.cpu().data.numpy()
        off = off.cpu().data.numpy()


        keep_inds = np.where(cls_map > self.thresh[1])[0]             #取出置信度大于阈值的索引

        if len(keep_inds) > 0:
            boxes = p_boxes[keep_inds]            #R网络的建议框
            cls = cls_map[keep_inds]
            off = off[keep_inds]
        else:
            return np.array([[]])

        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return np.array([[]])

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]            #剩余的建议框
        keep_off = off[keep]

        #
        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        #用R网络建议框的坐标进行实际框反算
        align_topx = keep_boxes[:,0] + keep_off[:,0] * bw
        align_topy = keep_boxes[:,1] + keep_off[:,1] * bh
        align_bottomx = keep_boxes[:,2] + keep_off[:,2] * bw
        align_bottomy = keep_boxes[:,3] + keep_off[:,3] * bh

        boxes_align = np.vstack([align_topx,               #加入数组
                               align_topy,
                               align_bottomx,
                               align_bottomy,
                               keep_cls[:, 0],
                             ])

        boxes_align = boxes_align.T

        return boxes_align

    def detect_onet(self, im, r_boxes):

        h, w, c = im.shape              #获取原图大小

        if r_boxes[0].size == 0:
            return np.array([]),np.array([])

        r_boxes = self.square_bbox(r_boxes)            #将R网络输出的框转化为正方形
        r_boxes[:, 0:4] = np.round(r_boxes[:, 0:4])       #转化为数组

        [ y1, y2, x1, x2] = self.pad(r_boxes, w, h)           #返回处理后的所有框坐标
        num_boxes = r_boxes.shape[0]          #得到总框数

        cropped_ims_tensors = []
        for i in range(num_boxes):

            r_img = im[y1[i]:y2[i] + 1, x1[i]:x2[i] + 1, :]            #原图裁剪
            crop_im = cv2.resize(r_img, (48, 48))                #resize成48*48
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)           #转化为张量
            # cropped_ims_tensors[i, :, :, :] = crop_im_tensor
            cropped_ims_tensors.append(crop_im_tensor)                    #加入列表
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        # if self.rnet_detector.use_cuda:        #是否使用cuda
        #     feed_imgs = feed_imgs.cuda()
        with torch.no_grad():
            cls_map, off, landmark = self.onet_detector(feed_imgs)          #O网络前向计算，返回置信度、两个点坐标偏移量、五个关键点偏移量

        cls_map = cls_map.cpu().data.numpy()
        off = off.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]                  #取出置信度大于阈值的索引

        if len(keep_inds) > 0:
            boxes = r_boxes[keep_inds]              #得到大于阈值的O网络建议框
            cls = cls_map[keep_inds]
            off = off[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return np.array([]),np.array([])

        keep = utils.nms(boxes, 0.7, mode="Minimum")            #NMS计算，去除重复框

        if len(keep) == 0:
            return np.array([]),np.array([])

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]                #得到剩余的建议框
        keep_reg = off[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1


        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw              #反算得到实际框坐标
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]                      #得到建议框左上角坐标的x，y坐标
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ])

        boxes_align = boxes_align.T                  #实际框两点坐标

        landmark =  np.vstack([
                                 align_landmark_topx + keep_landmark[:, 0] * bw,            #根据建议框左上角坐标对关键点在原图位置进行反算
                                 align_landmark_topy + keep_landmark[:, 1] * bh,
                                 align_landmark_topx + keep_landmark[:, 2] * bw,
                                 align_landmark_topy + keep_landmark[:, 3] * bh,
                                 align_landmark_topx + keep_landmark[:, 4] * bw,
                                 align_landmark_topy + keep_landmark[:, 5] * bh,
                                 align_landmark_topx + keep_landmark[:, 6] * bw,
                                 align_landmark_topy + keep_landmark[:, 7] * bh,
                                 align_landmark_topx + keep_landmark[:, 8] * bw,
                                 align_landmark_topy + keep_landmark[:, 9] * bh,
                                 ])

        landmark_align = landmark.T     #五个关键点坐标

        return boxes_align, landmark_align


    def detect_face(self,img):
        """
        人脸侦测主函数

        """

        t = time.time()            #计算时间

        # pnet侦测
        # if self.pnet_detector:
        boxes_P = self.detect_pnet(img)          #返回p网络输出框
        if boxes_P is None:
            return np.array([]),np.array([])

        t1 = time.time() - t
        t = time.time()

        # rnet侦测
        # if self.rnet_detector:
        boxes_R = self.detect_rnet(img, boxes_P)           #返回R网络输出框
        if boxes_R is None:
            return np.array([]),np.array([])

        t2 = time.time() - t
        t = time.time()

        # onet侦测
        # if self.onet_detector:
        boxes_O, landmark_O = self.detect_onet(img, boxes_R)          #返回两个点坐标和五个关键点坐标
        if boxes_O is None:
            return np.array([]),np.array([])

        t3 = time.time() - t
        t = time.time()
        print("time " + '{0}'.format(t1+t2+t3) + '  p_net {0}  r_net {1}  o_net {2}'.format(t1, t2, t3))

        return boxes_O, landmark_O
