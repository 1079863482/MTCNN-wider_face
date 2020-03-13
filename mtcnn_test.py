import cv2
from mtcnn.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.vision import vis_face
import torch


if __name__ == '__main__':


    pnet, rnet, onet = create_mtcnn_net(p_model_path=r"./model_store/pnet_epoch.pt", r_model_path=r"./model_store/rnet_epoch.pt", o_model_path=r"./model_store/onet_epoch.pt", use_cuda=False)

    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("test_img/test1.jpg")
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         #将图片转化为RGB格式

    bboxs, landmarks = mtcnn_detector.detect_face(img)               #将图片传入人脸侦测模块中，返回人脸框坐标和五个关键点坐标
    save_name = 'timg_.jpg'                                  #保存图片
    vis_face(img_bg,bboxs,landmarks, save_name)            #将返回的信息在图上画出
