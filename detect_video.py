
import numpy as np
from mtcnn.detect import create_mtcnn_net, MtcnnDetector
# from New_detect import create_mtcnn_net, MtcnnDetector
from mtcnn.vision import vis_face
import cv2


def detect_video(video_path, detector):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vout = cv2.VideoWriter()

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            out_bbx,_ = detector.detect_face(frame)
            for box in out_bbx:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)
                # for i in range(0, 10, 2):
                #     cv2.circle(frame, (int(box[4 + i]), int(box[5 + i])), 3, (0, 255, 0))

            cv2.imshow('img', frame)
            # vout.write(frame)

            # cv2.waitKey(int(1000 / fps))
            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./model_store/pnet_epoch.pt",
                                        r_model_path="./model_store/rnet_epoch.pt",
                                        o_model_path="./model_store/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=48)

    # img = cv2.imread("test_img/timg3.jpg")
    # img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    # bboxs, landmarks = mtcnn_detector.detect_face(img)


    video_path = r'X:\mtcnn-pytorch\test_video\video.mp4'

    detect_video(video_path,mtcnn_detector)

