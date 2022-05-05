import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import cv2
import numpy as np
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective


class InferenceEngine(object):
    def __init__(self,
                 weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
                 source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
                 data=ROOT / 'data/coco128.yaml',  # dataset.yaml path 验证数据集？
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 view_img=False,  # show results
                 save_txt=False,  # save results to *.txt
                 save_conf=False,  # save confidences in --save-txt labels
                 save_crop=False,  # save cropped prediction boxes
                 nosave=False,  # do not save images/videos
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 visualize=False,  # visualize features
                 update=False,  # update all models
                 project=ROOT / 'runs/detect',  # save results to project/name
                 name='exp',  # save results to project/name
                 exist_ok=False,  # existing project/name ok, do not increment
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=False,  # use FP16 half-precision inference
                 dnn=False,  # use OpenCV DNN for ONNX inference
                 ):
        self.weights=weights
        self.source=source
        self.data=data
        self.imgsz=imgsz
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.device=device
        self.view_img=view_img
        self.save_txt=save_txt
        self.save_conf=save_conf
        self.save_crop=save_crop
        self.nosave=nosave
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.augment=augment
        self.visualize=visualize
        self.update=update
        self.project=project
        self.name=name
        self.exist_ok=exist_ok
        self.line_thickness=line_thickness
        self.hide_labels=hide_labels
        self.hide_conf=hide_conf
        self.half=half
        self.dnn=dnn

        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=dnn, data=data)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = \
            self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine

        half &= (self.pt or self.jit or self.onnx or self.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if half else self.model.model.float()

    def detect(self, img):
        # Dataloader
        # 载入数据
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride, auto=self.pt)

        # Run inference
        # 开始预测
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        # 对图片进行处理
        im0 = img
        # Padded resize
        im = letterbox(im0, self.imgsz, self.stride, auto=self.pt)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # 预测
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        # 用于存放结果
        detections = []

        # Process predictions
        for i, det in enumerate(pred):  # per image 每张图片
            seen += 1
            # im0 = im0s.copy()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                            xywh[3]]  # 检测到目标位置，格式：（left，top，w，h）

                    cls = self.names[int(cls)]
                    conf = float(conf)
                    detections.append({'class': cls, 'conf': conf, 'position': xywh})
        # # 输出结果
        # for i in detections:
        #     print(i)

        # # 推测的时间
        # LOGGER.info(f'({t3 - t2:.3f}s)')
        return detections


if __name__ == "__main__":
    engine = InferenceEngine()
    image = cv2.imread("data/images/bus.jpg")
    result = engine.detect(image)
    print(result)


