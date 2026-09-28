#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77
@File    ：start_train.py
@IDE     ：PyCharm
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：悬链主启动程序
@Date    ：2024/8/15 15:14
'''
import time
from ultralytics import YOLO


# yolov8n模型训练：训练模型的数据为'A_my_data.yaml'，轮数为100，图片大小为640，设备为本地的GPU显卡，关闭多线程的加载，图像加载的批次大小为4，开启图片缓存
#model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov5su.pt')  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.pt")
# results = model.train(data='A_my_data.yaml', epochs=200, imgsz=640, device=[0], workers=0, batch=4, cache=True)  # GPU开始训练
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="V6_dataset.yaml",
    epochs=200,
    patience=50,
    imgsz=640,
    batch=4,
    mosaic=0,
    mixup=0,
    copy_paste=0,
    amp=False,
    workers=0,
    device=[0],
    hsv_h=0,  # 关掉 HSV 增强
    hsv_s=0,
    hsv_v=0,
    plots=True,
    nbs=16,          # 梯度累积，等效 batch=16，改善训练稳定性
    name=f"Baseline_yolov8n.find"
)
# todo A_my_data.yaml请切换为你本地的绝对路径，如果是本地的绝对路径，请填写绝对路径，如：'D:/my_data/A_my_data.yaml'

time.sleep(10) # 睡眠10s，主要是用于服务器多次训练的过程中使用


