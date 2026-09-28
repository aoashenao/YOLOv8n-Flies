#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-8.2.77
@File    ：start_val.py
@IDE     ：PyCharm
@Author  ：肆十二（付费咨询QQ: 3045834499） 粉丝可享受99元调试服务
@Description  ：TODO 添加文件描述
@Date    ：2024/8/15 15:15
'''
from ultralytics import YOLO

# 加载自己训练好的模型，填写相对于这个脚本的相对路径或者填写绝对路径均可
# model = YOLO("runs/detect/yolov8n/weights/best.pt")
# model = YOLO("runs/detect/yolov8-flies-v1/weights/best.pt")
# model = YOLO("runs/ablation/exp1_ghost/weights/best.pt")
# 开始进行验证，验证的数据集为'A_my_data.yaml'，图像大小为640，批次大小为4，置信度分数为0.25，交并比的阈值为0.6，设备为0，关闭多线程（windows下使用多线程加载数据容易出现问题）
model = YOLO("runs/detect/V6_CG_All_P3P4V2/weights/best.pt")
validation_results = model.val(
    data='changelight_data.yaml',
    imgsz=640,
    batch=4,
    conf=0.25,
    iou=0.6,
    device="0",
    workers=0,
    split='test',
    plots=True,
    project='runs/3_light_change',   # 保存的父目录
    name='Flies_test_high1.4'     # 保存的子文件夹名称
)