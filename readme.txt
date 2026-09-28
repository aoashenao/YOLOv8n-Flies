YOLOv8n-Flies

A lightweight detection model for automated house fly (Musca domestica) monitoring on sticky traps under variable outdoor illumination, built on YOLOv8n with a Contrast Guided Block (CGB), a P3/P4 dual-scale detection head， and the WIoUv3 loss.

Paper: [under review]

[Model Structure](docs/model_structure.png)

Performance

| Model | Precision/% | Recall/% | mAP@0.5/% | Params/M | FLOPs/G | Weight/MB |
| YOLOv8n (baseline) | 97.9 | 94.2 | 96.0 | 3.0110 | 8.1942 | 5.97 |
| YOLOv8n-Flies*| 97.1 | 94.0 | 96.3 | 0.7428| 5.0020 | 1.64 |

Robust across natural illumination of 200–20,000 lx; ~7.7 FPS on a Raspberry Pi 5 (CPU-only) via ONNX.

Usage
bash
git clone https://github.com/aoashenao/YOLOv8n-Flies.git
cd YOLOv8n-Flies
pip install -r requirements.txt

python train.py        # 200 epochs, batch 16, imgsz 640, SGD, seed 42 (see paper)
python val.py          # evaluation
python export_onnx.py  # ONNX export for Raspberry Pi 5 deployment
python generate_hot.py # HI-Res-CAM activation maps
```

Pre-trained weights (YOLOv8n-Flies.pt / .onnx / .yaml) are in [models](./models).

 Data
[sample_data](./sample_data) provides representative annotated images (site identifiers removed). The full dataset is available from the corresponding author upon reasonable request (see the paper's Data Availability Statement).
Citation

bibtex
@article{jin2026yolov8nflies,
  title   = {A Lightweight House Fly Detection Model for Outdoor Monitoring:
             Algorithm Optimization and Field Validation under Variable Illumination},
  author  = {Jin, Jiaao and Dai, Bo and Peng, Heng and Dong, Haowei and Yang, Anjie and Jin, Tao},
  journal = {Insects},
  year    = {2026},
  note    = {under review}
}

 License

Contact: Tao Jin (jintao@usst.edu.cn)