# YOLOv8n-Flies
Lightweight house fly (Musca domestica) detection model for outdoor sticky-trap monitoring, based on YOLOv8n.
 Performance

| Model | Precision/% | Recall/% | mAP@0.5/% | Params/M | FLOPs/G | Weight/MB |
|---|---|---|---|---|---|---|
| YOLOv8n (baseline) | 97.9 | 94.2 | 96.0 | 3.0110 | 8.1942 | 5.97 |
| YOLOv8n-Flies| 97.1 | 94.0 | 96.3| 0.7428| 5.0020 | 1.64 |

 Robust across natural illumination of 200–20,000 lx; ~7.7 FPS on a Raspberry Pi 5 (CPU-only) via ONNX.
 Field validated at three outdoor sites; four months of continuous monitoring captured diurnal/seasonal patterns.

Repository Structure

```
YOLOv8n-Flies/
├── docs/                  # model structure, HI-Res-CAM maps, training/validation visualizations
├── models/                # trained weights and model configuration
├── sample_data/           # representative field images (site identifiers removed)
│   ├── contrast_test/     # examples for contrast/activation visualization
│   └── light_test/        # low/ | medium/ | high/ natural illumination examples
├── train.py               # model training
├── val.py                 # model evaluation
├── export_onnx.py         # ONNX export for edge deployment
└── generate_hot.py        # HI-Res-CAM activation map generation
```

## Usage

```bash
git clone https://github.com/aoashenao/YOLOv8n-Flies.git
cd YOLOv8n-Flies
pip install -r requirements.txt

python train.py        # 200 epochs, batch 16, imgsz 640, SGD, seed 42 (see paper)
python val.py          # evaluation
python export_onnx.py  # ONNX export for Raspberry Pi 5 deployment
python generate_hot.py # HI-Res-CAM activation maps (examples in docs/)
```

## Data

Pre-trained weights (`.pt` / `.onnx`) and the model configuration (`.yaml`) are provided in [`models/`](./models).

[`sample_data/`](./sample_data) provides representative sticky-trap images collected under different natural illumination conditions (site identifiers removed). The complete field dataset is available from the corresponding author upon reasonable request (see the paper's Data Availability Statement).

## Citation

```bibtex
@article{jin2026yolov8nflies,
  title   = {A Lightweight House Fly Detection Model for Outdoor Monitoring:
             Algorithm Optimization and Field Validation under Variable Illumination},
  author  = {Jin, Jiaao and Dai, Bo and Peng, Heng and Dong, Haowei and Yang, Anjie and Jin, Tao},
  journal = {Insects},
  year    = {2026},
  note    = {under review}
}
```

## License

[MIT](./LICENSE) · Contact:  Tao Jin (jintao@usst.edu.cn)
