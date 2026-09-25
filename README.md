# 人参分级检测系统 (Ginseng Grade Detection)

基于 YOLOv8 + PyQt5 的桌面端目标检测系统，对人参按外观进行等级分类。

## 检测类别

| ID | 类别 |
|----|------|
| 0  | 特级 |
| 1  | 一级 |
| 2  | 二级 |
| 3  | 三级 |

## 技术栈

- Python 3.9
- PyQt5（桌面界面）
- Ultralytics YOLOv8（目标检测）
- OpenCV（图像处理）

## 项目结构

```
.
├── MainProgram.py        # 主程序入口
├── Config.py             # 检测类别与模型路径配置
├── train.py              # 模型训练脚本
├── detect_tools.py       # 检测工具函数
├── CameraTest.py         # 摄像头实时检测
├── VideoTest.py          # 视频文件检测
├── imgTest.py            # 单张图片检测
├── UIProgram/            # 界面代码与资源
├── Font/                 # 字体资源
├── datasets/             # 数据集（不入库，需自行准备）
├── runs/                 # 训练产物（不入库）
└── requirements.txt
```

## 环境配置

```bash
conda create -n py39 python=3.9
conda activate py39
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 训练用 GPU 版 PyTorch 需根据本机 CUDA 版本单独安装 torch（见 https://pytorch.org）。

## 运行

```bash
python MainProgram.py
```

启动后进入登录界面，登录后在主界面可选择图片 / 视频 / 摄像头进行检测。

## 模型训练

1. 准备数据集放入 `datasets/images/{train,val}`，标签对应 `datasets/labels/`，并按 `datasets/data.yaml` 的结构配置路径。
2. 运行：

```bash
python train.py
```

训练结果保存在 `runs/detect/`，最优权重在 `runs/detect/train/weights/best.pt`。

## 说明

- `runs/`、`datasets/`、`Four/`、`*.pt`、`*.mp4`、`*.zip` 已在 `.gitignore` 中排除，clone 后需自行准备数据集与预训练权重 `yolov8n.pt`（可从 https://github.com/ultralytics/assets/releases 下载）。