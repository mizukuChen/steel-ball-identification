# 钢珠识别项目

## 项目目标
识别画面中钢珠的位置、大小、数量。先在PC上跑通，后续移植到 K230 / 树莓派。
计划两套识别方案：传统CV（尚未开始）+ YOLO（合成数据训练流程已跑通）。

---

## 目录结构

```
steel ball identification/
├── data/
│   ├── raw_photos/          手机拍摄的原始钢珠照片（11张，白色编织布背景）
│   ├── templates/           从原始照片提取的钢珠模板（透明PNG，11个）
│   │   └── _debug/          提取时的圆检测可视化（人工核验用）
│   ├── backgrounds/
│   │   ├── downloaded/      从 picsum.photos 下载的真实场景背景（40张）
│   │   └── generated/       程序生成的噪音/纹理背景（40张）
│   ├── synthetic/           合成训练数据集
│   │   ├── images/train/    1700张
│   │   ├── images/val/      300张
│   │   ├── labels/          YOLO格式标注（与images同结构）
│   │   └── data.yaml        YOLO训练用的数据集配置
│   └── real_world/          （预留）后续实际部署场景采集的真实数据，用于微调
│
├── src/
│   └── data_pipeline/
│       ├── extractor.py             钢珠提取（HoughCircles检测圆+抠图生成透明PNG）
│       ├── background_generator.py  程序化背景生成（纯色/噪音/渐变/云纹/条纹/棋盘格）
│       └── synthesizer.py           合成数据核心逻辑（贴合+光影匹配+阴影+YOLO标注生成）
│
├── scripts/                  命令行入口，调用 src/ 里的逻辑
│   ├── run_extract.py             提取钢珠模板
│   ├── run_generate_backgrounds.py 生成程序化背景
│   ├── run_synthesize.py          生成合成数据集
│   └── run_train.py               YOLO训练
│
├── models/
│   ├── yolo11n.pt            YOLO11 nano 预训练权重（首次运行自动下载的缓存）
│   └── runs/steelball_test/  已有的2-epoch测试训练结果（见下文"已完成的训练测试"）
│
├── configs/                  （预留，暂未使用）
├── tests/                    （预留，暂未使用）
└── requirements.txt
```

---

## 环境搭建（新机器）

### 1. Python环境
本项目在 conda 环境 `opencv-nuedc` 下开发（Python 3.12）。新机器建议新建一个干净环境：

```bash
conda create -n steelball python=3.12 -y
conda activate steelball
```

### 2. 安装依赖

```bash
pip install opencv-python numpy ultralytics
```

**重要：torch/GPU 要单独处理**

`requirements.txt` 里锁定的 `torch==2.13.0` 是本机（无独显）装到的 **CPU版本**（`torch-2.13.0+cpu`）。
如果新机器有 NVIDIA 显卡，**不要**直接 `pip install -r requirements.txt`，而是去 [pytorch.org](https://pytorch.org/get-started/locally/) 按你的 CUDA 版本获取对应安装命令，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

装完用下面命令确认GPU被识别：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 数据管线各步骤（如需重新生成/调整参数）

按顺序执行，均从项目根目录运行：

```bash
# 1. 从 data/raw_photos 提取钢珠模板 -> data/templates/
python scripts/run_extract.py

# 2. 生成程序化噪音/纹理背景 -> data/backgrounds/generated/
python scripts/run_generate_backgrounds.py

# 3. 合成训练数据集（默认2000张，可改 run_synthesize.py 里 main() 的 num_images 参数）
python scripts/run_synthesize.py
```

`data/backgrounds/downloaded/` 里的40张真实背景是用 `curl` 从 picsum.photos 批量下载的（无版权免费图库），不属于代码管线的一部分，新机器如果需要重新下载可以直接复用现有文件（已经在仓库/项目目录里，无需重新下载）。

---

## YOLO 训练

```bash
python scripts/run_train.py --epochs 3 --model n --batch 16
```

参数说明（见 `scripts/run_train.py`）：
- `--model`：n/s/m/l/x（nano到extra-large，n最快）
- `--imgsz`：默认640，与合成数据集分辨率一致
- `--batch`：默认16，GPU显存够的话可以调大

训练结果保存在 `models/runs/<name>/`，权重在 `weights/best.pt`。

### 已完成的训练测试（本机CPU，仅供速度参考）

在无独显的机器上（Intel Core Ultra 7 155H，纯CPU），跑了2个epoch作为流程验证：

| 项目 | 结果 |
|------|------|
| 硬件 | CPU only（torch-2.13.0+cpu），无NVIDIA显卡 |
| 单epoch耗时 | 约6.25分钟（1700张训练图，batch=16） |
| 2 epoch总耗时 | 0.209小时 ≈ 12.5分钟 |
| 推理速度 | 53.2ms/张（约19 FPS，CPU） |
| mAP50 (epoch2) | 0.995 |
| mAP50-95 (epoch2) | 0.819 |

**⚠️ 精度数字仅作流程验证参考，不代表真实场景效果**：合成数据集背景/模板来源有限（80张背景+11个模板反复组合），模型很容易过拟合到"认识这几种背景/模板"，2个epoch就到 mAP50=0.995 是不正常的高。**后续必须用真实场景数据做微调验证**，不能只看这个数字判断模型好坏。

新机器上换GPU训练后，可以对比这个CPU基准算出加速比，并跑更多epoch（比如50-100）看真实收敛效果。

---

## 待办 / 下一步

- [ ] GPU机器上跑完整训练（50-100 epoch），观察是否过拟合
- [ ] 收集少量真实场景照片（20-50张），标注后用于微调（`data/real_world/`）
- [ ] 传统CV方案（霍夫圆检测等），作为YOLO的对比baseline
- [ ] 两套方案效果对比
- [ ] 移植到 K230（模型需用 nncase 转换为 .kmodel，配合K230 SDK重写推理部分）/ 树莓派（可直接pip部署，可能需要量化优化速度）
