# Reliability-Guided RGB–紅外線融合人體網格重建（HMR_mu_final）

以**凍結的 HMR2.0** 為基礎，從 RGB 影像、熱紅外線（IR）影像，或兩者配對輸入，重建 SMPL 人體網格（姿態 + 體型）。
只訓練三個輕量模組，共 **5.67M 參數**；HMR2.0 的 ViT-H backbone 與 SMPL head（670M 參數）完全不動。

## 方法

Encoder 由兩條**共用同一組凍結 ViT-H 權重**的分支組成：

- **RGB branch**：原封不動的 HMR2.0，只在 RGB+IR 配對輸入時執行，提供每一層的參考特徵。
- **fusion branch**：處理融合後的 token，SMPL decoder 讀取它的最後輸出。

| 模組 | 位置 | 做什麼 | 參數 |
|---|---|---|---|
| **ARG**（Adaptive Reliability Gate） | encoder 之前 | 逐 token 預測 IR 權重 α，以凸組合融合 RGB 與 IR 的 patch token，送進 fusion branch | 1.32M |
| **MA**（Modality Adapter） | fusion branch 後 16 個 block | 有上限的小幅殘差修正，補償凍結 ViT 面對 IR 特徵時的差異 | 2.68M |
| **DCG**（Deep Cross-modal Gate） | fusion branch 第 15/19/23/27/31 個 block 之後 | 讀取 RGB branch 同層特徵 r，逐 token 更新 `x ← x + β(r − x)` | 1.67M |

- ARG 與 DCG 都是凸組合：輸出永遠落在兩組真實特徵之間，不會產生凍結 head 沒看過的數值。
- DCG 的 β 由 `sigmoid(MLP([LN(x), LN(r)]))` 產生，最後一層零初始化、bias 為負，訓練開始時 β ≈ 0.018，等同恆等映射。
- **單一輸入**（RGB-only 或 IR-only）只走 fusion branch 並經過 MA，不經 ARG、DCG，也不執行 RGB branch。
  RGB-only 時 MA 關閉，輸出與原始 HMR2.0 逐位元相同。

程式位置：ARG 在 `models/multimodal/mmped_hmr2_unified_encoder.py`（`per_token_conf_mlp`），
MA 是 `ResidualBodyBlockAdapter`（同檔，程式中沿用舊名 `body_adapters`），
DCG 是 `CrossModalDeepConvexGate`（`models/multimodal/mmped_unified_hmr2.py`，程式中為 `xmodal_deep_adapters`）。

---

## 目錄結構

| 路徑 | 內容 |
|---|---|
| `train.py` | 訓練入口 |
| `tools/eval_2d_mu.py` | LLVIP / OTP2 評估（2D + 相對 pGT 的 3D） |
| `tools/demo_mu*.py` | 網格 / 骨架視覺化 |
| `tools/export_release_weights.py` | 把訓練 checkpoint 轉成只含可訓練模組的發佈權重 |
| `tools/generate_thermal_smplify_pgt_v5.py` | 產生熱影像 pGT（凍結檔，勿修改） |
| `scripts/train_main.sh`、`scripts/eval_main.sh` | 五階段訓練、評估發佈權重 |
| `configs/train/main/` | 五個訓練階段的設定檔 |
| `configs/paths/` | 機器本地路徑（`local.yaml`）與範本（`example.yaml`） |
| `train/`、`models/`、`datasets/`、`utils/`、`smplify/` | 程式碼 |
| `data/` | SMPL 模型、平均參數、關節回歸器、GMM 先驗、遮擋增強素材、pGT npz |
| `weights/` | HMR2.0 權重、發佈權重 |

## 安裝

測試環境：Python 3.7.16、PyTorch 1.13.1 + CUDA 11.7、單張 24 GB GPU（RTX 4090）。

```bash
conda create -n rgbt-hmr python=3.7 -y
conda activate rgbt-hmr
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.21.6 scipy==1.7.3 opencv-python pyyaml tqdm einops timm==0.6.13 yacs \
            smplx==0.1.28 chumpy torchgeometry==0.1.2 pycocotools tensorboard \
            scikit-image==0.17.2 joblib pytorch-lightning==1.9.5
pip install pyrender trimesh   # 只有 demo 需要
```

## 權重

| 檔案 | 大小 | 來源 |
|---|---|---|
| `weights/hmr2/epoch=35-step=1000000.ckpt` | 2.7 GB | [4D-Humans](https://github.com/shubham-goel/4D-Humans) 釋出的 HMR2.0 checkpoint |
| `weights/main/model_final.pt` | 23 MB | 本方法的發佈權重：只含 ARG / MA / DCG |

發佈權重不含 HMR2.0 的部分。`tools/eval_2d_mu.py` 載入時會依 `--train_cfg` 中的 `MODEL.HMR2_CHECKPOINT`
讀取 HMR2.0，再疊上發佈權重。匯出時已檢查訓練 checkpoint 中的 503 個 backbone / head 張量與 HMR2.0 逐位元相同。

## 路徑設定

所有絕對路徑都集中在 **`configs/paths/local.yaml`**，搬移專案或資料集後只要改這個檔案：

```bash
cp configs/paths/example.yaml configs/paths/local.yaml   # 再填入自己的路徑
```

| 區塊 | 鍵 | 內容 |
|---|---|---|
| `DATASET_ROOTS` | `LLVIP_ROOT` | LLVIP 根目錄（需含 `infrared/`、`visible/`、`annotations/`） |
| | `LLVIP_VISIBLE_ROOT` | LLVIP 可見光影像（與 IR 像素對齊），即 `<LLVIP_ROOT>/visible` |
| | `OTP2_DATASET_ROOT`、`OTP2_ROOT` | OpenThermalPose2 根目錄與其 `train/` |
| `PATHS` | — | 專案內的檔案（相對路徑，不需修改） |
| `EXTERNAL` | `CAMERAHMR_ROOT`、`CAMERAHMR_CHECKPOINT` | 只有重新產生 pGT 時需要 |

LLVIP 資料夾結構（`annotations/` 是 LLVIP-Pose 的 2D 關鍵點標註，放進原始 LLVIP 資料夾裡）：

```
LLVIP/
├── infrared/{train,test}/*.jpg    # 熱影像（LLVIP 原始檔）
├── visible/{train,test}/*.jpg     # 配對 RGB（LLVIP 原始檔）
└── annotations/                   # LLVIP-Pose：train_llvip.json、test_llvip.json
```

訓練設定檔中的權重路徑（`weights/...`、`logs/...`）都是相對路徑，**所有指令請在專案根目錄執行**。

`weights/`、`data/smpl/`、`data/dataset_extras/` 不進 git。SMPL 模型請自行從 https://smpl.is.tue.mpg.de/ 下載，
放到 `data/smpl/SMPL_{NEUTRAL,MALE,FEMALE}.pkl`。

---

## 評估

```bash
bash scripts/eval_main.sh       # 評估 weights/main/model_final.pt，輸出到 eval_results/main/
```

單一設定：

```bash
python tools/eval_2d_mu.py --dataset llvip --eval_mode rgb_ir --eval_3d \
    --checkpoint weights/main/model_final.pt \
    --train_cfg  configs/train/main/s5_dcg.yaml
```

`--eval_mode`：`rgb_ir`、`ir_only`、`rgb_only`。`--train_cfg` 必填：融合模組的部分超參數無法從權重形狀推得。

### 結果（`weights/main/model_final.pt`，root = 雙髖中點，焦距 5000）

| 設定 | 2D-MPJPE (px) ↓ | PCK@0.05 ↑ | PCK@0.1 ↑ | 3D-MPJPE (mm) ↓ | PA-MPJPE (mm) ↓ |
|---|---|---|---|---|---|
| LLVIP RGB+IR | 10.51 | 78.99 | 94.70 | 123.22 | 69.62 |
| LLVIP 僅 IR | 12.68 | 75.27 | 90.68 | 128.86 | 70.67 |
| OTP2 僅 IR | 6.21 | 88.14 | 97.64 | 94.22 | 59.54 |
| LLVIP 僅 RGB（= HMR2.0） | 15.55 | 64.12 | 88.24 | 143.55 | 77.17 |

3D 指標是相對 SMPLify 偽標註（pGT）計算，不是動作捕捉的真值。

## 訓練

分五個階段，每個階段從上一階段的 checkpoint 接續（`IO.PRETRAINED_CHECKPOINT`）。每階段 10k 步、batch size 8，單張 4090 約 1.5 小時。

| 階段 | 設定檔 | 訓練的模組 | 資料 |
|---|---|---|---|
| S1 | `s1_arg.yaml` | ARG | LLVIP RGB+IR |
| S2 | `s2_ma.yaml` | ARG + MA | LLVIP（RGB+IR 0.6 / 僅 IR 0.3 / 僅 RGB 0.1） |
| S3 | `s3_frozen_gate.yaml` | MA（ARG 的 gate MLP 凍結） | LLVIP 0.6 + OTP2 IR 0.4 |
| S4 | `s4_refine.yaml` | 同 S3，再訓練一輪 | 同 S3 |
| S5 | `s5_dcg.yaml` | DCG（其餘全部凍結） | LLVIP RGB+IR |

```bash
bash scripts/train_main.sh      # 五階段依序執行，輸出到 logs/main/<階段>/，最後匯出 logs/main/model_final.pt
```

- 只跑單一階段：`python train.py --train_cfg configs/train/main/<階段>.yaml --paths_cfg configs/paths/local.yaml`
- 監督：2D 關鍵點 + pGT SMPL 參數（pose 1.0、betas 0.1）。
- S2–S4 從上一階段的 `checkpoint_last.pt` 接續；S5 從 S4 的 `checkpoint_best.pt` 接續。
- 每個完整 checkpoint 約 2.7 GB（每階段會存 last、best 與一個時間戳記版本），五階段約需 40 GB 磁碟。
- 訓練沒有固定亂數種子，重新訓練的數字會與上表有小幅差異。

---

## pGT 生成

熱影像資料集沒有 3D 標註。訓練用的 pGT npz（`data/dataset_extras/thermal_pgt_camsmplify_v5/`）是從人工標註的 2D 關鍵點，
用 CameraHMR 初始化再跑 SMPLify 擬合得到的（v5 協定）。

| 檔案 | 內容 |
|---|---|
| `tools/generate_thermal_smplify_pgt_v5.py` | 主程式：讀 2D 標註、CameraHMR 初始化、呼叫 SMPLify、品質篩選、寫出 npz |
| `smplify/thermal_camsmplify_v4.py` | SMPLify 最佳化器（以初始化為錨的三階段擬合） |

這兩個檔案是產生目前 pGT 的原始版本，**請勿修改**：

```
73d0b348e7244ee9d870824fcc8959e238d2f04bfdf5768e73f34dea8eeecde0  tools/generate_thermal_smplify_pgt_v5.py
9f420a279b4e32070cd6c0c01680a386c5fce3829b5506e374f08bf2397401e9  smplify/thermal_camsmplify_v4.py
```

流程：

1. **讀 2D 標註**：LLVIP（COCO 格式 json）、OTP2（label txt），轉成 SPIN 的 24 個關節。熱影像中較不可靠的關節降低信心權重（腳踝、手腕 ×0.75、臀部 ×0.70）。
2. **截斷判定**：嚴重截斷不擬合；截斷的樣本降低 SMPL 監督權重並固定體型。
3. **CameraHMR 初始化**：LLVIP 與 OTP2 都用熱影像（複製成 3 通道）。
4. **SMPLify 三階段擬合**：只調相機（50 步）→ 加入全域朝向與體型（50 步）→ 全部參數並拉住初始化（100 步）。
5. **品質篩選**：依重投影誤差等條件決定 `valid_fit` 與 `pgt_quality`（rejected / weak / full）；沒通過的樣本仍保留 2D 監督。

```bash
bash scripts/gen_pgt_v5.sh --dataset both --split all
# 輸出：data/dataset_extras/thermal_pgt_camsmplify_v5_regen/（不會覆蓋訓練用的 v5）
```

需要 [CameraHMR](https://github.com/pixelite1201/CameraHMR) 的程式與權重（`EXTERNAL.CAMERAHMR_*`）。
當初產生時的完整指令沒有留下紀錄，重新產生的 SMPL 參數與已發表的 npz 有小差異（pose 最多 0.034 rad、shape 最多 0.18）。

## Demo

```bash
python tools/demo_mu.py \
    --checkpoint weights/main/model_final.pt \
    --train_cfg configs/train/main/s5_dcg.yaml \
    --dataset LLVIP --mode rgb_ir \
    --image_folder <LLVIP 紅外線影像資料夾> \
    --bbox_file <bbox json> \
    --output_folder demo_output/LLVIP/rgb_ir
```

- `--mode`：`rgb_ir` / `ir_only` 疊在熱影像上，`rgb_only` 疊在可見光影像上。
- `--bbox_file` 是 json：`{"影像檔名": [[x1, y1, x2, y2], ...], ...}`，每張影像可有多個人。
- `rgb_ir` 模式會依 `LLVIP_VISIBLE_ROOT/test/<檔名>` 找配對的 RGB 影像。
- `--max_images N` 只處理前 N 張。

---

## 已知問題

- **LLVIP 沒有官方 validation split**：訓練期驗證使用 LLVIP test npz，因此 `checkpoint_best.pt` 是在 test set 上挑選的。
- 3D-MPJPE 會隨 `--root_joint` 改變：`hip_mid` 與 `ankle` 相差約 2 倍，不可混在同一張表。
- 發佈權重訓練時編碼器內含一個 cross-modal transformer，但其輸出殘差上限設為 0（`MAA_RESIDUAL_MAX_SCALE: 0`），對結果沒有作用。
  本專案的設定檔改為不建立它（`USE_CMT: false`），推論結果逐位元相同。

## 致謝

建立在 [SPIN](https://github.com/nkolot/SPIN) 與 [4D-Humans / HMR2.0](https://github.com/shubham-goel/4D-Humans) 之上，
參考 [MMPedestron](https://github.com/BubblyYi/MMPedestron)、[BAT](https://github.com/SparkTempest/BAT)、[ViPT](https://github.com/jiawen-zhu/ViPT) 的設計。
SMPLify 姿態先驗來自 [SMPLify-X](https://github.com/vchoutas/smplify-x)。
