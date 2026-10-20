# 熱影像 pGT 生成

LLVIP 和 OTP2 只有 2D 關鍵點，沒有 3D 標註。訓練用的 SMPL 偽標註（pGT）放在
`data/dataset_extras/thermal_pgt_camsmplify_v5/`，做法是先用 CameraHMR 初始化，再根據人工標註的 2D 關鍵點跑 SMPLify 擬合（v5 協定）。

## 流程

1. **讀 2D 標註**：LLVIP 是 COCO 格式的 json，OTP2 是 label txt，兩者都轉成 SPIN 的 24 個關節。
   熱影像裡比較不可靠的關節會降低信心權重：腳踝、手腕 ×0.75，臀部 ×0.70。
2. **截斷判定**：嚴重截斷的人不做擬合；輕微截斷的保留，但降低 SMPL 監督權重並固定體型。
3. **CameraHMR 初始化**：兩個資料集都用熱影像，複製成 3 通道輸入。
4. **SMPLify 三階段擬合**：只調相機（50 步）→ 加入全域朝向與體型（50 步）→ 所有參數一起調，並拉住初始化（100 步）。
5. **品質篩選**：依重投影誤差等條件給出 `valid_fit` 和 `pgt_quality`（rejected / weak / full）。沒通過的樣本仍保留 2D 監督。

## 重新產生

```bash
bash scripts/gen_pgt_v5.sh --dataset both --split all
# 輸出到 data/dataset_extras/thermal_pgt_camsmplify_v5_regen/，不會覆蓋訓練用的 v5
```

需要 [CameraHMR](https://github.com/pixelite1201/CameraHMR) 的程式與權重，路徑填在 `configs/paths/local.yaml` 的 `EXTERNAL.CAMERAHMR_*`。

當初產生時的完整指令沒有留下紀錄，所以重新產生的結果和發佈的 npz 會有些微差異（pose 最多 0.034 rad，shape 最多 0.18）。

## 凍結的檔案

下面兩個檔案就是產生目前 pGT 的版本，請不要修改：

| 檔案 | 內容 |
|---|---|
| `tools/generate_thermal_smplify_pgt_v5.py` | 主程式：讀標註、初始化、呼叫 SMPLify、品質篩選、寫出 npz |
| `smplify/thermal_camsmplify_v4.py` | SMPLify 最佳化器 |

```
73d0b348e7244ee9d870824fcc8959e238d2f04bfdf5768e73f34dea8eeecde0  tools/generate_thermal_smplify_pgt_v5.py
9f420a279b4e32070cd6c0c01680a386c5fce3829b5506e374f08bf2397401e9  smplify/thermal_camsmplify_v4.py
```
