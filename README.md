# YOLOv8 臉部表情辨識（Face Emotion Detection）實作流程報告

本文說明如何使用 **YOLOv8** 與 **Roboflow 已標記資料集**，在 **Google Colab（GPU）** 上完成臉部表情偵測模型的訓練、預測與結果視覺化。

本流程屬於完整的 **Object Detection 實作**。

---

## 一、報告目標

* 使用 YOLOv8 進行 **臉部表情偵測（Object Detection）**
* 偵測對象：人臉（bounding box）
* 分類內容：表情情緒（如 angry、happy、sad、neutral 等）
* 輸出結果：

  * 人臉位置（Bounding Box）
  * 對應情緒類別與信心分數（confidence score）

---

## 二、實作環境

* 平台：Google Colab
* 硬體：GPU（NVIDIA）
* 深度學習框架：Ultralytics YOLOv8
* 資料集來源：https://universe.roboflow.com/emotions-dectection/human-face-emotions/dataset/30
  （已標記為 YOLOv8 格式）
* 參考資料：https://zhuanlan.zhihu.com/p/1927118824295102227

---

## 三、實作步驟說明

### Step 1：安裝 YOLOv8 套件

使用 Ultralytics 官方套件進行 YOLOv8 的訓練與推論。

```bash
!pip install ultralytics
```
<img width="1713" height="664" alt="step1" src="https://github.com/user-attachments/assets/c9e1179a-0d24-47b4-bf48-8e1560e91750" />

---

### Step 2：掛載 Google Drive

將 Google Drive 掛載至 Colab，以便存取資料集與保存訓練結果。

```python
from google.colab import drive
drive.mount('/content/drive')
```
<img width="991" height="130" alt="step2" src="https://github.com/user-attachments/assets/82a4e04b-f3a2-48e6-a6d2-480f43a88d9e" />

---

### Step 3：解壓資料集

解壓由 Roboflow 下載的 YOLOv8 格式資料集。

由於檔名包含空白，需使用雙引號包住完整路徑。

```bash
!unzip "/content/drive/MyDrive/A1/Human face emotions.v30--prueba-2-tfg-implementacion-amb82mini-yolv7-tiny.yolov8.zip" -d /content/
```
<img width="1746" height="519" alt="step3" src="https://github.com/user-attachments/assets/58ad382c-6f6b-4761-8d32-29d88b1f5153" />

---

### Step 4：確認資料集結構

確認資料集已正確解壓，並包含 YOLOv8 所需的結構。

```bash
!ls /content/
```

正確結構應包含：

```
train/
valid/
test/
data.yaml
```

<img width="754" height="206" alt="step4" src="https://github.com/user-attachments/assets/7167ee38-a8fe-48d6-a52e-020309bc8b60" />


### Step 5：檢查 data.yaml 設定

`data.yaml` 定義了訓練與驗證資料路徑，以及情緒分類類別。

```bash
!cat /content/data.yaml
```

範例內容：

```yaml
train: /content/train/images
val: /content/valid/images

nc: 8
names: [angry, content, disgust, fear, happy, neutral, sad, surprise]
```
<img width="1218" height="503" alt="step5" src="https://github.com/user-attachments/assets/15a2dc4e-74b0-445f-b470-fca75ff02eb9" />

---

### Step 6：模型訓練（YOLOv8）

使用 YOLOv8 小模型（yolov8n）進行訓練，適合資料量不大且作為學習練習用途。

```bash
!yolo task=detect \
mode=train \
model=yolov8n.pt \
data=/content/data.yaml \
epochs=30 \
imgsz=640 \
batch=8 \
project=/content/drive/MyDrive/yolo_emotion_practice \
name=exp1
```
<img width="1713" height="583" alt="step6" src="https://github.com/user-attachments/assets/01780ab8-4498-4af7-90e9-8f9487f1db1e" />


訓練完成後，模型權重與評估結果會儲存在指定的 Google Drive 目錄中。

---

## 四、模型預測與結果視覺化

### Step 7：載入套件

使用 Python 套件顯示訓練後的圖表與預測結果。

```python
from PIL import Image
import matplotlib.pyplot as plt
```
<img width="1122" height="270" alt="step7" src="https://github.com/user-attachments/assets/f05a2115-30ec-4aca-ac04-75591377b705" />

*輸出結果:*

<img width="640" height="330" alt="result1" src="https://github.com/user-attachments/assets/d7672a5c-69e8-4848-91d6-44aada3dca57" />


### Step 8：顯示混淆矩陣（Confusion Matrix）

混淆矩陣用於分析不同情緒類別之間的誤判情況。

```python
img = Image.open("/content/drive/MyDrive/yolo_emotion_practice/exp1/confusion_matrix.png")
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
```
<img width="1211" height="203" alt="step8" src="https://github.com/user-attachments/assets/14b52b83-67ff-496d-93cc-71aa9c66a498" />

*輸出結果:*

<img width="484" height="368" alt="result2" src="https://github.com/user-attachments/assets/8d7f7517-4ecb-4d17-9b78-52a6ff43998e" />


### Step 9：顯示驗證集預測結果

顯示模型在驗證集上的實際預測範例，包含人臉框與情緒標籤。

```python
img = Image.open("/content/drive/MyDrive/yolo_emotion_practice/exp1/val_batch0_pred.jpg")
plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
```
<img width="1220" height="177" alt="step9" src="https://github.com/user-attachments/assets/9f7b1c04-f527-47b6-a6ff-038815bbf262" />

*輸出結果:*

<img width="482" height="482" alt="result3" src="https://github.com/user-attachments/assets/d33f55c3-1aa7-4801-9b7f-70be5aa420b0" />


## 五、實作成果與學習重點

* 成功使用 YOLOv8 完成臉部表情偵測模型訓練
* 熟悉 YOLOv8 資料格式與訓練流程
* 能解讀模型評估指標（loss、mAP、confusion matrix）
* 完成從資料集準備、訓練到結果視覺化的完整 Computer Vision 專案流程

---

## 六、應用與延伸

本報告可延伸應用於：

* 即時攝影機表情辨識
* 人機互動（HCI）
* 行銷行為分析
* 情緒分析相關研究

也可進一步嘗試：

* 增加訓練 epochs
* 更換 YOLOv8 模型尺寸（yolov8s / yolov8m）
* 調整 confidence threshold 與資料增強策略

## 七、心得與學習反思

透過本次臉部表情辨識實作，我實際體驗了從資料集準備、模型訓練到結果分析的完整電腦視覺流程。以往在學習深度學習相關概念時，多半停留在模型原理或單一程式碼範例，較少有機會將整個流程實際串接完成。本次使用 YOLOv8 與 Roboflow 已標記資料集 進行實作，讓我對 Object Detection 專案的實務細節有更清楚的理解。

在資料準備階段，我學會如何使用 Roboflow 提供的 YOLOv8 格式資料集，並確認資料夾結構與 data.yaml 設定是否正確。這個步驟讓我理解到，模型訓練能否順利進行，往往取決於資料格式是否嚴謹，而非僅僅是模型本身的參數設定。實際操作時，也遇到檔名包含空白導致指令錯誤的問題，透過排查與修正路徑，讓我更熟悉在 Linux / Colab 環境中處理檔案的方式。

在模型訓練部分，我選擇使用 YOLOv8n 作為基礎模型，並在 GPU 環境下進行訓練。透過觀察訓練過程中 loss 與 mAP 的變化，我開始能理解這些指標所代表的意義，而不再只是「數字是否變小或變大」。這也讓我體會到模型訓練並非一次就能得到理想結果，而是需要根據資料量與任務需求，選擇合適的模型大小與訓練參數。

在結果分析階段，透過查看 confusion matrix 與驗證集預測圖片，我能直觀地看到模型在不同表情類別上的辨識效果，也發現某些情緒（如 neutral 與 happy）較容易混淆。這讓我理解到模型效能不只是一個整體準確率，而是需要進一步分析各類別的表現，才能找出實際可改進的方向。

總結，這次實作讓我對 YOLOv8 在實際應用上的使用方式有更深入的認識，也培養了我將理論轉化為實務成果的能力。未來若有進一步時間，我希望能嘗試調整訓練參數、增加訓練資料，或延伸至即時攝影機的表情辨識應用，以深化對電腦視覺與深度學習實務的理解。
