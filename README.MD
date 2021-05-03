# MLP-Classifier-on-fashion-mnist(numpy)
```110 人工智慧 期中報告 ```

使用 numpy 實作 MLP 分類器，學習如何使用 forward、backward 和 update 去構成最基本的深度學習過程，這邊我們使用 fashion-mnist 資料集做練習。

# 資料夾架構
## folder
- data: fashion-mnist 資料集
- output: 模型 log 
- scripts: 自動化腳本

## code
- model.py
- train.py
- utils.py
- sort_losg

## other
- README.MD
- requirements.txt

# 如何使用
## 1.安裝必要套件
```
pip install -r requirements.txt
```

## 2.下載 fashion-mnist 放到 data 底下
到 [這裡](https://github.com/zalandoresearch/fashion-mnist) 下載，可以看到以下四個檔案:
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz

## 3.跑訓練
```python
python train.py
```