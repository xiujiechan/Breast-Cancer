import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from streamlit import columns
from zstandard import train_dictionary

# 載入資料集

# 匯入套件提供的函式，可直接載入資料集
#from sklearn.datasets import load_breast_cancer

# 載入資料
cancer = load_breast_cancer()

# 顯示資料註釋
print(cancer.DESCR)

#資料框
columns = [
    '半徑_平均值', '紋理_平均值', '周長_平均值', '面積_平均值', '平滑度_平均值',
    '緊密度_平均值', '凹面_平均值', '凹點_平均值', '對稱性_平均值', '分形維度_平均值',
    '半徑_標準誤差', '紋理_標準誤差', '周長_標準誤差', '面積_標準誤差', '平滑度_標準誤差',
    '緊密度_標準誤差', '凹面_標準誤差', '凹點_標準誤差', '對稱性_標準誤差', '分形維度_標準誤差',
    '半徑_最大值', '紋理_最大值', '周長_最大值', '面積_最大值', '平滑度_最大值',
    '緊密度_最大值', '凹面_最大值', '凹點_最大值', '對稱性_最大值', '分形維度_最大值'
]

# 將數據載入資料框
df = pd.DataFrame(cancer.data, columns=columns) 
df['ID'] = range(len(df))  # 增加 ID 列
df['target'] = cancer.target  #增加 target 列
print(df.columns.tolist())
df.reset_index(drop=True, inplace=True)


# 重新排序
df = df[['ID', 'target'] + columns]
print(df.head())

# 驗證數據框
print(df.columns)
print(df['target'].head())
print(df['ID'].head())

#ignore
train_x = df.drop(columns=['target', 'ID'], axis=1, errors='ignore')

# 將載入的資料匯入資料框
df = pd.DataFrame(cancer.data, columns=columns)

# 取得標準答案
y = pd.Series(cancer.target)

# 分割特徵和目標
train_x = df.drop(columns=['target','ID'], axis=1)
train_y = df['target']

print(train_x.head())
print(train_y.head())

# 數據集路徑
file_path = r"C:\Users\user\Desktop\Breast Cancer Wisconsin\wdbc.data"

# 讀取資料
train = pd.read_csv(file_path, header=None)

#確認列名
print(train.columns)

#刪除不需要的列
train_x = train.drop(columns, axis=1)

#定義目標變量
train_y = train['target']

# 備份資料，以便之後再利用
train_x_saved = train_x.copy()
test_x_saved = train_x.copy()  # 假設test_x使用相同數據

# 讀取資料的函數
def load_data():
    train_x, test_x = train_x_saved.copy(), test_x_saved.copy()
    return train_x, test_x


# 將需要轉換的變數放在list
num_cols = ['age', 'height', 'weight', 'amount',
            'medical_info_a1', 'medical_info_a2', 'medical_info_a3', 'medical_info_b1']

# -----------------------------------
# 標準化
train_x, test_x = load_data()
scaler = StandardScaler()
scaler.fit(train_x[num_cols])
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])


# Min-Max 縮放
train_x, test_x = load_data()
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# 讀取資料
train_x, test_x = load_data()


# 對上面所指定的欄位資料進行標準化
scaler = StandardScaler()
scaler.fit(train_x[num_cols])

# 以標準化後的資料來置換各欄位的原資料
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 結合訓練資料和測試資料並計算平均及標準差，稍後以此為基礎來進行標準化
scaler = StandardScaler()
scaler.fit(pd.concat([train_x[num_cols], test_x[num_cols]]))

# 進行標準化並置換各欄位原數值
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------


# 個別標準化訓練資料與測試資料(不好的例子)
scaler_train = StandardScaler()
scaler_train.fit(train_x[num_cols])
train_x[num_cols] = scaler_train.transform(train_x[num_cols])
scaler_test = StandardScaler()
scaler_test.fit(test_x[num_cols])
test_x[num_cols] = scaler_test.transform(test_x[num_cols])

# -----------------------------------
# Min-Max 縮放
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 以訓練資料定義多欄位的 Min-Max 縮放
scaler = MinMaxScaler()
scaler.fit(train_x[num_cols])

# 以轉換後的資料代換各欄位資料
train_x[num_cols] = scaler.transform(train_x[num_cols])
test_x[num_cols] = scaler.transform(test_x[num_cols])

# -----------------------------------
# 對數轉換
# -----------------------------------
x = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])

# 僅取對數
x1 = np.log(x)

# 加 1 後取對數
x2 = np.log1p(x)

# 取絕對值的對數後加上原本的符號
x3 = np.sign(x) * np.log(np.abs(x))

# -----------------------------------
# Box-Cox 轉換
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 將僅取正值的變數納入清單中以作轉換
# 必須注意若變數中含有缺失值須使用(~(train_x[c] <= 0.0)).all() 等方法
pos_cols = [c for c in num_cols if (train_x[c] > 0.0).all() and (test_x[c] > 0.0).all()]


# 定義以訓練資料來進行多欄位的 Box-Cox 轉換
pt = PowerTransformer(method='box-cox')
pt.fit(train_x[pos_cols])

# 以轉換後的資料來替各欄位資料
train_x[pos_cols] = pt.transform(train_x[pos_cols])
test_x[pos_cols] = pt.transform(test_x[pos_cols])

# -----------------------------------
# Yeo-Johnson 轉換
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

from sklearn.preprocessing import PowerTransformer

# 定義以訓練資料來進行多欄位的 Yeo-Johnson 轉換
pt = PowerTransformer(method='yeo-johnson')
pt.fit(train_x[num_cols])

# 以轉換後的資料來替換各欄位資料
train_x[num_cols] = pt.transform(train_x[num_cols])
test_x[num_cols] = pt.transform(test_x[num_cols])

# -----------------------------------
# clipping
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------
# 以每欄為單位，計算訓練資料的 1% quantile 及 99% quantile 作為閾值的上下限
p01 = train_x[num_cols].quantile(0.01)
p99 = train_x[num_cols].quantile(0.99)

# 將低於下限的數值 Clipping 為下限值、高於上限的數值 Clipping 為上限值
train_x[num_cols] = train_x[num_cols].clip(p01, p99, axis=1)
test_x[num_cols] = test_x[num_cols].clip(p01, p99, axis=1)

# -----------------------------------
# binning
# -----------------------------------
x = [1, 7, 5, 4, 6, 3]

# 使用 Pandas 套件的 cut 函式來執行分割作業

# 方法 1: 指定區間的數量為 3
binned = pd.cut(x, 3, labels=False)
print(binned)
# [0 2 1 1 2 0] - 顯示轉換後的數值屬於哪個區間

# 方法 2: 指定區間的範圍時 (小於等於 3.0、3.0 ~ 5.0、大於 5.0 以上)
bin_edges = [-float('inf'), 3.0, 5.0, float('inf')]
binned = pd.cut(x, bin_edges, labels=False)
print(binned)
# [0 2 1 1 2 0] - 顯示轉換後的數值屬於哪個區間

# -----------------------------------
# 將數值轉換成排序
# -----------------------------------
x = [10, 20, 30, 0, 40, 40]

# ---- 方法 1: 以 Pandas 的 rank 函式進行順序的轉換
rank = pd.Series(x).rank()
print(rank.values)
# 規則: 從  1 開始，有相同數值則將排序以平均值顯示
# [2. 3. 4. 1. 5.5 5.5]

# ---- 方法 2: 使用 Numpy 的 argsort 函式進行 2 次的排序轉萬
order = np.argsort(x)
rank = np.argsort(order)
print(rank)
# 規則: 從 0 開始，數值相同者索引小的排序在前
# [1 2 3 0 4 5]

# -----------------------------------
# RankGauss
# -----------------------------------
# 讀取資料
train_x, test_x = load_data()
# -----------------------------------

# 將訓練資料中數個欄位的特徵進行 RankGauss 轉換
transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution='normal')
transformer.fit(train_x[num_cols])

# 使用轉換後的資料代換各欄位資料
train_x[num_cols] = transformer.transform(train_x[num_cols])
test_x[num_cols] = transformer.transform(test_x[num_cols])
