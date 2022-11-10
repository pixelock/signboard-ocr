# 实验记录

## 文本检测

### 数据集

| 数据集名称 | 样本量 | 数据文件 | 说明 |
| --- | --- | --- | --- |
| ReCTS | 20000 | det/rects.txt ||
| LSVT | 30000 | det/lsvt.txt ||
| CTW | 25887 | det/ctw.txt | 将源文件中ignore部分的box也包括进来 |
| ShopSign | 1265 | det/shopsign.txt | |

### 实验数据

| 实验数据方案名称 | 训练集 | 样本数量  | 验证集 | 样本数量 |
| --- | --- |-------| --- |------|
| d1 | ReCTS + LSVT + CTW | 75887 | ShopSign | 1265 |

### DB++

