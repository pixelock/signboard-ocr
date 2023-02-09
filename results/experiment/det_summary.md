# 实验记录

## 文本检测

### 数据集

| 数据集名称 | 样本量 | 数据文件 | 说明 |
| --- | --- | --- | --- |
| ReCTS | 20000 | det/rects.txt ||
| LSVT | 30000 | det/lsvt.txt ||
| CTW | 25887 | det/ctw.txt | 将源文件中ignore部分的box也包括进来 |
| ShopSign | 1265 | det/shopsign.txt ||

### 实验数据

| 实验数据方案名称 | 训练集 | 样本数量  | 验证集 | 样本数量 |
| --- | --- |---| --- |---|
| d1 | ReCTS + LSVT + CTW | 75887 | ShopSign | 1265 |
| d2 | ReCTS + LSVT | 50000 | ShopSign | 1265 |
| d3 | LSVT | 30000 | ShopSign | 1265 |

### PPOCR v3

| 实验方案名称 | 数据集 | thresh | box_thresh | unclip_ratio | post.image_size | precision | recall | hmean |
| --- | --- | --- | --- | --- |---| --- | --- | --- |
| 开源模型 | | 0.3 | 0.5 | 1.5 | 1024 | 0.6275274056029233 | 0.5093929207039747 | 0.5623226369788257 |
| 开源模型 | | 0.3 | 0.4 | 2.0 | 1024 | 0.6188095791676355 | 0.5263001779711292 | 0.5688181235306689 |

### DB++

| 实验方案名称 | 数据集 | thresh | box_thresh | unclip_ratio | post.image_size | precision | recall | hmean |
| --- | --- | --- | --- | --- |---| --- | --- | --- |
| eastcrop640 | d2 | 0.3 | 0.4 | 2.0 | 1024 | 0.7272626564750194 | 0.6491002570694088 | 0.6859620709471815 |
