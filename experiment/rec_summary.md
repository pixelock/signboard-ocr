# 实验记录

## 文本检测

### 数据集

| 数据集名称 | 样本量 | 数据文件 | 说明 |
| --- | --- | --- | --- |
| ReCTS | 108828 | rec/rects.txt ||
| LSVT | 243015 | rec/lsvt.txt ||
| CTW | 193476 | rec/ctw.txt ||
| CTW Side | 149175 | rec/ctw-side.txt | 只去左右机位相机拍摄的图片, 排除前后相机拍摄的图片 |
| ShopSign | 9551 | rec/shopsign.txt ||
| Baidu CSR | 212023 | rec/baidu_chstr.txt ||
| Baidu CSR Tidy | 186479 | rec/baidu_chstr_tidy.txt | 剔除原数据集中文本长度为1的样本, 因为这部分样本大多图片质量很差 |

### 实验数据

| 实验数据方案名称 | 训练集 | 样本数量  | 验证集 | 样本数量 |
| --- | --- |---| --- |---|
| d1 | ReCTS | 108828 | ShopSign | 9551 |
| d2 | LSVT | 243015 | ShopSign | 9551 |
| d3 | CTW | 193476 | ShopSign | 9551 |
| d4 | CTW Side | 149175 | ShopSign | 9551 |
| d5 | Baidu CSR Tidy | 186479 | ShopSign | 9551 |
| d6 | ReCTS + LSVT + CTW Side + Baidu CSR Tidy | 687497 | ShopSign | 9551 |

### PPOCR v3

| 实验方案名称 | 数据集 | acc | norm_edit_dis | 实验备注说明 |
| --- | --- | --- | --- | --- |
| official | ShopSign验证集 | 0.7376478032752873 | 0.8923871860791661 | 官方开源模型参数 |

### ABINET

| 实验方案名称 | 数据集 | acc | norm_edit_dis | 实验备注说明 |
| --- | --- | --- | --- | --- |
| r45 | d1 | 0.5358640933977935 | 0.7580993001232743 | 使用预训练模型[abinet_vl_pretrained](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar)为基础, 继续训练 |
| r45-scratch | d1 | | | 不使用任何预训练模型, 从零开始训练 |
