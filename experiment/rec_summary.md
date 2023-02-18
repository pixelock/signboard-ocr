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

### ABINET

| 实验方案名称 | 数据集 | thresh | box_thresh | unclip_ratio | post.image_size | precision | recall | hmean |
| --- | --- | --- | --- | --- |---| --- | --- | --- |
