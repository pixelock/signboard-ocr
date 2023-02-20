# 不使用预训练模型训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-scratch-d1.yml
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-scratch-d1.yml -o Global.checkpoints="output/rec/abinet/r45-scratch/d1/latest"
# 使用预训练模型(英文数据集上训练)训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d1.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d1.yml -o Global.checkpoints="output/rec/abinet/r45/d1/latest"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d6.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"