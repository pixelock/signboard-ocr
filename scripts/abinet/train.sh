# 不使用预训练模型训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-scratch-d1.yml
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-scratch-d1.yml -o Global.checkpoints="output/rec/abinet/r45-scratch/d1/latest"
# 使用预训练模型(英文数据集上训练)训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d1.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d1.yml -o Global.checkpoints="output/rec/abinet/r45/d1/latest"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d6.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d6.yml -o Global.checkpoints="output/rec/abinet/r45/d6/latest"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d7.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-d7.yml -o Global.checkpoints="output/rec/abinet/r45/d7/latest"
# 使用Scene Text Recognition Data Augmentation, 做数据增强训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-straug-d1.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-straug-d1.yml -o Global.checkpoints="output/rec/abinet/r45-straug/d1/latest"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-straug-d6.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-straug-d6.yml -o Global.checkpoints="output/rec/abinet/r45-straug/d6/latest"
# 使用PaddleOCR使用的数据增量方法(RecConAug + RecAug), 做数据增强训练
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-ppaug-d6.yml -o Global.pretrained_model="plm/rec/rec_r45_abinet_train/abinet_vl_pretrained"
python PaddleOCR/tools/train.py -c configs/rec/abinet/r45-ppaug-d6.yml -o Global.checkpoints="output/rec/abinet/r45-ppaug/d6/latest"
