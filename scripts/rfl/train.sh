# STEP 1: 训练CNT分支
python PaddleOCR/tools/train.py -c configs/rec/rfl/cnt.yml -o Global.pretrained_model="plm/rec/rec_resnet_rfl_visual_train/best_accuracy"
python PaddleOCR/tools/train.py -c configs/rec/rfl/cnt.yml -o Global.checkpoints="output/rec/rfl/cnt/d1/latest"
# STEP 2: 联合训练CNT和Att分支, 注意将pretrained_model的路径设置为CNT训练得到的参数路径
python PaddleOCR/tools/train.py -c configs/rec/rfl/att.yml -o Global.pretrained_model="output/rec/rfl/cnt/d1/best_accuracy"
python PaddleOCR/tools/train.py -c configs/rec/rfl/att.yml -o Global.checkpoints="output/rec/rfl/att/d1/latest"
