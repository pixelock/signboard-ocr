python PaddleOCR/tools/train.py -c configs/det/db++/normal-d1.yml  -o Global.pretrained_model="plm/db++/ResNet50_dcn_asf_synthtext_pretrained"

python PaddleOCR/tools/train.py -c configs/det/db++/normal-d2.yml  -o Global.pretrained_model="plm/db++/ResNet50_dcn_asf_synthtext_pretrained"

python PaddleOCR/tools/train.py -c configs/det/db++/normal-d3.yml  -o Global.pretrained_model="plm/db++/ResNet50_dcn_asf_synthtext_pretrained"
python PaddleOCR/tools/train.py -c configs/det/db++/normal-d3.yml  -o Global.checkpoints="output/det/db++/normal/d3/latest"
