python PaddleOCR/tools/train.py -c configs/det/db++/normal-d1.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"

python PaddleOCR/tools/train.py -c configs/det/db++/normal-d2.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"

python PaddleOCR/tools/train.py -c configs/det/db++/normal-d3.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"
python PaddleOCR/tools/train.py -c configs/det/db++/normal-d3.yml -o Global.checkpoints="output/det/db++/normal/d3/latest"

python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d1.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"
python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d1.yml -o Global.checkpoints="output/det/db++/eastcrop640/d1/latest"

python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d2.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"
python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d2.yml -o Global.checkpoints="output/det/db++/eastcrop640/d2/latest"

python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d3.yml -o Global.pretrained_model="plm/det/db++/ResNet50_dcn_asf_synthtext_pretrained"
python PaddleOCR/tools/train.py -c configs/det/db++/eastcrop640-d3.yml -o Global.checkpoints="output/det/db++/eastcrop640/d3/latest"
