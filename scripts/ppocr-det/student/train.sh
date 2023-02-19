python PaddleOCR/tools/train.py -c configs/det/ppocr/student-d1.yml -o Global.checkpoints="plm/det/ch_PP-OCRv3_det_distill_train/student"
python PaddleOCR/tools/train.py -c configs/det/ppocr/student-d1.yml -o Global.checkpoints="output/det/ppocr/student/d1/latest"

python PaddleOCR/tools/train.py -c configs/det/ppocr/student-d2.yml -o Global.checkpoints="plm/det/ch_PP-OCRv3_det_distill_train/student"
python PaddleOCR/tools/train.py -c configs/det/ppocr/student-d2.yml -o Global.checkpoints="output/det/ppocr/student/d2/latest"