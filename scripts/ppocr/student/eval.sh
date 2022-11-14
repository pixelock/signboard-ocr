python PaddleOCR/tools/eval.py -c configs/det/ppocr/student-d1.yml  -o Global.checkpoints="plm/ppocr/ch_PP-OCRv3_det_distill_train/student"

python PaddleOCR/tools/eval.py -c configs/det/ppocr/student-d1.yml  -o Global.checkpoints="output/det/ppocr/student/d1/best_accuracy"
python PaddleOCR/tools/eval.py -c configs/det/ppocr/student-d1.yml  -o Global.checkpoints="output/det/ppocr/student/d1/best_accuracy" PostProcess.thresh=0.3 PostProcess.box_thresh=0.4 PostProcess.unclip_ratio=2.0

python PaddleOCR/tools/eval.py -c configs/det/ppocr/student-d1_1.yml  -o Global.checkpoints="output/det/ppocr/student/d1/best_accuracy"