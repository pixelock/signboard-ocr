python PaddleOCR/tools/eval.py -c configs/det/db++/normal-d1.yml  -o Global.checkpoints="output/det/db++/normal/d1/best_accuracy"

python PaddleOCR/tools/eval.py -c configs/det/db++/normal-d2.yml  -o Global.checkpoints="output/det/db++/normal/d2/best_accuracy"

python PaddleOCR/tools/eval.py -c configs/det/db++/normal-d3.yml  -o Global.checkpoints="output/det/db++/normal/d3/best_accuracy"
python PaddleOCR/tools/eval.py -c configs/det/db++/normal-d3.yml  -o Global.checkpoints="output/det/db++/normal/d3/best_accuracy" PostProcess.thresh=0.3 PostProcess.box_thresh=0.5 PostProcess.unclip_ratio=1.5
python PaddleOCR/tools/eval.py -c configs/det/db++/normal-d3.yml  -o Global.checkpoints="output/det/db++/normal/d3/best_accuracy" PostProcess.thresh=0.3 PostProcess.box_thresh=0.4 PostProcess.unclip_ratio=2.0

python PaddleOCR/tools/eval.py -c configs/det/db++/eastcrop640-d2.yml  -o Global.checkpoints="output/det/db++/eastcrop640/d2/best_accuracy"
python PaddleOCR/tools/eval.py -c configs/det/db++/eastcrop640-d2.yml  -o Global.checkpoints="output/det/db++/eastcrop640/d2/best_accuracy" PostProcess.thresh=0.3 PostProcess.box_thresh=0.4 PostProcess.unclip_ratio=2.0
python PaddleOCR/tools/eval.py -c configs/det/db++/eastcrop640-d3.yml  -o Global.checkpoints="output/det/db++/eastcrop640/d3/best_accuracy"
python PaddleOCR/tools/eval.py -c configs/det/db++/eastcrop640-d3.yml  -o Global.checkpoints="output/det/db++/eastcrop640/d3/best_accuracy" PostProcess.thresh=0.3 PostProcess.box_thresh=0.4 PostProcess.unclip_ratio=2.0