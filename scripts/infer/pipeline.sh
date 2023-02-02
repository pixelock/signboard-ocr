python PaddleOCR/tools/infer/predict_system.py --use_gpu=true --image_dir="data/ShopSign" \
  --det_model_dir="plm/det/ch_PP-OCRv3_det_infer" --det_limit_type=max --det_limit_side_len=1440 --det_db_box_thresh=0.4 --det_db_unclip_ratio=2.0 \
  --rec_model_dir="plm/rec/ch_PP-OCRv3_rec_infer" --rec_char_dict_path="PaddleOCR/ppocr/utils/ppocr_keys_v1.txt" \
  --use_angle_cls=true --cls_model_dir="plm/dir/ch_ppocr_mobile_v2.0_cls_infer" \
  --vis_font_path="PaddleOCR/doc/fonts/simfang.ttf" --draw_img_save_dir="results/infer/ppocrv3_ppocrv3"

python PaddleOCR/tools/infer/predict_system.py --use_gpu=true --image_dir="data/ShopSign" \
  --det_algorithm=DB++ --det_model_dir="output/det/db++/eastcrop640/d2/inference" --det_limit_type=max --det_limit_side_len=1440 --det_db_box_thresh=0.4 --det_db_unclip_ratio=2.0 \
  --rec_model_dir="plm/rec/ch_PP-OCRv3_rec_infer" --rec_char_dict_path="PaddleOCR/ppocr/utils/ppocr_keys_v1.txt" \
  --use_angle_cls=true --cls_model_dir="plm/dir/ch_ppocr_mobile_v2.0_cls_infer" \
  --vis_font_path="PaddleOCR/doc/fonts/simfang.ttf" --draw_img_save_dir="results/infer/eastcrop640-d2_ppocrv3"
