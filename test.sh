CUDA_VISIBLE_DEVICES="0" \
python test.py \
--checkpoint "model/SAM2-UNet-suim-150.pth" \
--test_image_path "SUIM/images/validation/" \
--test_gt_path "SUIM/annotations/validation/" \
--save_path "test_save/" \
--dataset suim