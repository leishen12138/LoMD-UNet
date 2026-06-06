CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/root/autodl-tmp/SAM2-UNet-main/sam2_hiera_large.pt" \
--train_image_path "/root/autodl-tmp/SAM2-UNet-main/SUIM/train_val/images/" \
--train_mask_path "/root/autodl-tmp/SAM2-UNet-main/SUIM/train_val/Mask/" \
--save_path "/root/autodl-tmp/SAM2-UNet-main/checkpoint" \
--dataset suim \
--epoch 150 \
--lr 0.001 \
--batch_size 12