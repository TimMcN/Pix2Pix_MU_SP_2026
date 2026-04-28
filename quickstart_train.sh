python train.py --name timing_test --dataroot None --model pix2pix_mu --batch_size 5 --epoch_count 1 \
--output_nc 1 --input_nc 1 --no_html --update_html_freq 10000000000 --direction BtoA --lambda 500 \
--no_flip --preprocess none --print_freq 1 --gan_mode lsgan --max_dataset_size 20 \
--dataset 9 --D_LR_Mul 0.1 --use_tensorboard --n_epochs 900 --lr 0.0002 --save_epoch_freq 100 \
--display_freq 20  --save_latest_freq 25

