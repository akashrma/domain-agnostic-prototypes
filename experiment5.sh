python main.py --training_mode 'supervised_etf' --train_domain 'MR' --tensorboard_log_dir 'logs/supervised_etf_MR_corrected' --random_seed 7 --num_classes 5 --multi_level_train True --model_dir 'pretrained_models/DeepLab_resnet_pretrained_imagenet.pth' --source_train_dir 'data/datalist/train_mr.txt' --source_val_dir 'data/datalist/val_mr.txt' --source_train_gt_dir 'data/datalist/train_mr_gt.txt' --source_val_gt_dir 'data/datalist/val_mr_gt.txt' --input_size_source 256 256 --snapshot_dir 'snapshot/supervised_etf_MR_corrected' --testfile_path 'data/datalist/test_mr.txt' --num_workers 128 --max_iters 2800