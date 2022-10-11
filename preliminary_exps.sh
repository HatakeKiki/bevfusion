# task 1.2 pre
# torchpack dist-run -np 4 python tools/train.py config_pre/nuscenes/det/centerhead/secfpn/camera+lidar/swint_v0p075/convfuser.yaml
# task 1.1 pre
# torchpack dist-run -np 4 python tools/train.py config_pre/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075.yaml
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml
# resize instead of random crop
# torchpack dist-run -np 4 python tools/train.py runs/run-008b4a8b-61dfd33e/configs.yaml
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml

# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml --run-dir runs/1.5_fade --auto-resume
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_no_fade.yaml --run-dir runs/1.5_no_fade --auto-resume
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075.yaml --run-dir runs/1.7_paste --auto-resume
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --run-dir runs/1.8_bevfusion --load_from runs/1.8_bevfusion/transfusion.pth
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_gaussian.yaml --run-dir runs/1.10_gaussian --auto-resume
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml
# torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_8_test/configs.yaml runs/1.8_transmvp_stop_8_test/ckpt/epoch_20.pth --eval-options 'jsonfile_prefix=./runs/1.8_transmvp_stop_8_test/result/test' --format-only
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_8_test/configs.yaml runs/1.8_transmvp_stop_8_test/ckpt/epoch_20.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.10_gaussian_stop_3/configs.yaml runs/1.10_gaussian_stop_3/ckpt/epoch_19.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.11_transmvp_masked/configs.yaml runs/1.11_transmvp_masked/epoch_1.pth --eval bbox
# export PYTHONPATH=${PYTHONPATH}:"/home/kiki/jq/MVP/CenterNet2"
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/fuse/voxelnet_0p075_virtual.yaml --run-dir runs/1.11_transmvp_masked_fpn --load_from runs/1.11_transmvp_masked_fpn/pretrained_transmvp.pth
# torchpack dist-run -np 4 python tools/train.py /home/kiki/jq/lss/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --run-dir runs/1.12_bev_lc --load_from runs/1.12_bev_lc/pretrained_l.pth

# CUDA_VISIBLE_DEVICES=3 torchpack dist-run -np 1 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml

# python nuscenes/eval/detection/evaluate.py runs/1.8_transmvp_stop_8_test/result --output_dir runs/1.8_transmvp_stop_8_test --eval_set test --version v1.0-test
# torchpack dist-run -np 4 python tools/train.py config_full/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml

# torchpack dist-run -np 4 python tools/visualize.py runs/1.8_transmvp_stop_8_test/configs.yaml --checkpoint runs/1.8_transmvp_stop_8_test/ckpt/epoch_20.pth --out-dir ./visual_file

# 20220928
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_bevfusion.yaml
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml --run-dir runs/1.8_transmvp_stop_15 --auto-resume
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/ckpt/epoch_17.pth --eval bbox
# python tools/benchmark.py runs/1.8_transmvp_stop_8_test/configs.yaml runs/1.8_transmvp_stop_8_test/ckpt/epoch_20.pth
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_gaussian.yaml
# 20221007
# torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --run-dir runs/1.12_bev_lc --load_from runs/1.12_bev_lc_no_flip/pretrained_l.pth
# torchpack dist-run -np 4 python tools/test.py runs/1.12_bev_lc/configs.yaml runs/1.12_bev_lc/ckpt/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml --run-dir runs/1.13_transmvp_c --load_from runs/1.13_transmvp_c/pretrained_l.pth

# torchpack dist-run -np 4 python tools/test.py runs/1.13_transmvp_c/configs.yaml runs/1.13_transmvp_c/epoch_6.pth --eval bbox --topN 200
# 20221009
# torchpack dist-run -np 4 python tools/test.py runs/1.13_transmvp_c/configs.yaml runs/1.13_transmvp_c/epoch_6.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/test.py runs/1.12_bev_lc/configs.yaml runs/1.12_bev_lc/epoch_6.pth --eval bbox --topN 200

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/test.py runs/1.6_bevfusion_l/configs.yaml runs/1.6_bevfusion_l/epoch_20.pth --eval bbox --topN 200
torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 100
torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 200
torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 300
torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 400


# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_feature.yaml --run-dir runs/1.13_semantic_bev --load_from runs/1.13_semantic_bev/pretrained_l.pth