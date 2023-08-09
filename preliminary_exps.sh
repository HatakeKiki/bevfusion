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
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 100
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 300
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 400

# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml --run-dir runs/1.8_transmvp_stop_15_test

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/train.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_feature.yaml --run-dir runs/1.13_semantic_bev --load_from runs/1.13_semantic_bev/pretrained_l.pth
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --eval_set train --version v1.0-train --eval-options 'jsonfile_prefix=./runs/result'
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_8_test/configs.yaml runs/1.8_transmvp_stop_15_test/epoch_20.pth --topN 300 --eval-options 'jsonfile_prefix=runs/1.8_transmvp_stop_15_test/' --format-only

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox --topN 200
# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox --topN 100
# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox --topN 300
# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox --topN 400

# 1105 test 2nd
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.2.4_test --load_from runs/1.8_transmvp_stop_15_test/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.2.2_test --load_from runs/1.8_transmvp_stop_15_test/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.2.4_threshold --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/test.py runs/1.15.5_transmvp_sem_c_embed_add/configs.yaml runs/1.15.5_transmvp_sem_c_embed_add/epoch_6.pth --eval bbox --eval-options 'jsonfile_prefix=runs/1.15.5_transmvp_sem_c_embed_add/'
# torchpack dist-run -np 4 python tools/test.py runs/1.16_transmvp_sem_c_mask/configs.yaml runs/1.16_transmvp_sem_c_mask/epoch_6.pth --eval bbox # --eval-options 'jsonfile_prefix=runs/1.16_transmvp_sem_c_mask/'

# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --topN 300 --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2.3_test/epoch_6.pth --eval bbox # --eval-options 'jsonfile_prefix=result_train/fuse' --format-only
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval-options 'jsonfile_prefix=result_train/lidar' --format-only

# 1105 test 2nd
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.6_threshold --load_from runs/1.8_transmvp_stop_15/epoch_20.pth
# torchpack dist-run -np 4 python tools/test.py runs/1.15.6_threshold/configs.yaml runs/1.15.6_threshold/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.2.4_test --load_from runs/1.8_transmvp_stop_15_test/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.7_uniform --load_from runs/1.8_transmvp_stop_15/epoch_20.pth --auto-resume
# torchpack dist-run -np 4 python tools/test.py runs/1.12.4_bev_lc_no_depth/configs.yaml runs/1.12.4_bev_lc_no_depth/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_test/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_test/configs.yaml runs/1.8_transmvp_stop_15_test/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2.4_test/configs.yaml runs/1.15.2.4_val/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.15.2.4_test/configs.yaml runs/1.15.2.4_test/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.15.2.4_test/configs.yaml runs/1.15.2.4_test/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2.4_test/configs.yaml runs/1.15.2.4_test/epoch_6.pth --topN 300 --eval-options 'jsonfile_prefix=runs/1.15.2.4_test' --format-only

# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p05_virtual.yaml --run-dir runs/1.8_transmvp_0p05

# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml runs/1.15.2.4_val/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_0p05/configs.yaml runs/1.8_transmvp_0p05/epoch_20.pth --eval bbox --eval-options 'jsonfile_prefix=runs/1.8_transmvp_0p05'
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --eval-options 'jsonfile_prefix=runs/1.8_transmvp_stop_15'

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_384.yaml --run-dir runs/1.15.2_384 --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/test.py runs/1.12.5_bev_lc/configs.yaml runs/1.12.5_bev_lc/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml runs/1.12_bev_lc/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_lc_384.yaml --run-dir runs/1.6.2_384_unfrozen --load_from runs/1.12_bev_lc/pretrained_l.pth #--auto-resume

# torchpack dist-run -np 4 python tools/test.py runs/1.6.2_384/configs.yaml runs/1.6.2_384/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_384.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_384.yaml --run-dir runs/1.15.2_384_3 --auto_resume # --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml --run-dir runs/1.17_bev_c

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/co nfigs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_transmvp_sem_c_embed/configs.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml --run-dir runs/1.8_transmvp_stop_15_mask2former_2
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.8_mask2former --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/test.py runs/1.15.2_384_3/configs.yaml runs/1.15.2_384_3/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.6.2_384/configs.yaml runs/1.6.2_384/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml --run-dir runs/1.8_transmvp_stop_15_mask2former_2
# torchpack dist-run -np 4 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.8_mask2former --load_from runs/1.8_transmvp_stop_15_mask2former_test/epoch_20.pth
# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_mask2former_test/configs.yaml runs/1.8_transmvp_stop_15_mask2former/epoch_20.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_mask2former/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox #  --eval-options 'jsonfile_prefix=runs/1.8_transmvp_stop_15_mask2former_test'
# torchpack dist-run -np 4 python tools/test.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_mask2former_test/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox --eval-options 'jsonfile_prefix=json_file'
# torchpack dist-run -np 3 python tools/train.py config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml --run-dir runs/1.15.8_mask2former --load_from runs/1.8_transmvp_stop_15_mask2former/epoch_20.pth
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_mask2former/configs.yaml runs/1.8_transmvp_stop_15_mask2former/epoch_20.pth --eval bbox --eval-options 'jsonfile_prefix=runs/1.8_transmvp_stop_15_mask2former'
# torchpack dist-run -np 4 python tools/test.py runs/1.15.8_mask2former/configs.yaml runs/1.15.8_mask2former/epoch_6.pth --eval bbox --eval-options 'jsonfile_prefix=runs/1.15.8_mask2former'

# torchpack dist-run -np 4 python tools/test.py runs/1.17_bev_c/configs.yaml pretrained/camera-only-det.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.17_bev_c/configs.yaml runs/1.17_bev_c/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_bevfusion.yaml runs/1.6_bevfusion_l/epoch_20.pth --eval bbox
# 2022/12/27
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_bevfusion_msda.yaml --run-dir runs/1.6.3_msda

# 2022/12/27
# python tools/benchmark.py runs/1.6_bevfusion_l/configs.yaml runs/1.6_bevfusion_l/epoch_20.pth
# python tools/benchmark.py runs/1.6.3_msda/configs.yaml runs/1.6.3_msda/epoch_20.pth
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_bevfusion.yaml --run-dir runs/1.6_bevfusion_l_stop_20 --auto-resume

# 2022/12/30
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075_bevfusion.yaml --run-dir runs/1.6_centerpoint_stop_15 --auto-resume
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/centerhead/secfpn/lidar/voxelnet_0p075_bevfusion_stop_20.yaml --run-dir runs/1.6_centerpoint_stop_20 --auto-resume

# 2023/01/03
# torchpack dist-run -np 4 python tools/train.py config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_bevfusion.yaml --run-dir runs/debug_lidar

# 2023/01/05
# torchpack dist-run -np 1 python tools/test.py runs/1.6.3_msda_init/configs.yaml runs/1.6.3_msda_init/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15/configs.yaml runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox # --eval-options 'jsonfile_prefix=./runs/1.8_transmvp_stop_15/result' --format-only
# torchpack dist-run -np 4 python tools/test.py runs/1.8_transmvp_stop_15_mask2former_2/configs.yaml runs/1.8_transmvp_stop_15_mask2former_2/epoch_20.pth --eval-options 'jsonfile_prefix=./runs/1.8_transmvp_stop_15_mask2former_2/result' --format-only

# 2023/01/15
# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml \
#     --run-dir runs/debug_fuse

# torchpack dist-run -np 4 python tools/test.py \
# runs/1.15.2_transmvp_sem_c_embed/configs.yaml \
# runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
# runs/1.8_transmvp_stop_15/configs.yaml \
# runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_swinl.yaml \
#     --run-dir runs/1.15.2.8_swinl_unfreeze \
#     --load_from runs/1.8_transmvp_stop_15/epoch_20.pth


# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem.yaml \
#     --run-dir runs/1.15.2.7_poolv2 \
#     --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_swinl.yaml \
#     --run-dir runs/1.15.2.6_swinl \
#     --auto-resume
    # --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

    # --auto-resume
    # runs/1.15.2_transmvp_sem_c_embed/configs.yaml \


# CUDA_VISIBLE_DEVICES="0" torchpack dist-run -np 1 python tools/test.py \
#     runs/1.8_transmvp_stop_15/configs.yaml \
#     runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox


# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_swinl.yaml \
#     --run-dir runs/1.15.2.6_swinl_bevpoolv2 \
#     --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/transmvp_c_sem_swinl.yaml \
#     --run-dir runs/1.15.2.8_swinl_unfreeze \
#     --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/test.py \
#     runs/2.1_two_stage_debug/configs.yaml \
#     runs/1.8_transmvp_stop_15/ckpt/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/visualize.py \
#     runs/2.1_two_stage_debug/configs.yaml \
#     runs/1.8_transmvp_stop_15/ckpt/epoch_20.pth \
#     --out-dir ./visual

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.8_transmvp_stop_15/configs.yaml \
#     runs/1.8_transmvp_stop_15/ckpt/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.8_transmvp_stop_15_mask2former_2/configs.yaml \
#     runs/1.8_transmvp_stop_15_mask2former_2/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/two_stage_transfusion/secfpn/lidar/voxelnet_0p075.yaml
#     --run-dir runs/2.1_two_stage \
#     --load_from runs/1.8_transmvp_stop_15/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/two_stage_transfusion/secfpn/lidar/voxelnet_0p075.yaml \
#     --run-dir runs/2.5 \
#     --load_from pretrained/transmvp_mask_rcnn.pth

# torchpack dist-run -np 4 python tools/train.py \
#     runs/1.7.1_mvp/configs.yaml \
#     --run-dir runs/1.7.1_mvp \
#     --auto-resume

# CUDA_VISIBLE_DEVICES=2 python tools/test.py \
#     runs/1.6_centerpoint_stop_20/configs.yaml \
#     runs/1.6_centerpoint_stop_20/epoch_20.pth --eval bbox \
#     --eval-options 'jsonfile_prefix=./runs/1.6_centerpoint_stop_20/result' --not_dist


# torchpack dist-run -np 4 python tools/train.py \
#     runs/debug_speed_camera_only/configs.yaml \
#     --run-dir runs/debug_speed_camera_only

# torchpack dist-run -np 1 python tools/train.py \
#     runs/1.6_bevfusion_l/configs.yaml \
#     --run-dir runs/debug_speed_bevfusion_l

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.15.2_transmvp_sem_c_embed/configs.yaml \
#     runs/1.15.2_transmvp_sem_c_embed/epoch_6.pth --eval bbox 



# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.8_transmvp_stop_15_htc/configs.yaml \
#     runs/1.8_transmvp_stop_15_htc/ckpt/epoch_17.pth --eval bbox \
#     --eval-options 'jsonfile_prefix=./runs/1.8_transmvp_stop_15_htc/result_17'

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.8_transmvp_stop_15_htc/configs.yaml \
#     runs/1.8_transmvp_stop_15_htc/epoch_20.pth --eval bbox \
#     --eval-options 'jsonfile_prefix=./runs/1.8_transmvp_stop_15_htc/result_20'

# torchpack dist-run -np 4 python tools/train.py \
#     config_lidar/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_virtual.yaml \
#     --run-dir runs/1.8_transmvp_stop_15_htc --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     runs/1.15.2_transmvp_sem_c_embed/configs.yaml \
#     --run-dir runs/1.15.2_htc_2 --load_from runs/1.8_transmvp_stop_15_htc/epoch_20.pth

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.17.2_bev_c_uniform/configs.yaml \
#     runs/1.17.2_bev_c_uniform/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     runs/3.5_bev_attn/configs.yaml \
#     runs/3.5_bev_attn/ckpt/epoch_11.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
#     --run-dir runs/1.17.1_bev_c \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint \
#     pretrained/swint-nuimages-pretrained.pth \
#     --run-dir runs/1.17.2_bev_c_uniform \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/bevattn.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint \
#     pretrained/swint-nuimages-pretrained.pth \
#     --run-dir runs/3.13_bev_attn \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint \
#     pretrained/swint-nuimages-pretrained.pth \
#     --load_from runs/1.12_bev_lc/pretrained_l.pth \
#     --run-dir runs/3.16_bev_attn_fusion \
#     --auto-resume


# torchpack dist-run -np 4 python tools/test.py \
#     runs/3.10_bev_attn/configs.yaml \
#     runs/3.10_bev_attn/epoch_20.pth --eval bbox 

# torchpack dist-run -np 4 python tools/test.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_lss.yaml \
#     runs/1.12.4_bev_lc_no_depth/epoch_6.pth --eval bbox 

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.12.4_bev_lc_no_depth/configs.yaml \
#     runs/1.12.4_bev_lc_no_depth/epoch_6.pth --eval bbox 

# torchpack dist-run -np 4 python tools/test.py \
#     runs/1.12_bev_lc/configs.yaml \
#     runs/1.12_bev_lc/epoch_6.pth --eval bbox



# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/bevattn_depth_gt.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint \
#     pretrained/swint-nuimages-pretrained.pth \
#     --run-dir runs/3.27_bev_attn_depth \
#     --auto-resume

# torchpack dist-run -np 4 python tools/test.py \
#     runs/3.26_bev_attn_depth/configs.yaml \
#     runs/3.26_bev_attn_depth/epoch_1.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc.yaml \
#     --load_from runs/1.12_bev_lc/pretrained_lc_attn.pth \
#     --run-dir runs/3.24_bev_attn_fusion_8_heads_depth \
#     --auto-resume

# torchpack dist-run -np 4 python tools/test.py \
#     runs/5.1_bevattn_lc_lssfpn_1/configs.yaml \
#     runs/5.1_bevattn_lc_lssfpn_1/epoch_6.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     configs/BEVAttn_L.py \
#     pretrained/lidar-only-det.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     configs/SimBEV.py \
#     --run-dir runs/sim_bev_2 \
#     --load_from runs/1.12_bev_lc/pretrained_cl_lidar.pth \
#     --no_validate

# torchpack dist-run -np 4 -v python tools/train.py\
#     configs/BEVAttn_L.py \
#     --run-dir runs/4.5_bev_lidar \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     configs/BEVAttn_LC.py \
#     --load_from runs/1.6_bevfusion_l/epoch_20.pth \
#     --run-dir runs/5.1_bevattn_lc_lssfpn_1 \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     runs/3.22_bev_attn_fusion_8_heads_sem/configs.yaml \
#     --load_from runs/1.6_bevfusion_l/epoch_20.pth \
#     --run-dir runs/bevattn_lc_lssfpn_1_debug \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2_mult.yaml \
#     --load_from pretrained/lidar-only-det.pth \
#     --run-dir runs/5.21_bevattn_lc_lssfpn_mult_1_2 \
#     --auto-resume


# torchpack dist-run -np 4 python tools/test.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/bevattn.yaml \
#     runs/3.20_bev_attn/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/bevattn_swint_fpn.yaml \
#     --run-dir runs/3.35_mult_no_decoder \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/bevattn.yaml \
#     --model.encoders.camera.backbone.init_cfg.checkpoint \
#     pretrained/swint-nuimages-pretrained.pth \
#     --run-dir runs/5.22_single_1_bn \
#     --auto-resume



# torchpack dist-run -np 4 python tools/train.py \
#     runs/3.30_bev_attn_no_decoder_mult_fpn/configs.yaml \
#     --run-dir runs/3.30_bev_attn_no_decoder_mult_fpn --auto-resume

# torchpack dist-run -np 1 python tools/test.py \
#     runs/1.8_transmvp_stop_15/configs.yaml \
#     runs/1.8_transmvp_stop_15/epoch_20.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2_mult.yaml \
#     --load_from pretrained/lidar-only-det.pth \
#     --run-dir runs/5.30_mult_freeze_3_img \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2.yaml \
#     --load_from pretrained/lidar-only-det.pth \
#     --run-dir runs/5.29_single_freeze_3_img \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2.yaml \
#     --load_from pretrained/lidar_only_swint_nuim.pth \
#     --run-dir runs/5.31_single_nuim_freeze_3_img_cosine_2e-4
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2_mult.yaml \
#     --load_from pretrained/lidar_only_swint_nuim.pth \
#     --run-dir runs/5.36_mult_decoder_fuser_freeze_3_img_fcos_cosine_2e-4 \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2_mult.yaml \
#     --load_from pretrained/lidar_only_swint_nuim.pth \
#     --run-dir debug \
#     --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     config_fuse/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevattn_lc_sem_v2_mult_res101.yaml \
#     --load_from pretrained/lidar_only_res101_fpn_fcos3d.pth \
#     --run-dir runs/5.38_mult_decoder_fuser_freeze_1_res101_fcos3d_cosine_5e-5 \
#     --auto-resume

# runs/5.28_mult_0123_decoder_img_freeze_lidar_stage_3_fcos

# torchpack dist-run -np 4 python tools/test.py \
#     configs/DSVT_bev.py \
#     pretrained/dsvt_bev_nus.pth --eval bbox

# python tools/benchmark.py \
#     runs/1.8_transmvp_stop_15//configs.yaml \
#     runs/1.8_transmvp_stop_15/epoch_20.pth

# python tools/benchmark.py \
#     runs/1.7_mvp/configs.yaml \
#     runs/1.7_mvp/epoch_20.pth

# python tools/benchmark.py \
#     runs/1.6_bevfusion_l/configs.yaml \
#     runs/1.6_bevfusion_l/epoch_20.pth

# torchpack dist-run -np 4 python tools/train.py \
#     configs/DSVT_bevfusion.py \
#     --run-dir runs/6.9_single_dsvt_freeze_3_bs_4_cosine_4e-4_fix_bn_sparse_pool # \
    # --auto-resume

# torchpack dist-run -np 4 python tools/test.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/configs.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/ckpt/epoch_1.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/configs.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/ckpt/epoch_2.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/configs.py \
#     runs/6.6_mult_dsvt_freeze_3_bs_4_cosine_1e-4_fix_bn_gelu_fuser_ln/ckpt/epoch_3.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     configs/DSVT_bev.py \
#     pretrained/dsvt_bev_nus.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     configs/DSVT_bev.py \
#     --run-dir runs/7.5_dsvt --sync_bn \
#     --auto-resume

# torchpack dist-run -np 4 python tools/test.py \
#     configs/DSVT_bev.py \
#     pretrained/dsvt_bev_nus.pth --eval bbox

# torchpack dist-run -np 4 python tools/test.py \
#     configs/dsvt_transhead.py \
#     runs/7.2_dsvt_transhead/epoch_1.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     runs/7.2_dsvt_transhead/configs.py \
#     --run-dir runs/7.2_dsvt_transhead --auto-resume

# torchpack dist-run -np 4 python tools/test.py \
#     configs/BEVAttn_LC.py \
#     runs/5.40_mult_12_decoder_fuser_freeze_3_img_cosine_1e-4_sparse_ctr/epoch_1.pth --eval bbox

# torchpack dist-run -np 4 python tools/train.py \
#     configs/BEVAttn_LC.py \
#     --load_from pretrained/lidar_only_res101_fpn_fcos3d.pth \
#     --run-dir runs/5.40_mult_12_decoder_fuser_freeze_3_img_cosine_1e-4_sparse_ctr \
#     --auto-resume

torchpack dist-run -np 4 python tools/train.py \
    configs/DSVT_bev.py \
    --run-dir runs/7.6_dsvt --sync_bn \
    --auto-resume

# torchpack dist-run -np 4 python tools/train.py \
#     configs/BEVAttn_LC.py \
#     --load_from pretrained/lidar_only_res101_fpn_fcos3d.pth \
#     --run-dir runs/5.41_mult_12_decoder_fuser_freeze_3_img_cosine_1e-4_sparse_ctr \
#     --auto-resume