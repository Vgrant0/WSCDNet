#!/usr/bin/env bash

# 函数：执行指定的命令，如果执行时间小于30分钟，则重试
run_command() {
    local cmd=$1
    local min_time=$2 # 最小时间，以秒为单位

    while true; do
        local start_time=$(date +%s)

        eval $cmd

        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        if [ $duration -ge $min_time ]; then
            echo "命令执行成功，耗时 $(($duration / 60)) 分钟."
            break
        else
            echo "命令执行时间少于指定的 $(($min_time / 60)) 分钟，重试..."
            sleep 5  # 短暂休眠后重试
        fi
    done
}


# 指定GPU编号
export CUDA_VISIBLE_DEVICES=2,3

# 使用 run_command 函数执行每个实验s
# 参数：完整的命令字符串和最小执行时间（秒）
train_time=$((10 * 60))  # 10分钟
test_time=$((10))  # 20秒

run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090/pxclcd" $train_time

run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/levir.py 2 --work-dir work_dirs_3090/levir" $train_time

run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_dirs_3090/sysu" $train_time

run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_dirs_3090/clcd" $train_time
# run_command "bash tools/smfu_dist_train.sh /home/dongsj/fusiming/mmrscd-master/configs/0EfficientCD/whucd.py 2 --work-dir work_dirs_3090/whucd_dense_2_6_ssim" $train_time

#从检查点恢复
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/pxclcd.py 2 --work-dir work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/pxclcd_triangle_dense_2_6_ssim_0.93284506/iter_60000_best.pth" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/sysu.py 2 --work-dir work_dirs_3090_smfu/sysu_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/sysu_triangle_dense_2_6_ssim_0.70822251/best_mIoU_iter_29000.pth" $train_time
# run_command "bash tools/smfu_dist_train.sh configs/0EffiicientCD_smfu/clcd.py 2 --work-dir work_dirs_3090_smfu/clcd_triangle_dense_2_6_ssim_resume --resume --cfg-options load_from=/home/dongsj/fsm_code/mmrscd/work_dirs_3090_smfu/clcd_triangle_dense_2_6_ssim_0.64434123/best_mIoU_iter_8000.pth" $train_time