#!/bin/bash

# 预训练权重路径（用于初始化 student 和 teacher）
PRETRAINED_WEIGHT='/remote-home/huangyanglin/SSL4OCR/LMIM/ours_pretrain/ckpt/vit_small_checkpoint-19.pth'

# 数据和输出路径
PARENT_DATA_PATH='/root/model/Union14M/Union14M-U'
OUTPUT_FOLDER='/remote-home/huangyanglin/SSL4OCR/LMIM/ours_pretrain/output/ep10_union14m_mae_feat'
LOG_FILE="${OUTPUT_FOLDER}/pretrain_output.log"

# 创建输出目录
mkdir -p "${OUTPUT_FOLDER}"

# --- 准备数据路径 ---
SUB_DIRS=$(find "${PARENT_DATA_PATH}" -mindepth 1 -maxdepth 1 -type d)
if [ -z "$SUB_DIRS" ]; then
    echo "错误：在 ${PARENT_DATA_PATH} 下没有找到任何子目录！"
    exit 1
fi
echo "使用以下数据子目录: ${SUB_DIRS}"
echo "------------------------------------"
echo "训练模式: 特征重建 (Feature Reconstruction with Momentum Encoder)"
echo "训练日志将被写入到文件: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node=8 main_pretrain.py \
    --pretrained_weight ${PRETRAINED_WEIGHT} \
    --batch_size 256 \
    --model mae_vit_small_patch4 \
    --mask_ratio 0.75 \
    --epochs 10 \
    --warmup_epochs 1 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --use_momentum_encoder \
    --momentum 0.996 \
    --output_dir "${OUTPUT_FOLDER}" \
    --log_dir "${OUTPUT_FOLDER}" \
    --data_path ${SUB_DIRS} > ${LOG_FILE} 2>&1 &

echo "特征重建训练已启动，PID: $!"
echo "查看日志: tail -f ${LOG_FILE}"