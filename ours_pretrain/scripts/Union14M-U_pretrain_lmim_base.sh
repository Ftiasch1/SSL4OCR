#!/bin/bash

# --- 配置 (请根据实际情况修改) ---
TEACHER_MODEL_PATH='/remote-home/huangyanglin/SSL4OCR/LMIM/ours_pretrain/ckpt/vit_small_checkpoint-19.pth'
PARENT_DATA_PATH='/root/model/Union14M/Union14M-U'
OUTPUT_FOLDER='/remote-home/huangyanglin/SSL4OCR/LMIM/ours_pretrain/output/ep10_union14m_lmim_base' # 修改输出目录名以区分
LOG_FILE="${OUTPUT_FOLDER}/pretrain_output.log"

# --- 自动创建输出目录 ---
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "创建输出目录: ${OUTPUT_FOLDER}"
    mkdir -p "${OUTPUT_FOLDER}"
fi

# --- 准备数据路径 ---
# 使用 tr 将换行符转换为为空格，确保作为参数传递时格式正确
SUB_DIRS=$(find "${PARENT_DATA_PATH}" -mindepth 1 -maxdepth 1 -type d | tr '\n' ' ')

if [ -z "$SUB_DIRS" ]; then
    echo "错误：在 ${PARENT_DATA_PATH} 下没有找到任何子目录！"
    exit 1
fi

echo "------------------------------------"
echo "开始 Baseline 训练 (无语言分支)"
echo "Teacher 权重: ${TEACHER_MODEL_PATH}"
echo "日志文件: ${LOG_FILE}"
echo "------------------------------------"

# --- 启动命令 ---
nohup torchrun \
    --nproc_per_node=8 main_pretrain.py \
    --teacher_weight "${TEACHER_MODEL_PATH}" \
    --batch_size 256 \
    --model mae_vit_small_patch4 \
    --mask_ratio 0.80 \
    --epochs 10 \
    --warmup_epochs 1 \
    --norm_pix_loss \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --output_dir "${OUTPUT_FOLDER}" \
    --log_dir "${OUTPUT_FOLDER}" \
    --data_path ${SUB_DIRS} > "${LOG_FILE}" 2>&1 &

# 打印 PID
echo "训练已在后台启动，进程ID为: $!"
echo "你可以使用 'tail -f ${LOG_FILE}' 来查看实时日志。"