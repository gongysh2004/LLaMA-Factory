- [1. 安装](#1-安装)
- [2. 打造基于Qwen3-vl-30B-A3B-Instruct的中文作文自动阅卷助手](#2-打造基于qwen3-vl-30b-a3b-instruct的中文作文自动阅卷助手)
  - [2.1. 准备数据](#21-准备数据)
  - [2.2. 下载模型](#22-下载模型)
  - [2.3. 训练](#23-训练)


# 1. 安装

```
apt install -f  python3.10-venv
python3 -m venv .venv
source .venv/bin/activate
# use aliyun pypi index
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host = mirrors.aliyun.com
EOF

pip install wheel setuptools
```

```
pip install -e ".[torch,metrics]" --no-build-isolation
```

# 2. 打造基于Qwen3-vl-30B-A3B-Instruct的中文作文自动阅卷助手

参照 [llamafactory oneline](https://docs.llamafactory.online/docs/documents/best-practice/Automatic_marking)
## 2.1. 准备数据
```
mkdir -p /workspace/user-data/datasets/AES_Dataset
pushd /workspace/user-data/datasets/AES_Dataset
wget http://llamafactory-online-assets.oss-cn-beijing.aliyuncs.com/llamafactory-online/docs/v2.0/documents/xuhong/online/%E8%87%AA%E5%8A%A8%E9%98%85%E5%8D%B7/AES_Dataset.zip
unzip AES_Dataset.zip
```
把文字转换成图片：
```
cat > /workspace/user-data/datasets/AES_Dataset/text_to_image.py <<EOF
#多模态数据格式转换代码
#文本转图片
import os
from PIL import Image, ImageDraw, ImageFont

# ---------- 参数 ----------
INPUT_DIR  = "/workspace/user-data/datasets/AES_Dataset/essays"          # 原始 txt
OUTPUT_DIR = "/workspace/user-data/datasets/AES_Dataset/essays_png"     # 输出 png
WIDTH, HEIGHT = 1240, 1754      # A4 150 dpi
MARGIN        = 60              # 四边留白
FONT_SIZE     = 20
LINE_HEIGHT   = FONT_SIZE + 10
BG_COLOR, FG_COLOR = "white", "black"
FONT_PATH     = "/workspace/user-data/datasets/AES_Dataset/SIMHEI.TTF"  # 确保存在
# --------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

def pixel_wrap(text: str, font: ImageFont.FreeTypeFont, max_px: float, draw: ImageDraw.Draw):
    """逐字符量像素，强制折行，返回行列表"""
    lines, line = [], ""
    for ch in text:
        if draw.textlength(line + ch, font=font) <= max_px:
            line += ch
        else:
            if line:
                lines.append(line)
            line = ch
    if line:
        lines.append(line)
    return lines

for txt_name in os.listdir(INPUT_DIR):
    if not txt_name.endswith(".txt"):
        continue
    with open(os.path.join(INPUT_DIR, txt_name), encoding="utf-8") as f:
        text = f.read().strip()

    img  = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    max_width = WIDTH - 2 * MARGIN   # 可打印像素宽度
    y = MARGIN
    for para in text.splitlines():
        if para.strip() == "":
            y += LINE_HEIGHT
            continue
        # 按像素折行
        for line in pixel_wrap(para, font, max_width, draw):
            draw.text((MARGIN, y), line, font=font, fill=FG_COLOR)
            y += LINE_HEIGHT
            if y > HEIGHT - MARGIN:
                break
        if y > HEIGHT - MARGIN:
            break

    out_path = os.path.join(OUTPUT_DIR, txt_name.replace(".txt", ".png"))
    img.save(out_path)
    print("saved", out_path)

print("✅ 全部转换完成，右侧无截字。输出目录：", OUTPUT_DIR)
EOF
```

然后进行转换：
(.venv) root@gongysh-1:~/LLaMA-Factory# python /workspace/user-data/datasets/AES_Dataset/text_to_image.py


把数据集加入到数据集列表里data/dataset_info.json:
```
  "aes_data": {
    "file_name": "/workspace/user-data/datasets/AES_Dataset/aes_data.json",
    "formatting": "sharegpt",
    "columns": {
    "messages": "conversations",
    "images": "images"
    },
    "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "user",
    "assistant_tag": "assistant"
    },
    "customized_status": 8,
    "total_tokens": "319459",
    "num_samples": "300",
    "avg_tokens": "1064.86"
}
```

## 2.2. 下载模型
```
pip3 install modelscope
modelscope download Qwen/Qwen3-VL-30B-A3B-Instruct --local_dir /opt/data/model_scope/Qwen/Qwen3-VL-30B-A3B-Instruct
```
## 2.3. 训练

```
export NCCL_IB_DISABLE=1   #机器的ib网卡没连线，不能让nccl 使用网卡
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path /opt/data/model_scope/Qwen/Qwen3-VL-30B-A3B-Instruct \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen3_vl_nothink \
    --flash_attn auto \
    --dataset_dir data \
    --dataset aes_data \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --enable_thinking True \
    --report_to none \
    --output_dir saves/Qwen3-VL-30B-A3B-Instruct/lora/train_2025-12-24-16-01-20 \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --freeze_vision_tower True \
    --freeze_multi_modal_projector True \
    --image_max_pixels 589824 \
    --image_min_pixels 1024 \
    --video_max_pixels 65536 \
    --video_min_pixels 256
```



torchrun  --standalone --nnodes=1 --nproc-per-node=8  src/train.py \
--stage sft \
--model_name_or_path /opt/data/model_scope/LLM-Research/Meta-Llama-3-8B-Instruct  \
--do_train \
--dataset alpaca_en_demo \
--template llama3 \
--finetuning_type lora \
--output_dir  saves/llama3-8b/lora/ \
--overwrite_cache \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine \
--logging_steps 100 \
--save_steps 500 \
--learning_rate 1e-4 \
--num_train_epochs 2.0 \
--plot_loss \
--bf16


export NCCL_IB_DISABLE=1
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft.yaml