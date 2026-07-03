# SAR-Sense 应用镜像
# 基础：python 3.10 slim（对齐 rag_env_backup 的 Python 3.10.20）
FROM python:3.10-slim

# 系统 deps：
#   build-essential —— 编译部分没有预编译 wheel 的包
#   libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1 / libgomp1 —— ultralytics 依赖的 opencv 运行时需要
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 torch CPU 版（单独 index，避免默认拉 2.5GB CUDA 轮子；本地环境是 2.12.0+cpu）
RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖（独立层，requirements 不变时命中 build cache，改代码不用重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码 + 魔改 ultralytics + 模型权重 + 知识库数据 + 配置
# （.dockerignore 已排除 .env / logs / runtime / 参考项目 / 评测 等）
COPY . .

# 运行时目录占位（即便 volume 首次挂载前也要有挂载点）
RUN mkdir -p data/uploads logs runtime .cache/huggingface

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_ENDPOINT=https://hf-mirror.com \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.cache/huggingface

EXPOSE 5000

# 用项目入口启动（保留你熟悉的启动方式 + banner）
CMD ["python", "api_server_fastapi.py"]
