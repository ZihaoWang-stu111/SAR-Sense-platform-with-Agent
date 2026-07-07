# MinerU PDF 解析服务镜像（CPU pipeline 后端）
#
# 与 app 镜像隔离：MinerU 自己的 torch/依赖全在这个容器里，不污染 app。
# app 容器只通过 HTTP（http://mineru:8000）调用，不装 MinerU。
#
# 构建：docker compose up --build mineru
# 首次启动后需预热模型权重（1-2GB）：
#   docker exec sar-sense-mineru mineru-models-download --source modelscope
# 或直接上传一个 PDF 触发下载（首次解析会慢）。

# 基础镜像 pin 到 bookworm（Debian 12），避免 slim 动态标签漂移到 trixie（Debian 13）
# ——trixie 的 mesa 依赖 libllvm19（几百 MB），清华源对大文件常 502，卡死 apt。
FROM python:3.10-slim-bookworm

# 系统 deps：MinerU pipeline 走 opencv 系运行时，和 app 镜像同一套。
# 换阿里云 Debian 镜像源（清华源对大文件 deb 偶发 502）+ apt 重试 + --fix-missing 兜底。
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g; s|security.debian.org|mirrors.aliyun.com|g' \
        /etc/apt/sources.list 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=8 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 装 torch CUDA 版分两步（同 app 镜像套路，避免 typing_extensions 元数据冲突）：
#   先从清华 PyPI 装纯 python 依赖，再用 --no-deps 单独装 torch/torchvision 本体。
#   cu124 索引（CUDA 12.4）：cu121 最高只到 torch 2.5.1，不满足 MinerU >=2.6.0。
#   torch 2.6.0 是 MinerU 下限；RTX 2060 (Turing, CC 7.5) CUDA 12.4 支持。
#   vlm-engine 要 8GB+ VRAM，RTX 2060 6GB 跑不了，保持 backend=pipeline（模型自动用 GPU）。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "filelock" "typing-extensions>=4.10" "sympy>=1.1.0" "networkx>=3.0" "Jinja2>=3.0" "fsspec"
RUN pip install --no-cache-dir --no-deps torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

# MinerU pipeline extras（GPU 加速版；pipeline 后端 86.47 准确率，模型自动用 CUDA）。
# 清华 PyPI + 官方 PyPI 兜底：部分包（如 regex 新版本）清华源同步滞后，
# --extra-index-url 让 pip 回官方源取，避免 "from versions: none" 解析失败。
# six：MinerU pipeline 运行时依赖但未在 extras 里声明，必须显式装，
# 否则首次解析报 `No module named 'six'`（409 任务失败）。实测必须。
RUN pip install --no-cache-dir "mineru[pipeline]==3.4.2" "six" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://pypi.org/simple

# 模型权重源：modelscope 国内快；HF_ENDPOINT 兜底（部分包仍走 HF）。
ENV PYTHONUNBUFFERED=1 \
    MINERU_MODEL_SOURCE=modelscope \
    HF_ENDPOINT=https://hf-mirror.com

EXPOSE 8000

# --host 0.0.0.0 必填：默认 127.0.0.1，容器外（app 容器）访问不到。
CMD ["mineru-api", "--host", "0.0.0.0", "--port", "8000"]
