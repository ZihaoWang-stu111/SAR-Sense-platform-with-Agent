# SAR-Sense 应用镜像
# 基础：python 3.10 slim（对齐 rag_env_backup 的 Python 3.10.20）
FROM python:3.10-slim

# 系统 deps：
#   build-essential —— 编译部分没有预编译 wheel 的包
#   libgl1 / libglib2.0-0 / libsm6 / libxext6 / libxrender1 / libgomp1 —— ultralytics 依赖的 opencv 运行时需要
# 先换清华 Debian 镜像源（避开通过代理下 deb 包时的偶发 502），并加 apt 重试兜底
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 装 torch CPU 版分两步：
#   pytorch 官方 index 把纯 python 依赖（typing-extensions/Jinja2 等）重新打包，
#   其元数据命名（typing_extensions）与新 pip 严格校验冲突（期望 typing-extensions），会被全部拒收导致解析失败。
#   所以先从清华 PyPI 装好这些纯 python 依赖，再用 --no-deps 单独装 torch/torchvision 本体（只从 pytorch CPU index 取）。
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
        "filelock" "typing-extensions>=4.10" "sympy>=1.1.0" "networkx>=3.0" "Jinja2>=3.0" "fsspec"
RUN pip install --no-cache-dir --no-deps torch==2.12.0 torchvision==0.27.0 \
        --index-url https://download.pytorch.org/whl/cpu

# 再装其余依赖（独立层，requirements 不变时命中 build cache，改代码不用重装依赖）
# 清华 PyPI 镜像加速 + 官方 PyPI 兜底：部分包（如 regex 新版本）清华源同步滞后，
# 用 --extra-index-url 让 pip 回官方源取，避免 "from versions: none" 解析失败
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.org/simple

# 拷贝项目代码、精简后的自定义 ultralytics、模型权重和配置。
# 本地数据、向量库、日志与密钥由 .dockerignore 排除，运行时通过卷和环境变量提供。
COPY . .

# 使用非 root 用户运行应用；命名卷首次创建时会继承这些目录权限。
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p data logs runtime chroma_db .cache/huggingface \
    && chown -R app:app /app

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_ENDPOINT=https://hf-mirror.com \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.cache/huggingface

EXPOSE 5000

USER app

# 用项目入口启动（保留你熟悉的启动方式 + banner）
CMD ["python", "api_server_fastapi.py"]
