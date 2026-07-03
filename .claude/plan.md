# SAR-Sense Docker 容器化方案

## 目标
`docker compose up` 一条命令拉起整套（MySQL + 应用），浏览器打开 `localhost:5000` 即可用，知识库预载，无需消耗 DashScope 配额重新 embedding。

## 架构
```
docker-compose.yml
├─ db      mysql:8.0          端口 3306, utf8mb4, 建库 sar_sense, healthcheck
└─ app     本地构建镜像         端口 5000, depends_on db(healthy)
            ├─ 代码 + ultralytics + best.pt + chroma_db + data + 配置 全部 COPY 进镜像
            └─ data/uploads, logs, runtime, HF cache 挂载为命名 volume（运行时数据，跨重建持久）
```

## 要创建的文件（5 个）

### 1. `requirements.txt`（新建）
从 `rag_env_backup` 精选直接依赖（不照搬 281 个 pip freeze，避免 Windows-only 包污染 Linux 镜像）。关键项：
```
fastapi==0.112.4, uvicorn==0.47.0, python-multipart==0.0.28
pydantic==2.13.4, pydantic-settings==2.14.1, PyYAML, python-dotenv
SQLAlchemy==2.0.28, aiomysql==0.3.2, PyMySQL==1.2.0, cryptography==48.0.0
bcrypt==5.0.0, PyJWT==2.13.0
langchain==1.3.0, langchain-core==1.4.0, langchain-community==0.4.1
langchain-chroma==1.1.0, langchain-text-splitters==1.1.2, langchain-classic==1.0.7, langchain-protocol==0.0.15
chromadb==1.5.9, rank-bm25
sentence-transformers==5.5.0, transformers==5.8.1, sentencepiece==0.2.1
scikit-learn==1.7.2, numpy==1.26.4, pandas==2.2.3, pillow==12.2.0
dashscope==1.25.18, openai==2.42.0, tavily-python==0.7.24, onnxruntime==1.18.1
```
torch 单独装 CPU 版（见 Dockerfile）。

### 2. `Dockerfile`（新建）
- 基础镜像 `python:3.10-slim`
- 装系统库：`build-essential libgl1 libglib2.0-0`（OpenCV/PIL/matplotlib 需要 libgl1）
- **先装 torch CPU**：`pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu`（避免拉 2.5GB CUDA 轮子；本地是 `2.12.0+cpu`）
- `COPY requirements.txt` + `pip install`（独立层，利用 build cache）
- `COPY . .`
- `ENV`：`HF_HOME=/app/.cache/huggingface` `HF_ENDPOINT=https://hf-mirror.com` `KMP_DUPLICATE_LIB_OK=TRUE` `PYTHONUNBUFFERED=1`
- `EXPOSE 5000`
- `CMD ["python", "api_server_fastapi.py"]`（保留你熟悉的入口）

### 3. `.dockerignore`（新建）
排除：`.git/` `__pycache__/` `*.pyc` `.env`（**密钥绝不进镜像**）`参考项目/` `learning/` `LLM-wiki` `mcp-learning/` `eval/` `eval_rag25/` `tests/` `legacy/` `logs/` `server.log` `runtime/` `data/uploads/` `*.md` `.codegraph/` `.idea/` `.vscode/` `*.bat` `Thumbs.db`
保留：`chroma_db/` `data/`（不含 uploads）`manifest.json` `parent_docstore.json` `ultralytics/` `Detct_prdc/` `prompts/` `config/` `api/` `agent/` `rag/` `model/` `crud/` `models/` `schemas/` `services/` `utils/` `css/` `js/` `assets/` `api_server_fastapi.py`

### 4. `docker-compose.yml`（新建）
- `db`：mysql:8.0，env `MYSQL_ROOT_PASSWORD=root` `MYSQL_DATABASE=sar_sense`，command 强制 utf8mb4，healthcheck 用 `mysql -uroot -proot -e 'USE sar_sense'`（确保库已建，不只 ping），named volume `mysql_data`
- `app`：build `.`，端口 5000:5000，env 注入 `MYSQL_HOST=db MYSQL_PORT=3306 MYSQL_USER=root MYSQL_PASSWORD=root MYSQL_DATABASE=sar_sense` + `${DASHSCOPE_API_KEY}` `${TAVILY_API_KEY}`（从宿主 .env 读），`depends_on.db.condition: service_healthy`，volumes：`app_uploads:/app/data/uploads` `app_logs:/app/logs` `app_runtime:/app/runtime` `hf_cache:/app/.cache/huggingface`
- 顶层 `volumes:` 声明 5 个命名卷

### 5. `.env.example` 补充
现有已记录 `DASHSCOPE_API_KEY` `TAVILY_API_KEY`。补一段说明：Docker 部署时把这两个填进 `.env`，compose 自动读取注入容器；MySQL 连接参数由 compose 内置，无需手动设。

## 要改的代码（3 处，都是小手术，保留本地默认值不破坏现状）

### 改动 1：`config/db_conf.py:9`
```python
# 现在：写死 localhost:3306/root:root
ASYNC_DATABASE_URL = "mysql+aiomysql://root:root@localhost:3306/sar_sense?charset=utf8mb4"
# 改成：环境变量驱动，默认值 = 本地现状（本地不设 env 照常工作）
import os
ASYNC_DATABASE_URL = (
    f"mysql+aiomysql://{os.getenv('MYSQL_USER','root')}:{os.getenv('MYSQL_PASSWORD','root')}"
    f"@{os.getenv('MYSQL_HOST','localhost')}:{os.getenv('MYSQL_PORT','3306')}"
    f"/{os.getenv('MYSQL_DATABASE','sar_sense')}?charset=utf8mb4"
)
```

### 改动 2：`crud/metrics.py:11-16`
```python
# 现在：host="localhost" 写死
# 改成：读同一组 MYSQL_* 环境变量（同步 pymysql）
def _conn():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "root"),
        database=os.getenv("MYSQL_DATABASE", "sar_sense"),
        charset="utf8mb4", autocommit=True,
    )
```

### 改动 3：`config/rag.yml:4`
```yaml
# 现在：reranker_model_name: E:/models/Xorbits/bge-reranker-base  （本地 Windows 路径，容器里不存在）
# 改成：BAAI/bge-reranker-base —— reranker.py 已有两级回退 + HF_ENDPOINT=hf-mirror.com，首次启动自动从镜像站下载（约 1.1GB），缓存进 hf_cache 卷，之后秒加载
reranker_model_name: BAAI/bge-reranker-base
```

## 不用改的（已确认 Docker 友好）
- `utils/path_tool.py` 用 `__file__` 算项目根目录 —— 容器 WORKDIR=/app + COPY 后自动解析，所有 `get_abs_path("config/xxx.yml")` 正常
- `api/app.py` 静态目录用相对路径 `css/js/assets` —— WORKDIR 下正常
- 表自动建表 + 种子 admin —— MySQL 空库就绪后应用启动自动 `Base.metadata.create_all`
- BM25 索引：runtime/ 不打进镜像（Windows pickle 跨平台有风险），容器首次启动从 parent_docstore.json 重建（秒级）

## 预期体积
- 镜像约 2.5–3.5GB（torch + transformers + chromadb + sentence-transformers 是大头，ML 项目正常水平，不是问题）
- 首次 build 约 5–10 分钟；首次 `compose up` 时 db 初始化 + BGE 模型下载（~1.1GB，进 hf_cache 卷）约 3–5 分钟，之后重启都是秒级

## 用户操作步骤（完成后）
1. 在项目根目录 `.env` 填 `DASHSCOPE_API_KEY` 和 `TAVILY_API_KEY`
2. `docker compose up --build`（首次）
3. 等 app 日志出现 `Uvicorn running on http://0.0.0.0:5000`
4. 浏览器打开 `http://localhost:5000`，用 `admin/admin123` 登录
5. 之后日常 `docker compose up` / `docker compose down`

## 验证清单
- [ ] `docker compose up --build` 全程无报错
- [ ] `docker compose logs app` 出现 uvicorn 启动
- [ ] 浏览器 5000 能登录、知识库列表非空、对话能搜到 KB、SAR 图片能检测
- [ ] `docker compose down` 后再 `up`，知识库/上传文件仍在（volume 持久）
- [ ] 本地 `python api_server_fastapi.py`（不设 MYSQL_* env）仍正常 —— 验证默认值未破坏本地开发
