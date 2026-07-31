import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

# 加载本地 .env；系统环境变量优先，便于 Docker/生产平台安全注入配置。
load_dotenv()

def load_rag_config(config_path = get_abs_path("config/rag.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)

def load_chroma_config(config_path = get_abs_path("config/chroma.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)

def load_prompts_config(config_path = get_abs_path("config/prompts.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)

def load_agent_config(config_path = get_abs_path("config/agent.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.safe_load(f)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
