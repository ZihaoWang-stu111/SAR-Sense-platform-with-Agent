import yaml
from dotenv import load_dotenv
from utils.path_tool import get_abs_path

# 加载 .env 到环境变量（API key 等敏感配置不进代码库）。
# 放在 config_handler 顶部：任何入口只要 import 配置，.env 就已加载，不用每个入口重复写。
# override=True：.env 的值覆盖系统环境变量，保证配置以 .env 为准。
load_dotenv(override=True)

def load_rag_config(config_path = get_abs_path("config/rag.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_chroma_config(config_path = get_abs_path("config/chroma.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path = get_abs_path("config/prompts.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_agent_config(config_path = get_abs_path("config/agent.yml"), encoding = "UTF-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == '__main__':
   print(rag_conf["chat_model_name"])
   print(rag_conf["embedding_name"])