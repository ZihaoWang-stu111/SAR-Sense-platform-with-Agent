# Nginx 使用指南

## 快速开始

### 1. 下载 Nginx
```
https://nginx.org/en/download.html
下载稳定版 (Stable version)
解压到任意目录，比如 C:\nginx
```

### 2. 使用项目配置
```bash
# 复制配置文件到 Nginx 目录
copy nginx.conf C:\nginx\conf\nginx.conf

# 进入 Nginx 目录
cd C:\nginx

# 启动 Nginx
nginx

# 浏览器访问 http://localhost
```

### 3. 常用命令
```bash
# 启动
nginx

# 停止（强制）
nginx -s stop

# 优雅停止（处理完当前请求）
nginx -s quit

# 重新加载配置（改完配置后执行，不中断服务）
nginx -s reload

# 测试配置文件是否正确
nginx -t

# 查看版本
nginx -v
```

---

## 配置文件结构说明

```
nginx.conf
├── 全局配置
│   ├── worker_processes    # 进程数
│   └── events              # 事件模型
│
└── http 块
    ├── 日志格式
    ├── 限流配置
    ├── upstream            # 后端服务器组
    └── server              # 虚拟主机
        ├── listen          # 监听端口
        ├── location /css/  # CSS 静态文件
        ├── location /js/   # JS 静态文件
        ├── location /api/  # API 反向代理
        └── location /      # 默认路由
```

---

## 面试常见问题

### Q1: location 匹配顺序？
```
= 精确匹配    >  location = /api  → 只匹配 /api
^~ 前缀匹配   >  location ^~ /css → 匹配 /css 开头
~ 正则匹配    >  location ~ \.html$ → 匹配 .html 结尾
/ 通用匹配    >  location / → 兜底
```

### Q2: alias 和 root 的区别？
```nginx
# root 会保留 location 路径
location /css/ {
    root /var/www;   # 实际路径: /var/www/css/
}

# alias 会替换 location 路径
location /css/ {
    alias /var/www/assets/css/;  # 实际路径: /var/www/assets/css/
}
```

### Q3: proxy_pass 尾斜杠的区别？
```nginx
# 无尾斜杠：保留原路径
location /api/ {
    proxy_pass http://backend;   # /api/users → http://backend/api/users
}

# 有尾斜杠：替换路径
location /api/ {
    proxy_pass http://backend/;  # /api/users → http://backend/users
}
```

### Q4: 为什么关闭 proxy_buffering？
```
SSE（Server-Sent Events）需要实时推送数据。
如果开启缓冲，Nginx 会等数据积攒到一定量才发送给客户端，
导致消息延迟。关闭缓冲后，数据立即转发。
```

---

## 测试你的配置

### 测试1：静态文件是否生效
```
访问 http://localhost/css/style.css
应该能看到你的 CSS 文件内容
```

### 测试2：API 反向代理是否生效
```bash
# 先启动 FastAPI
python api_server_fastapi.py

# 访问 API
curl http://localhost/api/health
```

### 测试3：限流是否生效
```bash
# 快速发送大量请求
for i in {1..100}; do curl http://localhost/api/health & done

# 观察是否有 503 错误（被限流）
```

---

## 生产环境建议

| 配置项 | 开发环境 | 生产环境 |
|--------|----------|----------|
| worker_processes | 1 | auto (CPU核心数) |
| worker_connections | 1024 | 4096+ |
| SSL | 关闭 | 开启 (HTTPS) |
| 日志级别 | debug | error |
| 缓存 | 关闭 | 开启 |
| 限流 | 宽松 | 严格 |

---

## 常见问题排查

### 问题1：端口被占用
```bash
# 查看 80 端口占用
netstat -ano | findstr :80

# 修改 nginx.conf 中的 listen 端口
listen 8080;  # 改成其他端口
```

### 问题2：配置文件语法错误
```bash
# 测试配置
nginx -t

# 错误信息会指出哪一行有问题
```

### 问题3：页面或静态资源 404
```bash
# 当前配置把页面、静态资源和 API 都代理给 FastAPI。
# 先确认后端可访问：
curl http://127.0.0.1:5000/api/health
```
