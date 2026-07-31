"""SAR-Sense API 包。

应用入口位于 ``api.app``。这里不提前导入 FastAPI app，避免普通服务模块
导入 ``api.dependencies`` 时触发整套路由注册并形成循环依赖。
"""
