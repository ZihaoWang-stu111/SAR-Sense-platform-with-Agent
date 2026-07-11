"""Redis 限流 + Redis 分布式锁。

== Redis 分布式锁学习要点 ==
1. 加锁原子性：`SET key value NX EX timeout`
   - NX：key 不存在才设置，保证互斥
   - EX timeout：设置自动过期时间，防止持锁进程崩溃后形成死锁
   - 检查不存在、写入 token、设置过期时间由一条命令原子完成
   - 不能使用 GET 判断后再 SET，因为两步之间可能被其他请求插入

2. value 使用唯一 token（UUID），标识「本次」持锁者
   - 释放时必须对比 token
   - 防止出现：
       A 的锁过期
       B 获得新锁
       A 恢复后误删 B 的锁

3. 释放原子性：Lua 把 GET + 对比 token + DEL 合成一个原子操作
   - token 一致：删除锁
   - token 不一致：不做任何操作


== Redis 限流（固定窗口）==
1. 每一个限流维度对应一个 Redis key，例如：

       rl:user:5:chat
       rl:ip:1.2.3.4:login

2. 每次请求执行 INCR，原子地将计数加 1。

3. 第一次请求时设置 EXPIRE，窗口结束后计数器自动删除。

4. 使用 Lua 把 INCR + EXPIRE 合成一个原子操作，避免：

       INCR 成功
       程序突然崩溃
       EXPIRE 没执行
       key 永久不过期

5. count > max_calls 时返回 HTTP 429。

6. 固定窗口实现简单、性能高，但窗口交界处可能出现突刺。
   滑动窗口 ZSET 更准确，但更复杂；普通业务场景通常没必要。


== 降级 ==
TRAFFIC_CONTROL_ENABLED=false 时：
- 限流直接跳过
- 分布式锁变成 no-op
- 本地开发可以不依赖 Redis
"""

import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(override=True)

import redis.asyncio as aioredis
from fastapi import HTTPException


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

TRAFFIC_CONTROL_ENABLED = (
    os.getenv("TRAFFIC_CONTROL_ENABLED", "true").lower() != "false"
)


# 连接池在当前 Python 进程内共享，避免每次请求重新建立 Redis 连接
_pool = aioredis.ConnectionPool.from_url(
    REDIS_URL,
    decode_responses=True,
)


def get_redis() -> aioredis.Redis:
    """返回使用共享连接池的异步 Redis 客户端。"""
    return aioredis.Redis(connection_pool=_pool)


async def ping_redis() -> bool:
    """Redis 健康检查。

    Redis 可用返回 True，不可用返回 False。

    生产环境建议在应用启动时调用：
    Redis 不可用时终止启动，而不是静默关闭限流和分布式锁。
    """
    try:
        return bool(await get_redis().ping())
    except Exception:
        return False


# ============================================================
# Redis 固定窗口限流
# ============================================================

# 原子执行：
#
# 1. 将计数器加 1
# 2. 如果这是第一次请求，则设置窗口过期时间
# 3. 返回当前计数和剩余 TTL
#
# Lua 脚本执行期间，其他 Redis 命令不能插入脚本中间。
_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])

if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end

local ttl = redis.call('TTL', KEYS[1])

-- 如果因为旧代码或异常情况导致 key 没有过期时间，
-- 则重新补上过期时间，避免计数器永久存在。
if ttl < 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end

return {count, ttl}
"""


async def rate_limit(
    key: str,
    max_calls: int,
    window: int,
) -> None:
    """Redis 固定窗口限流，超限时抛出 HTTP 429。

    Args:
        key:
            限流维度键，不需要手动添加 rl: 前缀。

            例如：
                user:5:chat
                user:5:upload
                ip:1.2.3.4:login

        max_calls:
            一个窗口内最多允许通过的请求次数。

        window:
            窗口时长，单位为秒。

    示例：
        await rate_limit(
            key=f"user:{user_id}:chat",
            max_calls=20,
            window=60,
        )

        表示该用户在 60 秒内最多调用聊天接口 20 次。
    """
    if not TRAFFIC_CONTROL_ENABLED:
        return

    if not key:
        raise ValueError("限流 key 不能为空")

    if max_calls <= 0:
        raise ValueError("max_calls 必须大于 0")

    if window <= 0:
        raise ValueError("window 必须大于 0")

    r = get_redis()
    redis_key = f"rl:{key}"

    # eval 参数含义：
    #
    # _RATE_LIMIT_SCRIPT：要执行的 Lua 脚本
    # 1：脚本中有一个 KEYS 参数
    # redis_key：对应 Lua 中的 KEYS[1]
    # window：对应 Lua 中的 ARGV[1]
    count, ttl = await r.eval(
        _RATE_LIMIT_SCRIPT,
        1,
        redis_key,
        window,
    )

    count = int(count)
    ttl = max(int(ttl), 1)

    if count > max_calls:
        raise HTTPException(
            status_code=429,
            detail=f"请求过于频繁，{window}秒内限{max_calls}次",
            headers={
                # 告诉客户端大约等待多少秒后再尝试
                "Retry-After": str(ttl),
            },
        )


# ============================================================
# Redis 分布式锁
# ============================================================

# Lua 原子释放脚本：
#
# 1. GET 获取当前锁保存的 token
# 2. 与当前持锁者的 token 对比
# 3. 一致才删除
#
# 为什么不能在 Python 中分开写：
#
#     value = await r.get(lock_key)
#     if value == token:
#         await r.delete(lock_key)
#
# 因为 GET 和 DELETE 之间，锁可能刚好过期，
# 另一个请求可能已经获得了同名新锁。
_UNLOCK_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
else
    return 0
end
"""


@asynccontextmanager
async def redis_lock(
    lock_key: str,
    timeout: int = 60,
):
    """Redis 非阻塞分布式锁。

    进入 async with 时尝试获取锁；
    yield 期间表示当前任务持有锁；
    退出 async with 时自动安全释放锁。

    Args:
        lock_key:
            锁名。

            例如：
                lock:ingest:user:5
                lock:document:123
                lock:knowledge-base:8:rebuild

            相同 lock_key 之间互斥；
            不同 lock_key 可以并行执行。

        timeout:
            锁自动过期时间，单位为秒。

            它是防止进程崩溃后产生永久死锁的兜底时间，
            应当大于业务可能的最长执行时间，并留出余量。

    Raises:
        HTTPException(409):
            相同 lock_key 已经被其他任务持有。

    注意：
        当前实现没有自动续期。

        如果业务执行时间超过 timeout：

            A 的锁自动过期
            B 获得同名锁
            A 和 B 可能同时执行业务

        唯一 token 只能防止 A 最后误删 B 的锁，
        无法阻止锁过期后已经发生的重复执行。
    """
    if not TRAFFIC_CONTROL_ENABLED:
        yield
        return

    if not lock_key:
        raise ValueError("lock_key 不能为空")

    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")

    r = get_redis()

    # 唯一标识本次持锁者
    token = uuid.uuid4().hex

    # 原子加锁：
    #
    # NX：只有 key 不存在时才设置成功
    # EX：同时设置自动过期时间
    acquired = await r.set(
        lock_key,
        token,
        nx=True,
        ex=timeout,
    )

    if not acquired:
        raise HTTPException(
            status_code=409,
            detail="已有同类任务在进行中，请稍后再试",
        )

    try:
        # async with 内部的业务代码从这里开始执行
        yield

    finally:
        # 原子释放：
        #
        # Redis 中保存的 token 仍然等于本次 token 时才删除。
        # 如果锁已经过期，或者已经被其他请求重新获得，则不会删除。
        await r.eval(
            _UNLOCK_SCRIPT,
            1,
            lock_key,
            token,
        )