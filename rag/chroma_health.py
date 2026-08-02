class ChromaHealthError(RuntimeError):
    """Chroma 元数据存在，但持久化向量或 HNSW 无法正常使用。"""


def check_collection_health(
    collection,
    expected_count: int | None = None,
    *,
    exhaustive: bool = False,
    query_batch_size: int = 100,
) -> dict:
    """读取一条已存向量并反查自身，验证持久化数据和 HNSW 都可用。"""
    count = collection.count()
    if expected_count is not None and count != expected_count:
        raise ChromaHealthError(
            f"Chroma 记录数不一致：expected={expected_count}, actual={count}"
        )
    if count == 0:
        return {"count": 0, "probe_id": None, "dimension": None}

    try:
        get_kwargs = {"include": ["embeddings", "metadatas"]}
        if not exhaustive:
            get_kwargs["limit"] = 1
        sample = collection.get(**get_kwargs)
    except Exception as exc:
        raise ChromaHealthError(f"Chroma 无法读取已存 embedding：{exc}") from exc

    ids = sample.get("ids") or []
    embeddings = sample.get("embeddings")
    if not ids or embeddings is None or len(embeddings) == 0:
        raise ChromaHealthError("Chroma 有记录，但无法读取一条已存 embedding")

    if exhaustive and len(ids) != count:
        raise ChromaHealthError(
            f"Chroma 无法读取全部 embedding：expected={count}, actual={len(ids)}"
        )

    query_embeddings = []
    for probe_id, embedding in zip(ids, embeddings):
        if embedding is None or len(embedding) == 0:
            raise ChromaHealthError(f"Chroma 中 {probe_id} 的 embedding 为空")
        query_embeddings.append(
            embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        )

    for start in range(0, len(ids), query_batch_size):
        batch_ids = ids[start : start + query_batch_size]
        batch_embeddings = query_embeddings[start : start + query_batch_size]
        try:
            result = collection.query(
                query_embeddings=batch_embeddings,
                n_results=1,
                include=["distances"],
            )
        except Exception as exc:
            raise ChromaHealthError(f"Chroma HNSW 查询失败：{exc}") from exc

        result_ids = result.get("ids") or []
        result_distances = result.get("distances") or []
        if len(result_ids) != len(batch_ids):
            raise ChromaHealthError("Chroma HNSW 查询返回数量异常")
        for index, (probe_id, nearest_ids) in enumerate(zip(batch_ids, result_ids)):
            if not nearest_ids:
                raise ChromaHealthError(
                    f"Chroma HNSW 查询未返回结果：probe={probe_id}"
                )
            if nearest_ids[0] == probe_id:
                continue
            nearest_distance = None
            if index < len(result_distances) and result_distances[index]:
                nearest_distance = result_distances[index][0]
            if nearest_distance is None or nearest_distance > 1e-6:
                raise ChromaHealthError(
                    "Chroma 自查询结果异常："
                    f"probe={probe_id}, result={nearest_ids[0]}, "
                    f"distance={nearest_distance}"
                )

    return {
        "count": count,
        "probe_id": ids[0],
        "dimension": len(query_embeddings[0]),
    }
