from mcp.server import MCPServer


mcp = MCPServer(
    "sar-sense-detection-metrics",
    description="SAR-Sense 目标检测指标计算示例",
)


@mcp.tool()
def calculate_detection_metrics(tp: int, fp: int, fn: int) -> str:
    """根据 TP、FP、FN 计算 Precision、Recall 和 F1。"""
    if min(tp, fp, fn) < 0:
        raise ValueError("TP、FP、FN 不能为负数")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return (
        f"Precision={precision:.2%}, "
        f"Recall={recall:.2%}, "
        f"F1={f1:.2%}"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
