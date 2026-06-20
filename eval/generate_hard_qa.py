"""
生成 Tier-2 hard 题集：多段综合 + 改写式 + 跨文件 comparison

每题都自动验证 gold_snippet 在对应 source file 里 verbatim 出现（NFKC + 去空白后的子串包含）。
不通过的题打印失败，方便修订。

输出：eval/qa_dataset_v2.json  (20 题，hard tier，独立于现有 50 题)
"""
import json
import os
import sys
import io
import unicodedata
from pathlib import Path

# Windows GBK 终端无法显示 emoji，统一改用 utf-8 输出
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 三类难题
# M = Multi-paragraph synthesis (问题需要 2+ 段拼答)
# P = Paraphrased (问题用 A 词，答案在原文用 B 词)
# C = Cross-file comparison (答案分散在 2+ 文件)

QUESTIONS = [
    # ================== 多段综合 M (8 题) ==================
    {
        "id": "qa_h01",
        "question": "MBE-Net 是如何同时对抗 SAR 图像中散斑噪声和复杂背景干扰的？",
        "gold_answer": "EEWB 通过小波池化抑制散斑；ACF-FPN 通过上下文引导融合增强复杂背景鲁棒性",
        "gold_filename": "MBE-Net算法详解.txt",
        "gold_snippet": "有效抑制散斑噪声，同时保持边缘锐度",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h02",
        "question": "MBE-Net 的 LD-DEH 检测头是怎么同时做到轻量化和高定位精度的？",
        "gold_answer": "任务解耦设计 + 三阶段细节增强 + 推理时权重融合",
        "gold_filename": "MBE-Net算法详解.txt",
        "gold_snippet": "支持在推理阶段将多个卷积层融合为单一操作",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h03",
        "question": "MBE-Net 的训练采用了什么基础模型、多少轮、什么优化器和学习率？",
        "gold_answer": "基础模型 YOLO11n，350 epochs，SGD 优化器，初始学习率 0.01",
        "gold_filename": "MBE-Net算法详解.txt",
        "gold_snippet": "初始学习率：0.01",
        "qa_type": "fact",
        "_hardness": "M",
    },
    {
        "id": "qa_h04",
        "question": "SAR 图像有哪些主要的几何畸变？这些畸变在海洋检测里是否需要处理？",
        "gold_answer": "透视收缩、叠掩、阴影；在海洋场景影响小，但近岸需要注意",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "这些畸变在海洋场景中影响相对较小",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h05",
        "question": "为什么港口区域比开阔海域更难做 SAR 舰船检测？",
        "gold_answer": "港口设施产生强杂波；舰船密集停靠相互遮挡 NMS 容易抑制相邻目标",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "NMS（非极大值抑制）可能错误地抑制相邻目标",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h06",
        "question": "YOLO11 在训练策略上相比早期版本做了哪些优化？",
        "gold_answer": "马赛克数据增强、混合增强、多尺度训练、余弦退火、EMA 参数平滑",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "EMA（指数移动平均）模型参数平滑",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h07",
        "question": "YOLO 在 SAR 舰船检测中的主要挑战是什么？该怎么改进？",
        "gold_answer": "小目标、密集场景、散斑噪声；引入注意力、定制 backbone、改 FPN、知识蒸馏",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "采用知识蒸馏等轻量化技术",
        "qa_type": "concept",
        "_hardness": "M",
    },
    {
        "id": "qa_h08",
        "question": "HRSID 和 SSDD 在数据规模、传感器来源、分辨率上有什么差异？",
        "gold_answer": "HRSID 来自 Sentinel-1/TerraSAR-X 等高分辨率切片；SSDD 多源（含 RadarSat-2）含 1m-15m 多分辨率，1160 张图 2456 个目标",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "图像数量：1160张SAR图像",
        "qa_type": "comparison",
        "_hardness": "M",
    },

    # ================== 改写式 P (6 题) - question 用近义词，逼测召回鲁棒性 ==================
    {
        "id": "qa_h09",
        # paraphrase: "颗粒状噪声" 改成 "纹理斑驳"，"乘性噪声模型" 改成 "服从乘法关系"
        "question": "为什么 SAR 图像看起来纹理斑驳？这种噪声服从乘法关系还是加法关系？",
        "gold_answer": "由于多个散射体回波相干叠加，呈颗粒状，服从乘性噪声模型",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "表现为图像中的颗粒状噪声，服从乘性噪声模型",
        "qa_type": "fact",
        "_hardness": "P",
    },
    {
        "id": "qa_h10",
        # paraphrase: "全天候" 改成 "不挑天气", "全天时" 改成 "白天黑夜均可"
        "question": "SAR 为什么白天黑夜、不挑天气都能成像？",
        "gold_answer": "微波能穿透云层、雨雾，不受光照条件限制",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "微波能够穿透云层、雨雾和一定程度的植被",
        "qa_type": "concept",
        "_hardness": "P",
    },
    {
        "id": "qa_h11",
        # paraphrase: "解耦头" 改成 "分支独立", "分类回归任务" 改成 "判类别和定位置"
        "question": "YOLO 的检测头中，判断类别的部分和确定位置的部分为什么要分支独立？",
        "gold_answer": "解耦设计避免分类和回归的特征冲突，提升各自性能",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "解耦头（Decoupled Head）将分类和回归任务分离",
        "qa_type": "concept",
        "_hardness": "P",
    },
    {
        "id": "qa_h12",
        # paraphrase: "AP_large/medium/small" 用大小目标的具体面积阈值反问
        "question": "评估 SAR 检测时，所谓「中等目标」指的是面积在多少像素之间？",
        "gold_answer": "32² < 面积 < 96² 像素",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "AP_medium：中等目标（32²<面积<96²像素）的AP",
        "qa_type": "fact",
        "_hardness": "P",
    },
    {
        "id": "qa_h13",
        # paraphrase: "AIS" 是术语，问"自动识别系统"
        "question": "OpenSARShip 数据集的特殊之处是它带有什么辅助标注信息？",
        "gold_answer": "AIS（自动识别系统）信息",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "包含AIS信息作为辅助标注",
        "qa_type": "fact",
        "_hardness": "P",
    },
    {
        "id": "qa_h14",
        # paraphrase: "马赛克增强" 用"四张图拼接"反问
        "question": "在 YOLO 训练里，把四张图拼成一张的数据增强方式叫什么？",
        "gold_answer": "马赛克数据增强（Mosaic Augmentation）",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "马赛克数据增强（Mosaic Augmentation）：将四张图像拼接为一张",
        "qa_type": "fact",
        "_hardness": "P",
    },

    # ================== 跨文件 / 跨段 comparison C (6 题) ==================
    {
        "id": "qa_h15",
        # 答案在 MBE.txt + YOLO.txt 都有 hint，gold 选 YOLO.txt 的具体参数
        "question": "MBE-Net 基于的 YOLO11n 参数量大约是多少？",
        "gold_answer": "约 2.6M",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "YOLO11n（Nano）：最轻量版本，参数量约2.6M",
        "qa_type": "fact",
        "_hardness": "C",
    },
    {
        "id": "qa_h16",
        # 跨段 comparison: 数据集 + 评估指标段
        "question": "HRSID 数据集典型的图像切片尺寸是多少？",
        "gold_answer": "通常为 800×800 像素",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "图像尺寸：通常为800×800像素的切片",
        "qa_type": "fact",
        "_hardness": "C",
    },
    {
        "id": "qa_h17",
        # 跨段：CFAR 是传统方法 + 应用在 SAR
        "question": "传统 SAR 舰船检测方法 CFAR 的核心思路是什么？",
        "gold_answer": "基于背景杂波统计模型，自适应设定虚警阈值",
        "gold_filename": "SAR舰船检测概述.txt",
        "gold_snippet": "通过估计背景杂波分布设定自适应阈值",
        "qa_type": "concept",
        "_hardness": "C",
    },
    {
        "id": "qa_h18",
        # 跨文件: 在 SAR 概述里说"小目标特征丢失", 在 SAR 特性里说"多次下采样后丢失"
        "question": "为什么 SAR 中的小型舰船经过多次下采样后容易彻底丢失？",
        "gold_answer": "多次下采样后小目标特征可能完全丢失",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "多次下采样后小目标特征可能完全丢失",
        "qa_type": "concept",
        "_hardness": "C",
    },
    {
        "id": "qa_h19",
        # 改写：原文用"边界框"，问"目标定位框"
        "question": "DFL Loss 把目标定位框的坐标建模成什么形式？",
        "gold_answer": "建模为离散概率分布",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "将边界框坐标建模为离散概率分布",
        "qa_type": "concept",
        "_hardness": "P",
    },
    {
        "id": "qa_h20",
        # 跨文件应用场景列表
        "question": "SAR 舰船检测在民用领域典型的应用场景有哪些？",
        "gold_answer": "海上交通监控、渔业管理、海上搜救、海洋环境保护",
        "gold_filename": "SAR舰船检测概述.txt",
        "gold_snippet": "渔业管理：监测非法捕捞活动",
        "qa_type": "concept",
        "_hardness": "M",
    },
]

# ============= 验证 + 写出 =============
def normalize(s: str) -> str:
    """与 eval/metrics.py 的 is_hit 完全一致"""
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(s.split())  # 去掉所有空白
    return s.lower()


def verify():
    data_dir = Path('data')
    failures = []
    for q in QUESTIONS:
        src_path = data_dir / q['gold_filename']
        if not src_path.exists():
            failures.append((q['id'], f'文件不存在: {src_path}'))
            continue
        content = src_path.read_text(encoding='utf-8')
        norm_content = normalize(content)
        norm_snippet = normalize(q['gold_snippet'])
        if norm_snippet not in norm_content:
            failures.append((q['id'], f'snippet 不在原文里: "{q["gold_snippet"][:40]}..."'))
    return failures


def main():
    failures = verify()
    if failures:
        print('❌ 有题目验证失败：')
        for qid, reason in failures:
            print(f'  {qid}: {reason}')
        return

    out_path = Path('eval/qa_dataset_v2.json')
    # 输出时去掉 _hardness 字段（仅供我们参考）
    output = []
    for q in QUESTIONS:
        clean = {k: v for k, v in q.items() if not k.startswith('_')}
        output.append(clean)

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    # 统计
    from collections import Counter
    by_hardness = Counter(q['_hardness'] for q in QUESTIONS)
    by_type = Counter(q['qa_type'] for q in QUESTIONS)
    by_file = Counter(q['gold_filename'] for q in QUESTIONS)

    print(f'✅ 已写出 {len(output)} 题到 {out_path}')
    print()
    print(f'按难题类型：{dict(by_hardness)}')
    print(f'  M = multi-paragraph synthesis (多段综合)')
    print(f'  P = paraphrased (改写式)')
    print(f'  C = cross-file / cross-section comparison (跨文件)')
    print()
    print(f'按 qa_type：{dict(by_type)}')
    print(f'按文件：')
    for fn, n in by_file.most_common():
        print(f'  {n}/{len(QUESTIONS)}  {fn}')


if __name__ == '__main__':
    main()
