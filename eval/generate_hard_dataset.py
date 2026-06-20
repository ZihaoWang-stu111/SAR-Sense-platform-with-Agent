"""
构造 hard-only 题集：
  - 删掉原 50 题里 38 道"过于简单"的（hybrid_pc_no_rr 在 k=10 时 rank=1 的题）
  - 保留 12 道现有困难题
  - 新增 15 道针对 reranker 设计的难题
                       （改写式 + 多 chunk 干扰 + 综合推理）
  - 全部 gold_snippet 自动校验存在于 data/ 文件中

输出：eval/qa_dataset_hard.json （27 题）
"""
import json
import sys
import io
import csv
import unicodedata
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 第 1 步：从 results_raw.csv 找出 38 道"简单题"（hybrid_pc_no_rr@k=10 时 rank=1）
# ============================================================

def load_easy_qa_ids():
    """返回 hybrid_pc_no_rr 在 k=10 时 rank=1 的 qa_id 列表（即简单题）。"""
    easy = set()
    with open(ROOT / 'eval/results/results_raw.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['pipeline'] != 'hybrid_pc_no_rr' or int(row['k']) != 10:
                continue
            try:
                rank = int(float(row['first_hit_rank']))
            except (ValueError, TypeError):
                rank = None
            if rank == 1:
                easy.add(row['qa_id'])
    return easy


# ============================================================
# 第 2 步：15 道新难题 —— 针对 reranker 设计
# ============================================================
# 设计原则:
#   ✏️ P (改写式): query 用近义词，BM25 token 不匹配，vector 接近不准，靠 rerank 救
#   🎭 D (干扰陷阱): corpus 里多个 chunk 都"看着像答案"，rerank 必须区分
#   🔗 S (综合): query 需要的信息分散在多个段落里
#
# 每道题的 gold_snippet 都从 data/ 文件里精确摘录，是答案最 pivotal 的那个词组。

NEW_HARD = [
    # ============== ✏️ 改写式 (6 题) ==============
    {
        "id": "qa_d01",
        # paraphrase: 全天候/全天时 → 恶劣天气下/不挑光线
        "question": "为什么 SAR 在恶劣天气和黑夜里都还能成像？",
        "gold_answer": "微波穿透云层雨雾，不受光照限制，主动雷达不依赖外部光源",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "不受光照条件限制，可实现全天时成像",
        "qa_type": "concept",
        "_design": "P (paraphrase: 恶劣天气↔全天候)",
    },
    {
        "id": "qa_d02",
        # paraphrase: 几何增强 ↔ 翻转/旋转/缩放
        "question": "把图像随机翻转、旋转、缩放算哪一类数据增强？",
        "gold_answer": "几何增强",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "几何增强：随机翻转（水平/垂直）、旋转",
        "qa_type": "fact",
        "_design": "P (paraphrase: 几何变换 ↔ 翻转/旋转)",
    },
    {
        "id": "qa_d03",
        # paraphrase: 推理时权重融合 ↔ 部署时把卷积合并
        "question": "把多个卷积层在部署时合并成一个，能带来什么好处？",
        "gold_answer": "减少推理计算开销加速检测，且不损失精度",
        "gold_filename": "MBE-Net算法详解.txt",
        "gold_snippet": "减少推理时的计算开销，加速检测速度",
        "qa_type": "concept",
        "_design": "P (paraphrase: 部署时合并 ↔ 推理时权重融合)",
    },
    {
        "id": "qa_d04",
        # paraphrase: 长航迹合成 ↔ 飞行中
        "question": "SAR 是怎么从一个小天线得到等效大天线分辨率的？",
        "gold_answer": "雷达平台飞行过程中沿航迹方向把多个回波合成等效长孔径天线",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "将沿航迹方向（方位向）的多个回波合成为等效长孔径天线",
        "qa_type": "concept",
        "_design": "P (paraphrase: 小天线变大 ↔ 合成孔径)",
    },
    {
        "id": "qa_d05",
        # paraphrase: 散斑 ↔ 像沙子一样的纹理
        "question": "SAR 图像里那种粒状纹理是什么噪声？怎么形成的？",
        "gold_answer": "相干斑噪声，由分辨单元内多散射体回波相干叠加产生",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "由雷达分辨单元内多个散射体回波的相干叠加产生",
        "qa_type": "concept",
        "_design": "P (paraphrase: 粒状纹理 ↔ 散斑)",
    },
    {
        "id": "qa_d06",
        # paraphrase: 端到端 ↔ 一步到位/无候选区域
        "question": "YOLO 不需要先找候选区域再分类，这样设计有什么好处？",
        "gold_answer": "端到端训练 + 推理速度快",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "端到端训练：无需额外的候选区域生成步骤",
        "qa_type": "concept",
        "_design": "P (paraphrase: 一步到位 ↔ 端到端)",
    },

    # ============== 🎭 多 chunk 干扰 (5 题) ==============
    {
        "id": "qa_d07",
        # 干扰: EEWB / ACF-FPN / LD-DEH 三模块都说自己增强、轻量化、提升精度
        "question": "MBE-Net 中哪个模块负责把不同尺度的特征融合到一起？",
        "gold_answer": "ACF-FPN（自适应上下文聚焦特征金字塔网络）",
        "gold_filename": "MBE-Net算法详解.txt",
        "gold_snippet": "ACF-FPN是MBE-Net的特征融合颈部网络",
        "qa_type": "fact",
        "_design": "D (干扰: 三模块 chunk 都很相似)",
    },
    {
        "id": "qa_d08",
        # 干扰: 损失函数 chunk 里 CIoU / DFL / BCE / VFL 都被提及
        "question": "YOLO 用来把边界框坐标变成概率分布的损失函数叫什么？",
        "gold_answer": "DFL Loss（Distribution Focal Loss）",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "DFL Loss（Distribution Focal Loss）",
        "qa_type": "fact",
        "_design": "D (干扰: 多种损失函数 chunk 提及)",
    },
    {
        "id": "qa_d09",
        # 干扰: COCO / VOC / YOLO 三种标注格式都列出
        "question": "用 XML 文件存每张图片标注信息的格式是哪一种？",
        "gold_answer": "VOC 格式",
        "gold_filename": "HRSID与SSDD数据集.txt",
        "gold_snippet": "VOC格式：\n   - XML格式存储每张图像的标注信息",
        "qa_type": "fact",
        "_design": "D (干扰: COCO / YOLO / VOC 三种格式相邻)",
    },
    {
        "id": "qa_d10",
        # 干扰: SAR 多种成像模式（条带/聚束/扫描/TOPS）都被提及
        "question": "Sentinel-1 卫星主要采用的 SAR 成像模式是哪种？",
        "gold_answer": "TOPS 模式",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "TOPS模式：Sentinel-1卫星的主要成像模式",
        "qa_type": "fact",
        "_design": "D (干扰: 多种成像模式 chunk 相邻)",
    },
    {
        "id": "qa_d11",
        # 干扰: 多种检测器分类（两阶段/单阶段/Transformer）
        "question": "DETR 这种用注意力机制做检测的属于哪一类检测器？",
        "gold_answer": "Transformer 检测器",
        "gold_filename": "SAR舰船检测概述.txt",
        "gold_snippet": "Transformer检测器：如DETR及其变体",
        "qa_type": "fact",
        "_design": "D (干扰: 三类检测器 chunk 相邻)",
    },

    # ============== 🔗 综合 / 多段推理 (4 题) ==============
    {
        "id": "qa_d12",
        # 综合: 散斑滤波方法分散在 chunk 里
        "question": "Lee 滤波是哪一类方法，基于什么统计？",
        "gold_answer": "基于局部统计的自适应滤波",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "Lee滤波：基于局部统计的自适应滤波",
        "qa_type": "concept",
        "_design": "S (综合: 多种滤波方法 chunk)",
    },
    {
        "id": "qa_d13",
        # 综合: SAR 检测应用场景多个
        "question": "SAR 舰船检测在打击非法捕捞方面发挥什么作用？",
        "gold_answer": "通过监测渔船活动管理渔业资源、识别非法捕捞",
        "gold_filename": "SAR舰船检测概述.txt",
        "gold_snippet": "渔业管理：监测非法捕捞活动",
        "qa_type": "concept",
        "_design": "S (综合: 5 个应用场景里挑一个)",
    },
    {
        "id": "qa_d14",
        # 综合: YOLO11 architecture 改进多个
        "question": "YOLO11 里哪个模块结合了注意力机制？",
        "gold_answer": "C2PSA 模块",
        "gold_filename": "YOLO目标检测模型知识.txt",
        "gold_snippet": "C2PSA模块：结合注意力机制的特征增强模块",
        "qa_type": "fact",
        "_design": "S (综合: 多个 architecture 改进点)",
    },
    {
        "id": "qa_d15",
        # 综合: 极化方式有 4 种（单/双/全），含信息量推理
        "question": "哪种 SAR 极化方式信息最丰富，适合精细分类？",
        "gold_answer": "全极化（HH+HV+VH+VV）",
        "gold_filename": "SAR图像特性与挑战.txt",
        "gold_snippet": "全极化（HH+HV+VH+VV）：信息最丰富，适合精细分类",
        "qa_type": "fact",
        "_design": "S (综合: 三种极化方式对比)",
    },
]


# ============================================================
# 第 3 步：验证 + 写出
# ============================================================

def normalize(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(s.split())
    return s.lower()


def verify(qa_list, label):
    data_dir = ROOT / 'data'
    failures = []
    for q in qa_list:
        src_path = data_dir / q['gold_filename']
        if not src_path.exists():
            failures.append((q['id'], f'文件不存在: {src_path}'))
            continue
        content = src_path.read_text(encoding='utf-8')
        if normalize(q['gold_snippet']) not in normalize(content):
            failures.append((q['id'], f'snippet 不在原文里: "{q["gold_snippet"][:40]}..."'))
    if failures:
        print(f'❌ {label} 验证失败：')
        for qid, reason in failures:
            print(f'  {qid}: {reason}')
    else:
        print(f'✅ {label}：{len(qa_list)} 题 snippet 全部 verbatim 匹配')
    return failures


def main():
    # 加载原 50 题
    with open(ROOT / 'eval/qa_dataset.json', encoding='utf-8') as f:
        original = json.load(f)
    print(f'原始题集：{len(original)} 题')

    # 找出简单题
    easy_ids = load_easy_qa_ids()
    print(f'简单题（在 hybrid_pc_no_rr@k=10 时 rank=1）: {len(easy_ids)} 题')

    # 保留困难题
    hard_kept = [q for q in original if q['id'] not in easy_ids]
    print(f'保留困难题: {len(hard_kept)} 题')
    print(f'   IDs: {[q["id"] for q in hard_kept]}')

    # 新增 15 题
    print(f'\n新增 15 道：')
    by_design = defaultdict(int)
    for q in NEW_HARD:
        by_design[q['_design'][0]] += 1
    for design_type, n in by_design.items():
        label = {'P': '✏️ 改写式', 'D': '🎭 干扰陷阱', 'S': '🔗 综合推理'}[design_type]
        print(f'   {label}: {n} 题')

    # 验证
    print()
    fail1 = verify(hard_kept, '保留的旧困难题')
    fail2 = verify(NEW_HARD, '新增 15 道')
    if fail1 or fail2:
        print('\n⚠️  有题验证失败，未写出。请先修订上面失败的 snippet。')
        return

    # 合并 + 清理 _design 字段
    final = [q for q in hard_kept]
    for q in NEW_HARD:
        clean = {k: v for k, v in q.items() if not k.startswith('_')}
        final.append(clean)

    out_path = ROOT / 'eval/qa_dataset_hard.json'
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding='utf-8')

    # 汇总
    print(f'\n✅ 已写出 {len(final)} 题到 {out_path}')
    print(f'   = {len(hard_kept)} 道旧困难题 + {len(NEW_HARD)} 道新设计难题\n')

    # 按文件分布
    from collections import Counter
    by_file = Counter(q['gold_filename'] for q in final)
    by_type = Counter(q.get('qa_type', '?') for q in final)
    print(f'按源文件:')
    for fn, n in by_file.most_common():
        print(f'   {n}/{len(final)}  {fn}')
    print(f'按 qa_type: {dict(by_type)}')


if __name__ == '__main__':
    main()
