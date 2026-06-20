import os
from datetime import datetime

from utils.logger_handler import logger
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.rag_service import RagSummarizeService
from model.factory import chat_model
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
import requests
from tavily import TavilyClient


rag = RagSummarizeService()

_last_viz_path = None

scene_ids = ["S001", "S002", "S003", "S004", "S005", "S006", "S007", "S008", "S009", "S010",]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)


@tool(description="获取指定城市的实时天气信息，返回温度、体感温度、降水、风速等数据。")
def get_weather(city: str):
    """实时天气工具，先地理编码，再查询当前天气。"""
    city = city.strip()
    if not city:
        return "城市不能为空。"

    try:
        # 使用 requests.get ，自动拼接 params
        geocode_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=10
        )
        geocode_response.raise_for_status()  # 自动检查 HTTP 错误状态码
        geocode_data = geocode_response.json()  # 自动完成二进制读取和 JSON 反序列化

        results = geocode_data.get("results") or []
        if not results:
            return f"未查询到城市 {city} 的地理信息，请确认城市名称。"

        location = results[0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_name = location.get("name", city)
        admin1 = location.get("admin1", "")
        country = location.get("country", "")

        # 💡 修改 2：同样使用 requests.get 获取天气数据
        weather_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join(
                    [
                        "temperature_2m",
                        "apparent_temperature",
                        "relative_humidity_2m",
                        "precipitation",
                        "wind_speed_10m",
                        "weather_code",
                    ]
                ),
                "timezone": "auto",
            },
            timeout=10
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        current = weather_data.get("current") or {}
        if not current:
            return f"已定位到 {resolved_name}，但未获取到实时天气数据。"

        weather_code_map = {
            0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴", 45: "雾", 48: "冻雾",
            51: "小毛毛雨", 53: "毛毛雨", 55: "强毛毛雨", 61: "小雨", 63: "中雨",
            65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨",
            81: "较强阵雨", 82: "强阵雨", 95: "雷暴",
        }
        weather_text = weather_code_map.get(current.get("weather_code"), "未知天气")
        location_text = ", ".join(filter(None, [resolved_name, admin1, country]))

        return (
            f"{location_text} 当前天气：{weather_text}；"
            f"温度 {current.get('temperature_2m')}°C，"
            f"体感 {current.get('apparent_temperature')}°C，"
            f"相对湿度 {current.get('relative_humidity_2m')}%，"
            f"降水 {current.get('precipitation')} mm，"
            f"风速 {current.get('wind_speed_10m')} km/h。"
        )

    # 将原本底层的 URLError 替换为 requests 专属的网络异常基类
    except requests.exceptions.RequestException as e:
        logger.warning(f"天气查询网络请求失败: {str(e)}")
        return f"天气服务当前不可用，无法获取 {city} 的实时天气。"
    except Exception as e:
        logger.error(f"天气查询异常: {str(e)}", exc_info=True)
        return f"获取 {city} 天气时发生异常。"


@tool(description="获取用户当前真实的IP地理位置，返回城市、省份、国家等信息，以纯字符串形式返回")
def get_user_location() -> str:
    try:
        resp = requests.get("http://ip-api.com/json/?fields=city,regionName,country,lat,lon", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        city = data.get("city", "")
        region = data.get("regionName", "")
        country = data.get("country", "")
        if city:
            return f"{city}, {region}, {country}"
        return "未能获取到位置信息"
    except requests.exceptions.RequestException as e:
        logger.warning(f"IP定位失败: {str(e)}")
        return "位置服务不可用，无法获取当前城市。"


@tool(description="获取当前SAR检测场景的ID，以纯字符串形式返回，格式为S+三位数字（如S001）")
def get_scene_id() -> str:
    return random.choice(scene_ids)


@tool(description="获取当前日期和月份，返回格式为'YYYY-MM-DD，YYYY-MM'的字符串，包含完整的年-月-日信息。")
def get_current_month():
    now = datetime.now()
    return f"{now.strftime('%Y-%m-%d')}，{now.strftime('%Y-%m')}"



def generate_external_data():
    """
    {
        "scene_id": {
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            ...
        },
        "scene_id": {
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            "month" : {"场景特征": xxx, "检测性能": xxx, ...}
            ...
        },
        ...
    }
    :return:
    """
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部文件{external_data_path}不存在")

        with open(external_data_path,'r',encoding="UTF-8") as f :
            for line in f.readlines()[1:]:
                arr = line.strip().split(",")
                scene_id: str = arr[0].replace('"', "")
                scene_feature: str = arr[1].replace('"', "")
                detection_perf: str = arr[2].replace('"', "")
                model_status: str = arr[3].replace('"', "")
                perf_compare: str = arr[4].replace('"', "")
                time: str = arr[5].replace('"', "")

                if scene_id not in external_data:
                    external_data[scene_id] = {}

                external_data[scene_id][time] = {
                    "场景特征": scene_feature,
                    "检测性能": detection_perf,
                    "模型状态": model_status,
                    "性能对比": perf_compare,
                }

@tool(description="从外部系统中获取指定SAR检测场景在指定月份的检测记录，以纯字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(scene_id: str, month: str) -> str:
    generate_external_data()

    try:
        return external_data[scene_id][month]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到场景：{scene_id}在{month}的检测记录数据")
        return ""


@tool(description="获取指定港口城市的实时海况信息，返回浪高、涌浪高度、海面风速、水温等SAR成像关键参数。")
def get_sea_state(city: str) -> str:
    city = city.strip()
    if not city:
        return "城市不能为空。"

    port_coords = {
        "上海": (31.2, 122.1),
        "青岛": (36.0, 120.5),
        "厦门": (24.4, 118.2),
        "大连": (38.9, 121.7),
        "宁波": (29.9, 122.1),
        "深圳": (22.5, 114.2),
        "天津": (38.9, 117.9),
        "广州": (22.7, 113.7),
    }

    try:
        if city in port_coords:
            latitude, longitude = port_coords[city]
            resolved_name = city
            admin1 = ""
            country = "中国"
        else:
            geocode_response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
                timeout=10
            )
            geocode_response.raise_for_status()
            geocode_data = geocode_response.json()

            results = geocode_data.get("results") or []
            if not results:
                return f"未查询到城市 {city} 的地理信息，请确认城市名称。"

            location = results[0]
            latitude = location["latitude"]
            longitude = location["longitude"]
            resolved_name = location.get("name", city)
            admin1 = location.get("admin1", "")
            country = location.get("country", "")

        marine_response = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": ",".join([
                    "wave_height",
                    "wave_direction",
                    "wave_period",
                    "wind_wave_height",
                    "wind_wave_direction",
                    "wind_wave_period",
                    "swell_wave_height",
                    "swell_wave_direction",
                    "swell_wave_period",
                    "sea_surface_temperature",
                ]),
                "timezone": "auto",
            },
            timeout=10
        )
        marine_response.raise_for_status()
        marine_data = marine_response.json()

        current = marine_data.get("current") or {}
        if not current:
            return f"已定位到 {resolved_name} 近海，但未获取到海况数据。"

        location_text = ", ".join(filter(None, [resolved_name, admin1, country]))

        wave_height = current.get("wave_height")
        wind_wave_height = current.get("wind_wave_height")
        swell_wave_height = current.get("swell_wave_height")
        wave_period = current.get("wave_period")
        wave_direction = current.get("wave_direction")
        sst = current.get("sea_surface_temperature")

        def assess_sar_quality(wave_h):
            if wave_h is None:
                return "未知"
            if wave_h < 0.5:
                return "极佳（低海况，杂波少，适合小目标检测）"
            elif wave_h < 1.5:
                return "良好（中等海况，轻微杂波干扰）"
            elif wave_h < 3.0:
                return "一般（较高海况，杂波明显，小目标检测困难）"
            else:
                return "较差（高海况，强杂波，建议谨慎解读检测结果）"

        quality = assess_sar_quality(wave_height)

        return (
            f"{location_text} 近海当前海况：\n"
            f"有效波高 {wave_height}m（{quality}）；\n"
            f"风浪高度 {wind_wave_height}m，涌浪高度 {swell_wave_height}m；\n"
            f"波浪周期 {wave_period}s，波浪方向 {wave_direction}°；\n"
            f"海表温度 {sst}°C。"
        )

    except requests.exceptions.RequestException as e:
        logger.warning(f"海况查询网络请求失败: {str(e)}")
        return f"海况服务当前不可用，无法获取 {city} 的海况数据。"
    except Exception as e:
        logger.error(f"海况查询异常: {str(e)}", exc_info=True)
        return f"获取 {city} 海况时发生异常。"


@tool(description="对比多个SAR检测场景在同一月份的检测性能，入参scene_ids为逗号分隔的场景ID列表（如'S001,S003,S005'），month为月份（如'2025-03'）")
def compare_scenes(scene_ids: str, month: str) -> str:
    generate_external_data()

    ids = [s.strip() for s in scene_ids.split(",") if s.strip()]
    if not ids:
        return "场景ID列表不能为空。"

    result_parts = []
    for sid in ids:
        if sid not in external_data:
            result_parts.append(f"【{sid}】无记录")
            continue
        if month not in external_data[sid]:
            result_parts.append(f"【{sid}】{month} 无记录")
            continue

        data = external_data[sid][month]
        result_parts.append(
            f"【{sid}】\n"
            f"  场景特征：{data.get('场景特征', 'N/A')}\n"
            f"  检测性能：{data.get('检测性能', 'N/A')}\n"
            f"  模型状态：{data.get('模型状态', 'N/A')}\n"
            f"  性能对比：{data.get('性能对比', 'N/A')}"
        )

    if not result_parts:
        return f"所有场景在 {month} 均无记录。"

    return f"场景对比（{month}）：\n\n" + "\n\n".join(result_parts)


@tool(description="获取指定SAR检测场景在多个连续月份的历史检测趋势，入参scene_id为场景ID，months为逗号分隔的月份列表（如'2025-01,2025-02,2025-03'），不传则默认最近6个月")
def get_scene_trend(scene_id: str, months: str = "") -> str:
    generate_external_data()

    if scene_id not in external_data:
        return f"场景 {scene_id} 无任何检测记录。"

    if months:
        month_list = [m.strip() for m in months.split(",") if m.strip()]
    else:
        available_months = sorted(external_data[scene_id].keys(), reverse=True)
        month_list = available_months[:6][::-1]

    if not month_list:
        return f"场景 {scene_id} 无可用月份数据。"

    records = []
    for m in month_list:
        if m in external_data[scene_id]:
            records.append((m, external_data[scene_id][m]))

    if not records:
        return f"场景 {scene_id} 在指定月份无记录。"

    result_parts = [f"场景 {scene_id} 历史检测趋势：\n"]
    for m, data in records:
        perf = data.get("检测性能", "")
        result_parts.append(f"【{m}】{perf}")

    if len(records) >= 2:
        first_perf = records[0][1].get("检测性能", "")
        last_perf = records[-1][1].get("检测性能", "")
        result_parts.append(f"\n覆盖范围：{records[0][0]} 至 {records[-1][0]}，共 {len(records)} 个月数据。")

    return "\n".join(result_parts)

@tool(description="进入报告生成模式：调用后中间件将系统提示切换为 report_prompt，后续工具调用按报告流程执行。仅在用户意图为'报告生成'时调用。")
def fill_context_for_report():
    return "已进入报告生成模式"


@tool(description="联网搜索最新信息。query为搜索关键词，days为可选的时间范围（仅在最近N天内搜索，默认为0表示不限时间）。当用户明确要求'今天''最近X天'的新闻时，必须传入days参数。")
def web_search(query: str, days: int = 0) -> str:
    try:
        tavily_client = TavilyClient(api_key=agent_conf["tavily_api_key"])
        search_kwargs = {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": True,
        }
        if days > 0:
            search_kwargs["days"] = days
        response = tavily_client.search(**search_kwargs)

        if not response.get("results"):
            return f"未找到与「{query}」相关的搜索结果。"

        context = ""
        for i, result in enumerate(response["results"][:5], 1):
            title = result.get("title", "无标题")
            content = result.get("content", "无内容")
            content = " ".join(content.split())
            context += f"[搜索结果{i}] (来源:{title}): {content}\n"

        search_prompt = PromptTemplate.from_template(
            "你是专注于「基于搜索结果总结」的AI助手，需结合用户提问和联网搜索结果，生成简洁准确的概括回答。\n\n"
            "### 输入信息\n"
            "1. 用户提问：{input}\n"
            "2. 搜索结果(在下一个###之前内容均为搜索结果)：{context}\n\n"
            "### 严格遵守以下约束\n"
            "1. 回答必须完全基于搜索结果中的信息，不编造、不添加未提及的内容；\n"
            "2. 仅用中文回答，语气客观、简洁，不冗余；\n"
            "3. 严格围绕用户原始提问总结，不扩充问题范围；\n"
            "4. 若搜索结果与用户问题无关或信息不足，诚实说明；\n"
            "5. 仅输出概括内容本身，以纯文本字符串形式呈现。"
        )
        chain = search_prompt | chat_model | StrOutputParser()
        return chain.invoke({"input": query, "context": context})

    except Exception as e:
        logger.error(f"联网搜索异常: {str(e)}", exc_info=True)
        return f"联网搜索失败：{str(e)}"


@tool(description="对指定路径的SAR图像执行舰船目标检测，返回检测到的舰船数量、各目标置信度、平均置信度等结构化文本结果。适用于用户上传了SAR图像并要求进行检测分析的场景。")
def detect_ships(image_path: str) -> str:
    try:
        from ultralytics import YOLO
        from PIL import Image

        if not os.path.exists(image_path):
            return f"图像文件不存在：{image_path}，请确认路径是否正确。"

        model = YOLO('Detct_prdc/MBE-Net/weights/best.pt')
        image = Image.open(image_path)
        results = model.predict(source=image, imgsz=640, verbose=False)

        boxes = results[0].boxes
        ship_count = len(boxes)

        if ship_count == 0:
            return (
                "🔍 SAR舰船检测结果：\n"
                f"- 图像：{os.path.basename(image_path)}\n"
                f"- 图像尺寸：{image.size[0]}×{image.size[1]}\n"
                f"- 检测目标数：0\n"
                f"- 检测状态：未检测到舰船目标\n"
                "⚠️ 可能原因：海况干扰、目标过小、图像质量低。建议结合海况数据进一步分析。"
            )

        confidences = [float(box.conf[0]) for box in boxes]
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        low_conf_count = sum(1 for c in confidences if c < 0.6)

        detail_lines = []
        for i, conf in enumerate(confidences, 1):
            x1, y1, x2, y2 = boxes[i - 1].xyxy[0].tolist()
            detail_lines.append(
                f"  目标{i}：置信度 {conf:.2f}，位置 ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
            )

        quality = "高" if avg_conf >= 0.75 else ("中等" if avg_conf >= 0.6 else "偏低")
        warning = ""
        if low_conf_count > 0:
            warning = f"\n⚠️ 其中有 {low_conf_count} 个目标置信度低于0.6，可能存在误检或漏检，建议结合海况数据和专业知识进一步分析。"

        from PIL import ImageDraw
        import tempfile

        draw_img = image.copy().convert("RGB")
        draw = ImageDraw.Draw(draw_img)
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            color = "#00ff88" if conf >= 0.6 else "#ff8800"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.rectangle([x1, y1 - 14, x1 + 55, y1], fill=color)
            draw.text((x1 + 3, y1 - 13), f"{conf:.2f}", fill="#000000")

        output_dir = tempfile.mkdtemp()
        output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
        draw_img.save(output_path)

        global _last_viz_path
        _last_viz_path = output_path

        return (
            f"🔍 SAR舰船检测结果：\n"
            f"- 图像：{os.path.basename(image_path)}\n"
            f"- 图像尺寸：{image.size[0]}×{image.size[1]}\n"
            f"- 检测目标数：{ship_count}\n"
            f"- 平均置信度：{avg_conf:.2f}（检测质量：{quality}）\n"
            f"- 置信度范围：{min_conf:.2f} ~ {max_conf:.2f}\n"
            f"- 各目标详情：\n"
            + "\n".join(detail_lines)
            + warning
        )

    except Exception as e:
        logger.error(f"舰船检测异常: {str(e)}", exc_info=True)
        return f"舰船检测失败：{str(e)}"


@tool(description="提取指定文件中的文本内容。支持PDF(.pdf)、Word(.docx)、图片(.png/.jpg/.jpeg)的OCR文字识别、以及纯文本(.txt/.md)。传入文件路径，返回提取到的文字内容。")
def extract_file_content(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"文件不存在：{file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    try:
        if ext in (".txt", ".md", ".csv", ".json", ".log", ".py", ".yaml", ".yml"):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 3000:
                content = content[:3000] + f"\n\n... (文件总长 {len(content)} 字符，已截断前3000字符)"
            return f"📄 文件「{filename}」文本内容：\n\n{content}"

        elif ext == ".pdf":
            import fitz
            doc = fitz.open(file_path)
            pages = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    pages.append(text.strip())
            doc.close()
            content = "\n\n".join(pages)
            if not content:
                return f"📄 文件「{filename}」为扫描版PDF，未提取到文字内容（可能需要OCR）。"
            if len(content) > 3000:
                content = content[:3000] + f"\n\n... (PDF共{len(pages)}页，已截断前3000字符)"
            return f"📄 文件「{filename}」(PDF, {len(pages)}页) 文本内容：\n\n{content}"

        elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR()
            result, _ = engine(file_path)
            if not result:
                return f"📄 图片「{filename}」未识别到文字内容。"
            lines = []
            for box, text, conf in result:
                if conf is None:
                    conf = 1.0
                lines.append(text)
            content = "\n".join(lines)
            if len(content) > 3000:
                content = content[:3000] + f"\n\n... (OCR共识别{len(lines)}段文字，已截断前3000字符)"
            return f"📄 图片「{filename}」OCR识别结果（共{len(lines)}段文字）：\n\n{content}"

        elif ext == ".docx":
            from docx import Document
            doc = Document(file_path)
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n".join(paras)
            if not content:
                return f"📄 文件「{filename}」未提取到文字内容。"
            if len(content) > 3000:
                content = content[:3000] + f"\n\n... (Word共{len(paras)}段，已截断前3000字符)"
            return f"📄 文件「{filename}」(Word, {len(paras)}段) 文本内容：\n\n{content}"

        else:
            return f"不支持的文件格式「{ext}」，目前支持：txt, md, pdf, docx, png, jpg, jpeg"

    except Exception as e:
        logger.error(f"文件提取异常: {str(e)}", exc_info=True)
        return f"文件提取失败：{str(e)}"
