import json
import os
import sys
from pathlib import Path

# 从AudioDataset.py导入核心类和表情列表
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AudioDataset import UE_CurvesManager
from PreProcess import ctrl_expressions_list

# ===================== 配置参数 =====================
FPS = 25  # 固定帧率（与pred_test_padding.json一致）
GT_JSON_DIR = Path("./test/json/")  # GT文件所在目录（存放UE输出的json）
OUTPUT_DIR = Path("./test/json/converted/")  # 转换后文件输出目录
# 可选：指定需要转换的GT文件后缀/前缀，默认处理所有json
FILTER_SUFFIX = ".json"

# ===================== 核心转换函数 =====================
def convert_single_gt(gt_json_path: Path, fps: int = 25) -> dict:
    """
    将单个UE输出的GT JSON转换为pred_test_padding.json格式
    :param gt_json_path: UE GT文件路径
    :param fps: 目标帧率
    :return: 符合pred格式的字典
    """
    # 1. 初始化UE曲线管理器（加载GT数据）
    curve_manager = UE_CurvesManager(json_path=str(gt_json_path))
    
    # 2. 关键参数：视频总时长（从GT数据中读取）
    all_time_points = []
    for exp_key in ctrl_expressions_list:
        if exp_key in curve_manager.data:
            all_time_points.extend(curve_manager.data[exp_key]["time"])
    total_seconds = max(all_time_points) if all_time_points else 0.0
    total_frames = int(total_seconds * fps) + 1  # 向上取整
    
    # 3. 调用sample函数生成插值后的表情帧（time_point=0表示从视频开头开始采样）
    # sample函数返回：list[list[float]] → 外层是帧，内层是每个表情key的插值结果
    face_pred = curve_manager.sample(time_point=0.0, seconds=total_seconds, fps=fps)
    
    # 4. 构造motion_pred（与pred_test_padding.json一致，填充0）
    # motion_pred格式：N帧 × 55*3维，全0
    motion_pred = [[0.0 for _ in range(55*3)] for _ in range(total_frames)]
    
    # 5. 组装最终格式（与pred_test_padding.json完全对齐）
    converted_data = {
        "motion_pred": motion_pred,
        "face_pred": face_pred,
        "fps": fps,
        "frames": total_frames,
        "source_gt_path": str(gt_json_path.name)  # 保留源文件信息（可选）
    }
    
    return converted_data

def batch_convert_gt(gt_dir: Path, output_dir: Path, fps: int = 25):
    """
    批量转换目录下所有UE GT JSON文件
    :param gt_dir: GT文件目录
    :param output_dir: 转换后文件输出目录
    :param fps: 目标帧率
    """
    # 创建输出目录
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 遍历所有GT JSON文件
    gt_files = [f for f in gt_dir.iterdir() if f.name.endswith(FILTER_SUFFIX) and f.is_file()]
    if not gt_files:
        print(f"⚠️ 在目录 {gt_dir} 下未找到符合条件的GT文件（后缀{FILTER_SUFFIX}）")
        return
    
    # 批量转换
    for gt_file in gt_files:
        try:
            print(f"🔄 处理文件：{gt_file.name}")
            # 转换单个文件
            converted_data = convert_single_gt(gt_file, fps)
            # 构造输出文件名（保留原名称，添加_converted后缀）
            output_file_name = gt_file.stem + "_converted.json"
            output_file_path = output_dir / output_file_name
            # 保存转换后的数据（格式化输出，便于查看）
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=4)
            print(f"✅ 转换完成：{output_file_path}")
        except Exception as e:
            print(f"❌ 处理文件 {gt_file.name} 失败：{str(e)}")

# ===================== 执行批量转换 =====================
if __name__ == "__main__":
    print(f"开始批量转换GT文件，源目录：{GT_JSON_DIR}，输出目录：{OUTPUT_DIR}")
    batch_convert_gt(GT_JSON_DIR, OUTPUT_DIR, FPS)
    print("批量转换完成！")
