#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
轮胎缺陷检测算法模板 —— 已集成【实验6：胎侧杂物检测算法】

算法集成说明:
1. 在主窗口中，系统会自动调用本模板的 process_image 方法
2. 四个核心方法：load_image、preprocess_image、detect_defects、postprocess_result
3. 返回的结果图像将在右侧面板显示，检测信息将在底部显示

检测算法（实验6）流程:
    灰度化 -> 逐列灰度归一化 -> 分块背景归一化
    -> 黑帽变换（抑制帘布层横向条纹、凸显比背景暗的孤立小斑）
    -> Otsu 最大类间方差阈值分割 -> 形态学开运算去噪
    -> 连通域过滤（面积/长宽比/响应强度） -> 在原图上用红框标记缺陷
"""
import sys
import os
import datetime
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# 确保可以导入同目录下的 debris 模块（算法实现）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import debris


class TyreDefectDetector:
    """轮胎胎侧杂物缺陷检测器（基于黑帽变换 + Otsu 阈值分割）"""

    def __init__(self):
        """初始化检测器"""
        self.model = None
        self.config = self.load_config()
        print("[检测器] 初始化完成（胎侧杂物检测算法）")

    def load_config(self) -> Dict[str, Any]:
        """加载配置参数"""
        config = {
            'image_size': (640, 480),
            'threshold': 0.5,
            'model_path': './models/tyre_detector.pth',
            'classes': ['胎冠裂纹', '胎侧气泡', '帘线稀疏', '接头开', '异物'],
            'preprocessing': {
                'resize': False,
                'normalize': True,
                'denoise': False
            },
            # ==== 胎侧杂物检测算法参数（与 debris.detect_debris 保持一致）====
            'detect': {
                'bg_size': 25,            # 黑帽变换结构元大小
                'edge_margin': 20,        # 边缘抑制宽度（避免边界伪响应）
                'morph_size': 5,          # 形态学开运算窗口
                'flatten_block': 64,      # 分块背景归一化块大小
                'min_area': 150,          # 缺陷最小面积
                'max_area': 8000,         # 缺陷最大面积
                'min_h': 10, 'min_w': 10,  # 缺陷最小高/宽（剔除线状干扰）
                'max_ratio': 3.0,         # 最大长宽比（剔除细长条带）
                'min_mean_response': 95.0  # 最小响应强度（剔除帘线暗纹）
            }
        }
        return config

    # ------------------------------------------------------------------
    # 核心方法 1：加载图像
    # ------------------------------------------------------------------
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """加载图像数据（使用 PIL 读取为 RGB 数组）"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            print(f"[检测器] 加载图像: {image_path}  尺寸: {image.shape}")
            return image
        except Exception as e:
            print(f"[错误] 加载图像失败: {e}")
            return None

    # ------------------------------------------------------------------
    # 核心方法 2：图像预处理（灰度值调整）
    # ------------------------------------------------------------------
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理：灰度化 + 灰度值调整
          1) 灰度化：加权平均法
          2) 逐列灰度归一化：消除胎壁厚度导致的列方向灰度不均
          3) 分块二维背景归一化：进一步消除大尺度明暗渐变
        """
        try:
            gray = debris.to_gray(image)
            normalized = debris.column_gray_normalize(gray)
            fb = self.config['detect']['flatten_block']
            if fb:
                normalized = debris.block_gray_normalize(normalized, block=fb)
            print("[检测器] 完成图像预处理（灰度化 + 逐列/分块灰度归一化）")
            return normalized
        except Exception as e:
            print(f"[错误] 预处理失败: {e}")
            return debris.to_gray(image)

    # ------------------------------------------------------------------
    # 核心方法 3：缺陷检测（核心算法）
    # ------------------------------------------------------------------
    def detect_defects(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行胎侧杂物缺陷检测（输入为预处理后的灰度归一化图）

        流程：黑帽变换 -> 边缘抑制 -> Otsu 分割 -> 形态学开运算 -> 连通域过滤 -> 生成缺陷列表
        返回格式：{'has_defect', 'defects', 'confidence', 'processing_time'}
        """
        t0 = time.time()
        p = self.config['detect']
        try:
            # 1) 黑帽变换：抑制帘线条纹、凸显比背景暗的小斑缺陷
            response = debris.black_hat(image, p['bg_size']).astype(np.float64)

            # 2) 边缘抑制：图像最外圈受黑帽边界效应影响，会产生伪响应，置 0
            m = int(p['edge_margin'])
            if m > 0:
                response[:m, :] = 0
                response[-m:, :] = 0
                response[:, :m] = 0
                response[:, -m:] = 0

            # 3) Otsu 最大类间方差阈值分割
            T = debris.otsu_threshold(response)
            binary = (response > T).astype(np.uint8)

            # 4) 形态学开运算（先腐蚀后膨胀）去噪
            morphed = debris.opening(binary, p['morph_size'])

            # 5) 连通域过滤：面积 / 长宽比 / 响应强度
            cleaned, labeled, regions = debris.remove_small_objects(
                morphed, response=response,
                min_mean_response=p['min_mean_response'],
                min_area=p['min_area'], max_area=p['max_area'],
                min_h=p['min_h'], min_w=p['min_w'], max_ratio=p['max_ratio'])

            # 6) 把检测到的连通域转换为系统要求的缺陷列表格式
            defects = []
            for r in regions:
                x0, y0, x1, y1 = r['bbox']
                # 置信度：把响应强度归一化到 0~1
                conf = float(min(1.0, r['mean_response'] / 160.0))
                defects.append({
                    'type': '异物',                       # 胎侧杂物归为"异物"类
                    'confidence': conf,
                    'location': {'x': x0, 'y': y0,
                                 'width': x1 - x0 + 1, 'height': y1 - y0 + 1},
                    'severity': '严重' if r['area'] >= 600 else '中等',
                })

            print(f"[检测器] 执行缺陷检测算法：检测到 {len(defects)} 处胎侧杂物")

            return {
                'has_defect': len(defects) > 0,
                'defects': defects,
                'confidence': max([d['confidence'] for d in defects], default=1.0),
                'processing_time': time.time() - t0,
            }
        except Exception as e:
            print(f"[错误] 检测失败: {e}")
            return {'has_defect': False, 'defects': [], 'confidence': 0.0, 'error': str(e)}

    # ------------------------------------------------------------------
    # 核心方法 4：结果后处理
    # ------------------------------------------------------------------
    def postprocess_result(self, image: np.ndarray,
                           detection_result: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """结果后处理：在原始图像上用红框标记缺陷，并生成检测信息"""
        try:
            result_image = image.copy()

            # 在图像上标记检测到的缺陷（复用 debris 的矩形框绘制函数）
            if detection_result.get('has_defect'):
                regions = []
                for d in detection_result['defects']:
                    loc = d['location']
                    regions.append({'bbox': (loc['x'], loc['y'],
                                             loc['x'] + loc['width'] - 1,
                                             loc['y'] + loc['height'] - 1)})
                result_image = debris.draw_defects(result_image, regions)

            # 生成检测信息（result: 0=不通过, 1=通过, 2=检测失败）
            detection_info = self.generate_detection_info(detection_result)
            print("[检测器] 完成结果后处理")
            return np.ascontiguousarray(result_image), detection_info

        except Exception as e:
            print(f"[错误] 后处理失败: {e}")
            return image, {'result': 2, 'message': f'后处理失败: {e}', 'details': ''}

    def draw_defects(self, image: np.ndarray, defects: List[Dict[str, Any]]) -> np.ndarray:
        """（兼容保留）在图像上绘制缺陷标记"""
        regions = []
        for d in defects:
            loc = d['location']
            regions.append({'bbox': (loc['x'], loc['y'],
                                     loc['x'] + loc['width'] - 1,
                                     loc['y'] + loc['height'] - 1)})
        return debris.draw_defects(image, regions)

    def generate_detection_info(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成检测信息"""
        if detection_result.get('has_defect'):
            defect_list = []
            for i, defect in enumerate(detection_result['defects']):
                defect_desc = (f"{i+1}. {defect['type']} "
                               f"(置信度: {defect['confidence']:.2%}, 严重程度: {defect['severity']})")
                defect_list.append(defect_desc)
            info = {
                'result': 0,                      # 0=不通过
                'message': '检测不通过',
                'details': '发现胎侧杂物缺陷:\n' + '\n'.join(defect_list),
                'confidence': detection_result['confidence'],
                'defect_count': len(detection_result['defects'])
            }
        else:
            info = {
                'result': 1,                      # 1=通过
                'message': '检测通过',
                'details': '未检测到胎侧杂物缺陷',
                'confidence': detection_result.get('confidence', 1.0),
                'defect_count': 0
            }
        return info

    # ------------------------------------------------------------------
    # 完整流程
    # ------------------------------------------------------------------
    def process_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """完整的图像处理流程：加载 -> 预处理 -> 检测 -> 后处理"""
        try:
            # 1. 加载图像
            image = self.load_image(image_path)
            if image is None:
                return None, {'result': 2, 'message': '图像加载失败', 'details': ''}

            # 2. 预处理
            preprocessed = self.preprocess_image(image)

            # 3. 缺陷检测
            detection_result = self.detect_defects(preprocessed)

            # 4. 结果后处理
            result_image, detection_info = self.postprocess_result(image, detection_result)

            return result_image, detection_info

        except Exception as e:
            print(f"[错误] 处理流程失败: {e}")
            return None, {'result': 2, 'message': f'处理失败: {str(e)}', 'details': ''}


# 工具函数模板
def create_detection_report(detection_info: Dict[str, Any], image_path: str) -> str:
    """创建检测报告"""
    report = f"""
轮胎缺陷检测报告
================
图像文件: {os.path.basename(image_path)}
检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
检测结果: {detection_info['message']}
置信度: {detection_info.get('confidence', 0):.2%}
缺陷数量: {detection_info.get('defect_count', 0)}

详细信息:
{detection_info.get('details', '无')}
"""
    return report


def save_detection_result(result_image: np.ndarray, detection_info: Dict[str, Any],
                          output_dir: str, filename: str) -> bool:
    """保存检测结果"""
    try:
        from PIL import Image
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(result_image.astype(np.uint8)).save(
            os.path.join(output_dir, f"result_{filename}"))
        print(f"[检测器] 保存检测结果到: {output_dir}")
        return True
    except Exception as e:
        print(f"[错误] 保存失败: {e}")
        return False


# 单独测试：python detection_template.py <图像路径>
if __name__ == "__main__":
    detector = TyreDefectDetector()

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result_image, detection_info = detector.process_image(image_path)

        if result_image is not None:
            print("检测完成!")
            print(f"结果: {detection_info['message']}")
            print(f"详情: {detection_info['details']}")
        else:
            print("检测失败!")
    else:
        print("使用方法: python detection_template.py <image_path>")
