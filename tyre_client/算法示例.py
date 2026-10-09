#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮胎缺陷检测算法示例
演示如何快速实现一个简单的检测算法

这个示例展示了：
1. 使用OpenCV进行传统图像处理
2. 基于边缘检测的缺陷识别
3. 结果可视化和输出格式

使用方法：
1. 复制此代码到 src/algorithms/detection_template.py 中对应的方法
2. 或者参考此代码在主窗口中实现您的算法
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Any


def example_load_image(image_path: str) -> np.ndarray:
    """示例：加载图像"""
    # 使用OpenCV加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"无法加载图像: {image_path}")
    
    # 转换为RGB格式（OpenCV默认是BGR）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def example_preprocess_image(image: np.ndarray) -> np.ndarray:
    """示例：图像预处理"""
    # 1. 调整图像尺寸
    resized = cv2.resize(image, (640, 480))
    
    # 2. 转换为灰度图像
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    
    # 3. 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 4. 直方图均衡化增强对比度
    enhanced = cv2.equalizeHist(blurred)
    
    return enhanced


def example_detect_defects(image: np.ndarray) -> Dict[str, Any]:
    """示例：基于边缘检测的缺陷识别"""
    
    # 1. 边缘检测
    edges = cv2.Canny(image, 50, 150)
    
    # 2. 形态学操作，连接断开的边缘
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 3. 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. 分析轮廓，识别可能的缺陷
    defects = []
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 过滤太小的轮廓（可能是噪声）
        if area < 100:
            continue
            
        # 过滤太大的轮廓（可能是轮胎边缘）
        if area > 5000:
            continue
        
        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓的一些特征
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        aspect_ratio = float(w) / h
        
        # 基于特征判断缺陷类型
        defect_type = "未知缺陷"
        confidence = 0.5
        severity = "轻微"
        
        if circularity > 0.7:  # 接近圆形，可能是气泡
            defect_type = "胎侧气泡"
            confidence = min(0.9, circularity + 0.2)
            severity = "中等" if area > 500 else "轻微"
            
        elif aspect_ratio > 3:  # 长条形，可能是裂纹
            defect_type = "胎冠裂纹"
            confidence = min(0.9, aspect_ratio / 5)
            severity = "严重" if w > 100 else "中等"
            
        else:  # 其他形状
            defect_type = "异物"
            confidence = 0.6
            severity = "中等"
        
        # 添加到缺陷列表
        defect = {
            'type': defect_type,
            'confidence': confidence,
            'location': {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
            },
            'severity': severity,
            'area': int(area),
            'circularity': circularity
        }
        defects.append(defect)
    
    # 5. 生成检测结果
    has_defect = len(defects) > 0
    overall_confidence = 0.8 if has_defect else 0.9
    
    result = {
        'has_defect': has_defect,
        'defects': defects,
        'confidence': overall_confidence,
        'processing_time': 0.5,  # 实际应该计算真实处理时间
        'total_defects': len(defects)
    }
    
    return result


def example_postprocess_result(image: np.ndarray, detection_result: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """示例：结果后处理和可视化"""
    
    # 创建结果图像（在原图上绘制检测结果）
    result_image = image.copy()
    
    if detection_result['has_defect']:
        for i, defect in enumerate(detection_result['defects']):
            loc = defect['location']
            
            # 根据缺陷类型选择颜色
            if defect['type'] == '胎侧气泡':
                color = (0, 0, 255)  # 红色
            elif defect['type'] == '胎冠裂纹':
                color = (255, 165, 0)  # 橙色
            else:
                color = (255, 0, 0)  # 红色
            
            # 绘制矩形框
            cv2.rectangle(result_image, 
                         (loc['x'], loc['y']), 
                         (loc['x'] + loc['width'], loc['y'] + loc['height']), 
                         color, 2)
            
            # 添加标签
            label = f"{i+1}:{defect['type']}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 绘制标签背景
            cv2.rectangle(result_image,
                         (loc['x'], loc['y'] - label_size[1] - 10),
                         (loc['x'] + label_size[0], loc['y']),
                         color, -1)
            
            # 绘制标签文字
            cv2.putText(result_image, label, 
                       (loc['x'], loc['y'] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加置信度
            conf_text = f"{defect['confidence']:.2%}"
            cv2.putText(result_image, conf_text,
                       (loc['x'], loc['y'] + loc['height'] + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # 生成系统需要的检测信息格式
    if detection_result['has_defect']:
        # 生成详细信息
        defect_details = []
        for i, defect in enumerate(detection_result['defects']):
            detail = f"{i+1}. {defect['type']} (置信度: {defect['confidence']:.2%}, 严重程度: {defect['severity']})"
            defect_details.append(detail)
        
        detection_info = {
            'result': 0,  # 0=不通过
            'message': '检测不通过',
            'details': '\\n'.join(defect_details),
            'confidence': detection_result['confidence'],
            'defect_count': len(detection_result['defects'])
        }
    else:
        detection_info = {
            'result': 1,  # 1=通过
            'message': '检测通过',
            'details': f"未发现缺陷，置信度: {detection_result['confidence']:.2%}",
            'confidence': detection_result['confidence'],
            'defect_count': 0
        }
    
    return result_image, detection_info


def complete_example_process(image_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """完整的检测流程示例"""
    try:
        print(f"开始处理图像: {image_path}")
        
        # 1. 加载图像
        image = example_load_image(image_path)
        print(f"图像尺寸: {image.shape}")
        
        # 2. 预处理
        preprocessed = example_preprocess_image(image)
        print("完成图像预处理")
        
        # 3. 检测
        detection_result = example_detect_defects(preprocessed)
        print(f"检测完成，发现 {len(detection_result['defects'])} 个潜在缺陷")
        
        # 4. 后处理
        result_image, detection_info = example_postprocess_result(image, detection_result)
        print(f"检测结果: {detection_info['message']}")
        
        return result_image, detection_info
        
    except Exception as e:
        print(f"处理失败: {e}")
        return None, {
            'result': 2,  # 2=检测失败
            'message': '检测失败',
            'details': f'错误信息: {str(e)}',
            'confidence': 0.0
        }


# 测试代码
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # 运行检测
        result_image, detection_info = complete_example_process(image_path)
        
        if result_image is not None:
            # 显示结果
            print("\\n=== 检测结果 ===")
            print(f"状态: {detection_info['message']}")
            print(f"详情: {detection_info['details']}")
            print(f"置信度: {detection_info['confidence']:.2%}")
            
            # 可视化结果（可选）
            try:
                # 加载原图进行对比
                original = example_load_image(image_path)
                
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(original)
                plt.title('原始图像')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(result_image)
                plt.title(f'检测结果 - {detection_info["message"]}')
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("matplotlib未安装，跳过可视化")
        else:
            print("检测失败!")
    else:
        print("使用方法: python 算法示例.py <图像路径>")
        print("\\n这个示例展示了如何实现一个完整的轮胎缺陷检测算法")
        print("您可以参考这个代码来开发自己的检测算法")