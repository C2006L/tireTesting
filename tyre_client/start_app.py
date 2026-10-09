#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轮胎缺陷检测系统 - 简化版启动器
直接运行模式，无需服务，直接关闭
"""

import sys
import os

# 添加模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def start_application():
    """启动应用程序"""
    from PyQt5.QtWidgets import QApplication
    from pages.SimpleMainWindow import SimpleMainWindowPage
    
    app = QApplication(sys.argv)
    
    print("=" * 60)
    print("轮胎缺陷检测系统V3.0 (简化版) - 启动中...")
    print("=" * 60)
    print("系统特性:")
    print("✓ 无需手动开启服务 - 启动即可使用")
    print("✓ 直接点击'开始检测'按钮即可使用")
    print("✓ 支持批量检测文件夹中的图像")
    print("✓ 点击关闭按钮直接退出，无需确认")
    print("✓ 图像智能缩放，自动适应窗口大小")
    print("=" * 60)
    print("使用说明:")
    print("1. 点击左侧工具栏的'开始检测'按钮")
    print("2. 选择包含图像文件的文件夹")
    print("3. 系统将自动循环检测所有图像")
    print("4. 左侧显示原始图像，右侧显示检测结果")
    print("5. 图像自动缩放适应窗口，保证显示完整")
    print("6. 检测完成后可选择新的文件夹继续检测")
    print("7. 直接点击X按钮或菜单退出即可关闭程序")
    print("=" * 60)
    
    # 创建并显示主窗口
    main_window = SimpleMainWindowPage()
    
    print("[系统] 界面启动完成，可以开始使用!")
    print("[提示] 支持的图像格式: jpg, jpeg, png, bmp")
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == '__main__':
    try:
        start_application()
    except Exception as e:
        print(f"[错误] 启动失败: {e}")
        input("按回车键退出...")