import sys, os
sys.path.append(os.path.join(__file__, '../../'))
sys.path.append(os.path.join(__file__, '../../../algorithms'))
import json
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget, QLabel, QFileDialog, QMessageBox
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
import glob
from layouts.FlexiableLayout import FlexiableLayoutWindow
from widgets.MainTitleBar import MainTitleBarWidget
from widgets.MainWindowPanel import MainWindowPanelWidget
from widgets.MainMenu import MainMenuWidget
from pages.ConfirmWindow import ConfirmWindowPage
from pages.AlertWindow import AlertWindowPage
from common.commlib import loadQss

# 导入检测算法模板（可选）
try:
    from algorithms.detection_template import TyreDefectDetector
    DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"检测算法模块未找到: {e}")
    DETECTION_AVAILABLE = False

staticPath = os.path.join(__file__, '../../../') + '/static'
configPath = os.path.join(__file__, '../../../') + '/config'


class BatchDetectionThread(QThread):
    """批量检测线程"""
    # 信号定义
    imageProcessed = pyqtSignal(str, str, dict)  # 图片路径，结果图片路径，检测信息
    batchCompleted = pyqtSignal(int, int)  # 总数，成功数
    progressUpdated = pyqtSignal(int, int, str)  # 当前索引，总数，当前文件名
    errorOccurred = pyqtSignal(str)  # 错误信息
    
    def __init__(self, image_folder, detector=None):
        super().__init__()
        self.image_folder = image_folder
        self.detector = detector
        self.is_running = False
        self.should_stop = False
        
        # 支持的图像格式
        self.supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.PNG', '*.JPG', '*.JPEG', '*.BMP', '*.TIFF']
    
    def run(self):
        """执行批量检测"""
        try:
            self.is_running = True
            self.should_stop = False
            
            # 获取文件夹中的所有图像文件
            image_files = []
            for format_pattern in self.supported_formats:
                image_files.extend(glob.glob(os.path.join(self.image_folder, format_pattern)))
            

            
            total_count = len(image_files)
            success_count = 0
            
            for i, image_path in enumerate(image_files):
                if self.should_stop:
                    break
                    
                # 更新进度
                filename = os.path.basename(image_path)
                self.progressUpdated.emit(i + 1, total_count, filename)
                
                try:
                    # 执行检测
                    if self.detector and DETECTION_AVAILABLE:
                        result_image, detection_info = self.detector.process_image(image_path)
                    else:
                        result_image, detection_info = self.simulate_detection(image_path)
                    
                    # 发送结果
                    self.imageProcessed.emit(image_path, "", detection_info)
                    success_count += 1
                    
                    # 添加延时以便观察结果
                    self.msleep(1000)  # 暂停1秒
                    
                except Exception as e:
                    error_info = {
                        'result': 2,
                        'message': f'处理失败: {str(e)}',
                        'details': '',
                        'confidence': 0.0
                    }
                    self.imageProcessed.emit(image_path, "", error_info)
            
            # 批量处理完成
            self.batchCompleted.emit(total_count, success_count)
            
        except Exception as e:
            self.errorOccurred.emit(f"批量检测出错: {str(e)}")
        finally:
            self.is_running = False
    
    def simulate_detection(self, image_path):
        """模拟检测过程 - 结果图像与原图保持一致"""
        import random
        
        # ========================================================================
        # 算法集成区域 - 请在此处添加您的检测算法
        # ========================================================================
        
        # 步骤1: 加载原始图像数据
        # TODO: 在此处添加图像加载逻辑
        # 示例:
        # import cv2
        # original_image = cv2.imread(image_path)
        # if original_image is None:
        #     raise Exception(f"无法加载图像: {image_path}")
        
        # 步骤2: 图像预处理
        # TODO: 在此处添加预处理逻辑
        # 示例:
        # gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        # denoised_image = cv2.bilateralFilter(gray_image, 9, 75, 75)
        # enhanced_image = cv2.equalizeHist(denoised_image)
        
        # 步骤3: 缺陷检测算法
        # TODO: 在此处添加您的检测算法
        # 示例:
        # # 传统图像处理方法
        # edges = cv2.Canny(enhanced_image, 50, 150)
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # defects = self.analyze_contours(contours)
        # 
        # # 或者使用深度学习模型
        # model_input = self.preprocess_for_model(enhanced_image)
        # predictions = self.detection_model.predict(model_input)
        # defects = self.parse_predictions(predictions)
        
        # 步骤4: 结果后处理
        # TODO: 在此处添加结果处理逻辑
        # 示例:
        # if defects:
        #     # 在原图上标记缺陷位置
        #     result_image = original_image.copy()
        #     for defect in defects:
        #         x, y, w, h = defect['bbox']
        #         cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        #         cv2.putText(result_image, defect['type'], (x, y-10), 
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # else:
        #     result_image = original_image.copy()
        
        # ========================================================================
        # 算法集成区域结束
        # ========================================================================
        
        # 现在使用模拟结果（后续替换为真实算法）
        defect_types = ['胎冠裂纹', '胎侧气泡', '帘线稀疏', '接头开', '异物']
        has_defect = random.choice([True, False])
        
        if has_defect:
            defect_type = random.choice(defect_types)
            detection_info = {
                'result': 0,  # 0=不通过, 1=通过, 2=检测失败
                'message': '检测不通过',
                'details': f'发现缺陷: {defect_type}',
                'confidence': random.uniform(0.7, 0.95)
            }
        else:
            detection_info = {
                'result': 1,
                'message': '检测通过',
                'details': '无缺陷',
                'confidence': random.uniform(0.85, 0.98)
            }
        
        # 返回空的结果图像路径，由主线程生成与原图一致的结果图像
        return None, detection_info
    
    def stop(self):
        """停止检测"""
        self.should_stop = True


class SimpleMainWindowPage(FlexiableLayoutWindow):
    # 本地状态管理
    serviceFlag = False
    detectionFlag = False
    startTimestamp = None
    imgCheckCounter = 1
    
    # 批量检测相关
    batch_thread = None
    current_batch_folder = None
    total_images = 0
    processed_images = 0
    success_count = 0

    def __init__(self):
        # 主窗口尺寸
        self.windowWidth = 1520
        self.windowHeight = 900

        # 阴影设置
        self.shadowX = 0
        self.shadowY = 0
        self.shadowWidth = 60
        self.shadowColor = QColor(132, 132, 132)

        # 样式表
        self.qss = staticPath + '/qss/main_window.qss'

        # 窗口标题
        self.windowTitle = '轮胎缺陷监测系统V3.0 (简化版)'

        super(SimpleMainWindowPage, self).__init__(False, self.windowWidth, 
            self.windowHeight, self.shadowX, self.shadowY, self.shadowWidth, self.shadowColor)

        # 初始化检测器
        self.detector = None
        if DETECTION_AVAILABLE:
            try:
                self.detector = TyreDefectDetector()
                print("[系统] 检测算法模块已加载")
            except Exception as e:
                print(f"[警告] 检测器初始化失败: {e}")
                self.detector = None

        self.renderWindow()
        self.enableFunctions()
        self.startTimestamp = datetime.now()
        
        # 设置系统为直接运行模式（无需手动开启服务）
        self.serviceFlag = True  # 模拟服务已开启状态
        self.updateCheckInfoPanel(True, None, None, "系统就绪，可直接开始检测")
        
        # 初始化时清空图像显示区域
        QTimer.singleShot(100, self.initializeImagePanels)
        
        print("[系统] 初始化完成，系统处于直接运行模式，无需手动开启服务")

    def renderWindow(self):
        # 加载样式表
        self.setStyleSheet(loadQss(self.qss))

        # 布局
        self.mainGridLayout = QGridLayout()
        self.mainGridLayout.setContentsMargins(self.shadowWidth, self.shadowWidth, 
            self.shadowWidth, self.shadowWidth)
        self.mainGridLayout.setSpacing(0)

        self.setWindowTitle(self.windowTitle)
        
        # 主窗口标题栏
        self.menuBar = MainTitleBarWidget(self.windowTitle)
        self.menuBar.windowMove.connect(self.move)
        self.menuBar.windowFullScreen.connect(self.changeWindowSize)
        self.menuBar.installEventFilter(self)
        self.menuBar.showSubWindow.connect(self.showExitWindow)
        self.menuBar.showMainMenu.connect(self.showMenu)
        
        # 主窗口面板
        self.mainPanel = MainWindowPanelWidget()
        self.mainPanel.setObjectName('mainWindowPanel')
        self.mainPanel.installEventFilter(self)
        self.mainPanel.showSubWindow.connect(self.showCameraFunction)
        self.mainPanel.showNetworkWindow.connect(self.showNetworkFunction)
        self.mainPanel.showParaWindow.connect(self.showParaFunction)
        self.mainPanel.showTableWindow.connect(self.showTableFunction)
        self.mainPanel.showAlertWindow.connect(self.showAlertWindow)
        self.mainPanel.toggleService.connect(self.toggleService)
        self.mainPanel.toggleDetection.connect(self.toggleDetection)
        
        self.mainGridLayout.addWidget(self.menuBar)
        self.mainGridLayout.addWidget(self.mainPanel)
        self.mainWidget.setLayout(self.mainGridLayout)

        # 主菜单面板
        self.mainMenu = MainMenuWidget()
        self.mainMenu.setParent(self)
        self.mainMenu.hide()
        self.mainMenu.togglePanelSignal.connect(self.togglePanel)
        self.mainMenu.showExitWindowSignal.connect(self.showExitWindow)

        # 确认窗口
        self.confirmWindow = ConfirmWindowPage()
        self.confirmWindow.setWindowModality(Qt.ApplicationModal)
        self.confirmWindow.confirmSignal.connect(self.exitSystem)

        # 警告窗口
        self.alertWindow = AlertWindowPage()
        self.alertWindow.setWindowModality(Qt.ApplicationModal)

        self.show()

    def enableFunctions(self):
        """启用所有功能按钮 - 无需服务检查"""
        print("[系统] 启用所有功能按钮，无需服务检查")
        self.mainPanel.toolPanelWidget.setAllBtnsStatus(True)
        # 直接设置服务为已连接状态（纪必要显示）
        self.updateInfoPanel('serverStatusLabel', True)

    def setBtnEnable(self, name):
        """设置按钮可用"""
        if name == 'all':
            self.mainPanel.toolPanelWidget.setAllBtnsStatus(True)
        else:
            self.mainPanel.toolPanelWidget.setBtnEnable(name)

    def updateInfoPanel(self, item, status):
        """更新状态信息面板"""
        statusArr = ['已断开', '已连接', '连接中...']
        label = self.findChild(QLabel, item)
        if label:
            label.setText(statusArr[int(status)])
            if status == True:
                label.setProperty('statusLabel', 'connected')
            else:
                label.setProperty('statusLabel', 'disconnected')
            label.style().unpolish(label)
            label.style().polish(label)

    def startService(self):
        """开启服务 - 本地模拟"""
        self.serviceFlag = True
        # 更新检测信息面板
        self.updateCheckInfoPanel(True, None, None, None)
        # 模拟连接状态
        self.updateInfoPanel('serverStatusLabel', True)
        self.showAlertWindow('服务已开启 (本地模拟)')

    def stopService(self):
        """停止服务"""
        if self.serviceFlag:
            self.serviceFlag = False
            # 停止检测
            if self.detectionFlag:
                self.stopDetection()
                self.mainPanel.toolPanelWidget.stopDetectionRender()
            
            # 更新检测信息面板
            self.updateCheckInfoPanel(None, None, None, None)
            # 清空图像面板
            imgPanel = self.mainPanel.rightPanelWidget.imgPanel
            imgPanel.clearImgPanel('left')
            imgPanel.clearImgPanel('right')
            # 更新连接状态
            self.updateInfoPanel('serverStatusLabel', False)

    def toggleService(self, flag):
        """开启/停止服务"""
        if self.detectionFlag:
            return
        if flag:
            self.startService()
        else:
            self.stopService()

    def startDetection(self):
        """开始检测 - 直接运行，无需服务"""
        # 直接弹出文件选择对话框，无需检查服务状态
        print("[系统] 开始检测，直接运行模式")
        self.selectImageForDetection()

    def stopDetection(self):
        """停止检测"""
        if self.detectionFlag:
            self.detectionFlag = False
            
            # 停止批量检测线程
            if self.batch_thread and self.batch_thread.isRunning():
                self.batch_thread.stop()
                self.batch_thread.wait(3000)  # 等待最多3秒
                if self.batch_thread.isRunning():
                    self.batch_thread.terminate()
                self.batch_thread = None
            
            self.showAlertWindow("检测已停止")

    def toggleDetection(self, flag):
        """开始检测/暂停检测 - 完全移除服务依赖"""
        print(f"[系统] 切换检测状态: {flag}，无需服务检查")
        if flag:
            self.startDetection()
        else:
            self.stopDetection()

    def selectImageForDetection(self):
        """选择文件夹进行批量检测"""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择包含轮胎图像的文件夹",
            "",
            options=options
        )
        
        if folder_path:
            # 设置检测状态
            self.detectionFlag = True
            self.current_batch_folder = folder_path
            self.imgCheckCounter = 1
            self.processed_images = 0
            self.success_count = 0
            
            # 开始批量检测
            self.startBatchDetection(folder_path)
        else:
            # 用户取消选择，恢复按钮状态
            self.mainPanel.toolPanelWidget.stopDetectionRender()

    def startBatchDetection(self, folder_path):
        """开始批量检测"""
        try:
            # 创建批量检测线程
            self.batch_thread = BatchDetectionThread(folder_path, self.detector)
            
            # 连接信号
            self.batch_thread.imageProcessed.connect(self.onImageProcessed)
            self.batch_thread.batchCompleted.connect(self.onBatchCompleted)
            self.batch_thread.progressUpdated.connect(self.onProgressUpdated)
            self.batch_thread.errorOccurred.connect(self.onBatchError)
            
            # 更新界面状态(完全无需服务状态检查)
            self.updateCheckInfoPanel(True, None, None, "开始检测...")
            
            # 启动线程
            self.batch_thread.start()
            
            folder_name = os.path.basename(folder_path)
            self.showAlertWindow(f'开始批量检测文件夹: {folder_name}')
            print(f"[系统] 已启动批量检测: {folder_name}")
            
        except Exception as e:
            print(f"[错误] 启动批量检测失败: {e}")
            self.showAlertWindow(f'启动批量检测失败: {str(e)}')
            self.mainPanel.toolPanelWidget.stopDetectionRender()
    
    def onImageProcessed(self, image_path, result_path, detection_info):
        """处理单张图像完成的回调"""
        try:
            # 显示当前处理的图像
            self.displayCurrentImage(image_path)
            
            # 显示结果图像 - 使用原图作为结果图像
            if result_path and os.path.exists(result_path):
                self.displayResultImage(result_path)
            else:
                # 使用原图作为结果图像（与原图保持一致）
                self.displayResultImageSameAsOriginal(image_path)
            
            # 更新检测信息
            self.updateCheckInfoPanel(
                False, 
                True, 
                detection_info['result'], 
                detection_info['message']
            )
            
            # 更新成功计数
            if detection_info['result'] != 2:  # 不是检测失败
                self.success_count += 1
            
            self.processed_images += 1
            
        except Exception as e:
            print(f"处理图像回调失败: {e}")
    
    def onProgressUpdated(self, current, total, filename):
        """进度更新回调"""
        self.total_images = total
        progress_info = f"正在处理: {filename} ({current}/{total})"
        
        # 更新检测信息面板
        self.updateCheckInfoPanel(False, False, None, progress_info)
    
    def onBatchCompleted(self, total_count, success_count):
        """批量检测完成回调"""
        self.detectionFlag = False
        self.mainPanel.toolPanelWidget.stopDetectionRender()
        
        completion_msg = f"批量检测完成！\n" \
                        f"总计: {total_count} 张图像\n" \
                        f"成功: {success_count} 张\n" \
                        f"失败: {total_count - success_count} 张"
        
        self.showAlertWindow(completion_msg)
        
        # 更新最终状态
        final_result = 1 if success_count == total_count else 0
        self.updateCheckInfoPanel(
            False, 
            True, 
            final_result, 
            f"批量检测完成 ({success_count}/{total_count})"
        )
    
    def onBatchError(self, error_message):
        """批量检测错误回调"""
        self.detectionFlag = False
        self.mainPanel.toolPanelWidget.stopDetectionRender()
        self.showAlertWindow(f"批量检测出错:\n{error_message}")
    
    def displayCurrentImage(self, image_path):
        """显示当前处理的图像"""
        try:
            # 直接使用QPixmap加载图像
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 直接设置Pixmap到QLabel
                imgPanel = self.mainPanel.rightPanelWidget.imgPanel
                imgPanel.setLeftImage(pixmap)
                
                print(f"已显示原始图像: {os.path.basename(image_path)}, 尺寸: {pixmap.width()}x{pixmap.height()}")
            else:
                print(f"无法加载图像: {image_path}")
            
        except Exception as e:
            print(f"显示当前图像失败: {e}")
    
    def displayResultImage(self, result_path):
        """显示结果图像"""
        try:
            pixmap = QPixmap(result_path)
            if not pixmap.isNull():
                imgPanel = self.mainPanel.rightPanelWidget.imgPanel
                imgPanel.setRightImage(pixmap)
                
                print(f"已显示结果图像: {os.path.basename(result_path)}")
                
        except Exception as e:
            print(f"显示结果图像失败: {e}")
    
    def displayResultImageSameAsOriginal(self, image_path):
        """显示与原图保持一致的结果图像"""
        try:
            # ========================================================================
            # 算法结果显示区域 - 请在此处添加缺陷标记逻辑
            # ========================================================================
            
            # 直接使用原图作为结果图像
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # TODO: 在此处添加缺陷标记逻辑
                # 示例:
                # if detection_result.has_defects:
                #     # 在图像上绘制缺陷框和标签
                #     painter = QPainter(pixmap)
                #     painter.setPen(QPen(Qt.red, 3))
                #     for defect in detection_result.defects:
                #         x, y, w, h = defect.bbox
                #         painter.drawRect(x, y, w, h)
                #         painter.drawText(x, y-10, defect.type)
                #     painter.end()
                
                # 直接显示原图像（保证尺寸与左侧一致）
                imgPanel = self.mainPanel.rightPanelWidget.imgPanel
                imgPanel.setRightImage(pixmap)
                
                print(f"已显示结果图像（与原图一致）: {os.path.basename(image_path)}, 尺寸: {pixmap.width()}x{pixmap.height()}")
            else:
                print(f"无法加载结果图像: {image_path}")
                
            # ========================================================================
            # 算法结果显示区域结束
            # ========================================================================
                
        except Exception as e:
            print(f"显示结果图像（与原图一致）失败: {e}")
    
    def initializeImagePanels(self):
        """初始化图像面板"""
        try:
            imgPanel = self.mainPanel.rightPanelWidget.imgPanel
            imgPanel.clearImgPanel('left')
            imgPanel.clearImgPanel('right')
            print("已初始化图像面板")
        except Exception as e:
            print(f"初始化图像面板失败: {e}")
    
    def generateResultImage(self, detection_info):
        """生成模拟结果图像"""
        try:
            # 创建结果图像（根据检测结果设置不同颜色）
            result_image = QImage(640, 480, QImage.Format_RGB888)
            
            if detection_info['result'] == 0:  # 检测不通过
                result_image.fill(QColor(255, 200, 200))  # 红色背景
            elif detection_info['result'] == 1:  # 检测通过
                result_image.fill(QColor(200, 255, 200))  # 绿色背景
            else:  # 检测失败
                result_image.fill(QColor(200, 200, 200))  # 灰色背景
            
            # 转换为Pixmap并显示
            result_pixmap = QPixmap.fromImage(result_image)
            
            # 显示结果图像
            imgPanel = self.mainPanel.rightPanelWidget.imgPanel
            imgPanel.setRightImage(result_pixmap)
            
            status_text = "检测通过" if detection_info['result'] == 1 else ("检测不通过" if detection_info['result'] == 0 else "检测失败")
            print(f"已生成结果图像: {status_text}")
            
        except Exception as e:
            print(f"生成结果图像失败: {e}")

    def processDetection(self, fileName):
        """处理检测过程"""
        # 显示检测进行中状态
        self.updateCheckInfoPanel(False, False, None, None)
        
        # 延迟执行检测算法，模拟处理时间
        QTimer.singleShot(1000, lambda: self.runDetectionAlgorithm(fileName))
    
    def runDetectionAlgorithm(self, fileName):
        """运行检测算法 - 模板函数"""
        try:
            if self.detector and DETECTION_AVAILABLE:
                # 使用真实检测算法
                result_image, detection_info = self.detector.process_image(fileName)
                
                # 转换结果图像格式
                if result_image is not None:
                    # 如果是numpy数组，转换为QImage格式
                    try:
                        import numpy as np
                        if isinstance(result_image, np.ndarray):
                            height, width, channel = result_image.shape
                            bytes_per_line = 3 * width
                            q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                            buffer = q_image.bits().asstring(q_image.byteCount())
                            result_image_data = bytearray(buffer)
                        else:
                            result_image_data = result_image
                    except:
                        result_image_data = None
                else:
                    result_image_data = None
                
                self.displayDetectionResult(result_image_data, detection_info)
            else:
                # 使用模板检测算法
                self.runTemplateDetection(fileName)
                
        except Exception as e:
            self.handleDetectionError(str(e))
    
    def runTemplateDetection(self, fileName):
        """运行模板检测算法"""
        # =============================================
        # 检测算法模板区域 - 请在此处实现您的检测逻辑
        # =============================================
        
        # 1. 读取图像数据
        image_data = self.loadImageData(fileName)
        
        # 2. 预处理（可选）
        preprocessed_image = self.preprocessImage(image_data)
        
        # 3. 执行检测算法
        detection_result = self.detectDefects(preprocessed_image)
        
        # 4. 后处理和结果生成
        result_image, detection_info = self.postprocessResult(preprocessed_image, detection_result)
        
        # =============================================
        # 模板区域结束
        # =============================================
        
        # 显示检测结果
        self.displayDetectionResult(result_image, detection_info)
    
    def loadImageData(self, fileName):
        """加载图像数据 - 模板函数"""
        # TODO: 实现图像数据加载逻辑
        # 示例：使用PIL或OpenCV加载图像
        # from PIL import Image
        # import numpy as np
        # image = Image.open(fileName)
        # image_array = np.array(image)
        # return image_array
        
        print(f"[模板] 加载图像: {fileName}")
        return fileName  # 临时返回文件名
    
    def preprocessImage(self, image_data):
        """图像预处理 - 模板函数"""
        # TODO: 实现图像预处理逻辑
        # 例如：灰度转换、噪声去除、增强对比度等
        # gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        # denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        # return denoised
        
        print("[模板] 执行图像预处理")
        return image_data
    
    def detectDefects(self, preprocessed_image):
        """缺陷检测 - 模板函数"""
        # TODO: 实现缺陷检测算法
        # 例如：使用机器学习模型、图像处理算法等
        # model_result = your_ml_model.predict(preprocessed_image)
        # defects = process_model_output(model_result)
        # return defects
        
        print("[模板] 执行缺陷检测算法")
        
        # 模拟检测结果（请替换为真实算法）
        import random
        simulated_result = {
            'has_defect': random.choice([True, False]),
            'defect_type': random.choice(['胎冠裂纹', '胎侧气泡', '帘线稀疏', '接头开']),
            'confidence': random.uniform(0.7, 0.99),
            'defect_locations': [(100, 150), (200, 300)]  # 缺陷位置坐标
        }
        return simulated_result
    
    def postprocessResult(self, original_image, detection_result):
        """结果后处理 - 模板函数"""
        # TODO: 实现结果后处理逻辑
        # 例如：在图像上标记缺陷位置、生成检测报告等
        
        print("[模板] 执行结果后处理")
        
        # 生成结果图像（模拟）
        try:
            # 创建结果图像（这里用颜色区分检测结果）
            result_image = QImage(640, 480, QImage.Format_RGB888)
            if detection_result['has_defect']:
                result_image.fill(QColor(255, 200, 200))  # 红色背景表示检测到缺陷
            else:
                result_image.fill(QColor(200, 255, 200))  # 绿色背景表示通过
            
            buffer = result_image.bits().asstring(result_image.byteCount())
            result_image_data = bytearray(buffer)
        except:
            result_image_data = None
        
        # 生成检测信息
        detection_info = {
            'result': 0 if detection_result['has_defect'] else 1,  # 0=不通过, 1=通过
            'message': detection_result['defect_type'] if detection_result['has_defect'] else '检测通过',
            'confidence': detection_result['confidence'],
            'details': f"置信度: {detection_result['confidence']:.2%}"
        }
        
        return result_image_data, detection_info

    def updateCheckInfoPanel(self, first, isFinished, res, msg):
        """更新检测信息面板"""
        lineLabel = self.findChild(QLabel, 'productionLineLabel')
        checkingIcon = self.findChild(QLabel, 'checkingIcon')
        checkingLabel = self.findChild(QLabel, 'checkingLabel')
        resultLabel = self.findChild(QLabel, 'resultLabel')
        errorLabel = self.findChild(QLabel, 'errorLabel')
        
        # 移除服务状态检查，始终显示信息
        
        # 更新生产线信息
        if first and lineLabel:
            if self.current_batch_folder:
                folder_name = os.path.basename(self.current_batch_folder)
                lineLabel.setText(f'批量检测: {folder_name}')
            else:
                lineLabel.setText('轮胎缺陷检测系统 (批量模式)')
            lineLabel.show()
        elif isFinished and checkingIcon and checkingLabel and resultLabel and errorLabel:
            checkingIcon.hide()
            checkingLabel.hide()
            
            # 显示详细信息
            if msg:
                if self.total_images > 0:
                    progress_info = f"进度: {self.processed_images}/{self.total_images}"
                    errorLabel.setText(f"{msg}\n{progress_info}")
                else:
                    errorLabel.setText(msg)
                errorLabel.show()
            else:
                errorLabel.hide()
                
            if res == 1:
                resultLabel.setProperty('resultLabel', True)
                resultLabel.setText('检测通过')
            elif res == 0:
                resultLabel.setProperty('resultLabel', False)
                resultLabel.setText('检测不通过')
            else:
                resultLabel.setProperty('resultLabel', False)
                resultLabel.setText('检测失败')
                
            resultLabel.style().unpolish(resultLabel)
            resultLabel.style().polish(resultLabel)
            resultLabel.show()
        elif not isFinished and checkingIcon and checkingLabel and resultLabel and errorLabel:
            checkingIcon.show()
            
            if checkingLabel:
                if msg:  # 如果传入了自定义消息（如进度信息）
                    checkingLabel.setText(msg)
                else:
                    checkingLabel.setText('正在批量检测图像...')
                checkingLabel.show()
            
            resultLabel.hide()
            errorLabel.hide()

    # 本地功能实现
    def showCameraFunction(self):
        """相机调整功能"""
        if not self.serviceFlag:
            return self.showAlertWindow('请开启服务后再调整相机')
        if self.detectionFlag:
            return self.showAlertWindow('请暂停检测后再调整相机')
        self.showAlertWindow('相机调整功能 (本地模拟)\n可在此处实现相机参数调整')

    def showNetworkFunction(self):
        """网络设置功能"""
        if self.serviceFlag:
            return self.showAlertWindow('请停止服务后再设置网络参数')
        self.showAlertWindow('网络设置功能 (本地模拟)\n可在此处配置IP地址和端口')

    def showParaFunction(self):
        """参数设置功能"""
        if self.serviceFlag:
            return self.showAlertWindow('请停止服务后再设置参数')
        self.showAlertWindow('参数设置功能 (本地模拟)\n可在此处配置检测算法参数')

    def displayDetectionResult(self, result_image_data, detection_info):
        """显示检测结果"""
        try:
            # 显示结果图像
            if result_image_data:
                imgPanel = self.mainPanel.rightPanelWidget.imgPanel
                imgPanel.updateImgPanel(result_image_data, 'right')
            
            # 更新检测信息面板
            self.updateCheckInfoPanel(
                False, 
                True, 
                detection_info['result'], 
                detection_info['message']
            )
            
            # 显示详细信息
            detail_msg = f"检测完成!\n结果: {detection_info['message']}\n{detection_info['details']}"
            self.showAlertWindow(detail_msg)
            
        except Exception as e:
            self.handleDetectionError(f"显示结果失败: {str(e)}")
        finally:
            # 检测完成，重置状态
            self.detectionFlag = False
            self.mainPanel.toolPanelWidget.stopDetectionRender()
    
    def handleDetectionError(self, error_message):
        """处理检测错误"""
        print(f"[错误] {error_message}")
        self.updateCheckInfoPanel(False, True, 2, f"检测失败: {error_message}")
        self.showAlertWindow(f"检测过程中发生错误:\n{error_message}")
        
        # 重置状态
        self.detectionFlag = False
        self.mainPanel.toolPanelWidget.stopDetectionRender()
    
    def showTableFunction(self):
        """结果分析功能 - 查看批量检测统计"""
        if self.processed_images > 0:
            stats_msg = f"批量检测统计\n" \
                       f"\n当前文件夹: {os.path.basename(self.current_batch_folder) if self.current_batch_folder else '无'}" \
                       f"\n已处理: {self.processed_images} 张" \
                       f"\n成功: {self.success_count} 张" \
                       f"\n失败: {self.processed_images - self.success_count} 张" \
                       f"\n成功率: {(self.success_count/self.processed_images*100):.1f}%"
        else:
            stats_msg = "批量检测统计\n\n还未进行批量检测\n\n点击'开始检测'选择文件夹开始批量检测"
        
        self.showAlertWindow(stats_msg)

    def showMenu(self, msg):
        """显示主菜单"""
        if msg:
            self.mainMenu.setPosition()
            self.mainMenu.show()
        else:
            self.mainMenu.hide()

    def togglePanel(self, type):
        """显示/隐藏面板"""
        widget = self.findChild(QWidget, type)
        if widget:
            if widget.isHidden():
                widget.show()
            else:
                widget.hide()

    def showExitWindow(self):
        """直接退出系统，无需确认"""
        print("[系统] 用户点击关闭按钮，直接退出程序")
        # 停止所有正在运行的检测线程
        if self.detectionFlag and self.batch_thread and self.batch_thread.isRunning():
            print("[系统] 停止检测线程")
            self.batch_thread.stop()
            self.batch_thread.wait(1000)  # 等待1秒
        
        # 直接退出程序
        self.exitSystem()

    def exitSystem(self):
        """退出系统"""
        QApplication.instance().quit()

    def showAlertWindow(self, msg):
        """显示警告窗口"""
        self.alertWindow.setMessage(msg)
        self.alertWindow.show()

    def changeWindowSize(self, msg):
        """响应全屏信号"""
        if (msg == 'max'):
            if (self.isFullScreen()):
                self.mainGridLayout.setContentsMargins(self.shadowWidth, self.shadowWidth, 
                    self.shadowWidth, self.shadowWidth)
                self.showNormal()
            else:
                self.mainGridLayout.setContentsMargins(0, 0, 0, 0)
                self.showFullScreen()
            self.mainMenu.update()
        else:
            self.showMinimized()

    def move(self, pos):
        """响应移动窗口的信号"""
        if self.windowState() == Qt.WindowMaximized or self.windowState() == Qt.WindowFullScreen:
            return
        super(SimpleMainWindowPage, self).move(pos - QPoint(self.shadowWidth, self.shadowWidth))

    def eventFilter(self, obj, e):
        """事件过滤"""
        from PyQt5.QtGui import QEnterEvent, QMouseEvent
        if isinstance(e, QEnterEvent):
            self.setCursor(Qt.ArrowCursor)
        if isinstance(e, QMouseEvent):
            self.showMenu(False)
        return super(SimpleMainWindowPage, self).eventFilter(obj, e)
    
    def closeEvent(self, event):
        """窗口关闭事件，直接关闭无需确认"""
        print("[系统] 窗口关闭事件触发，直接关闭")
        # 停止所有正在运行的检测线程
        if self.detectionFlag and self.batch_thread and self.batch_thread.isRunning():
            print("[系统] 停止检测线程")
            self.batch_thread.stop()
            self.batch_thread.wait(1000)  # 等待1秒
        
        # 接受关闭事件，直接关闭窗口
        event.accept()

    def getShadowWidth(self):
        return self.shadowWidth

    def getMenuBarHeight(self):
        return self.menuBar.height()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = SimpleMainWindowPage()
    sys.exit(app.exec_())