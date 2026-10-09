import os, sys
sys.path.append(os.path.join(__file__, '../../'))
from datetime import datetime

from PyQt5.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap
from PyQt5.QtCore import Qt, QSize, QTimer
from widgets.PanelSplit import PanelSplitWidget
from common.commlib import loadQss, renderShadow

staticPath = os.path.join(__file__, '../../../') + '/static'

class ImagePanelWidget(QWidget):
  def __init__(self):
    super().__init__()

    # 标题高度
    self.titleHeight = 40
    # 图片和容器的内边距
    self.imgPadding = 5
    # 图片高宽比
    self.imgRatio = 1080 / 1920
    # 样式表
    self.qss = staticPath + '/qss/image_panel.qss'

    self.render()
      
  def render(self):
    self.setAttribute(Qt.WA_StyledBackground, True)
    self.setStyleSheet(loadQss(self.qss))

    # 布局
    self.imgPanelLayout = QGridLayout()
    self.imgPanelLayout.setContentsMargins(12, 12, 12, 6)
    self.imgPanelLayout.setSpacing(0)

    # 图片面板
    self.innerPanel = QWidget()
    self.innerPanel.setObjectName('imgInnerPanel')
    
    self.innerPanelLayout = QGridLayout()
    self.innerPanelLayout.setContentsMargins(0, 0, 0, 0)
    self.innerPanelLayout.setSpacing(0)
    self.innerTitle = QWidget()
    self.innerTitle.setObjectName('imgInnerTitle')
    self.innerTitle.setFixedHeight(self.titleHeight)
    self.innerTitleLayout = QHBoxLayout()
    self.innerTitleIcon = QLabel()
    self.innerTitleIcon.setObjectName('innerTitleIcon')
    self.innerTitleIcon.setFixedWidth(24)
    icon = QPixmap()
    icon.load(staticPath + '/icons/image.svg')
    self.innerTitleIcon.setPixmap(icon.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    self.innerTitleText = QLabel()
    self.innerTitleText.setObjectName('innerTitleText')
    self.innerTitleText.setText('图像信息')
    self.innerTitleText.setFont(QFont('Microsoft YaHei'))
    self.innerTitleLayout.addWidget(self.innerTitleIcon)
    self.innerTitleLayout.addWidget(self.innerTitleText)
    self.innerTitle.setLayout(self.innerTitleLayout)

    self.innerBody = QWidget()
    self.innerBodyLayout = QHBoxLayout()
    self.innerBodyLayout.setContentsMargins(0, 0, 0, 0)
    self.innerBodyLayout.setSpacing(0)

    self.innerBodyLeft = QWidget()
    self.innerBodyLeft.setObjectName('imgLeft')
    self.innerBodyLeftLayout = QGridLayout()
    self.innerBodyLeftLayout.setContentsMargins(0, 0, 0, 0)
    self.innerBodyLeftLayout.setSpacing(0)
    self.innerBodyLeftTitle = QWidget()
    self.innerBodyLeftTitleLayout = QHBoxLayout()
    self.innerBodyLeftTitleLayout.setContentsMargins(10, 0, 10, 0)
    self.innerBodyLeftName = QLabel()
    self.innerBodyLeftName.setObjectName('innerBodyLeftName')
    self.innerBodyLeftName.setFixedWidth(70)
    self.innerBodyLeftName.setText('原始图像')
    self.innerBodyLeftName.setFont(QFont('Microsoft YaHei'))
    self.innerBodyLeftTime = QLabel()
    self.innerBodyLeftTime.setObjectName('innerBodyLeftTime')
    self.innerBodyLeftTime.setFont(QFont('Microsoft YaHei'))
    self.innerBodyLeftTitleLayout.addWidget(self.innerBodyLeftName)
    self.innerBodyLeftTitleLayout.addWidget(self.innerBodyLeftTime)
    self.innerBodyLeftTitle.setLayout(self.innerBodyLeftTitleLayout)
    self.innerBodyLeftTitle.setFixedHeight(30)

    self.innerBodyLeftContent = QWidget()
    self.innerBodyLeftContentLayout = QGridLayout()
    self.innerBodyLeftContentLayout.setContentsMargins(self.imgPadding, 0, self.imgPadding, self.imgPadding)

    self.innerBodyLeftContentScroll = QScrollArea()
    self.innerBodyLeftContentScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    self.innerBodyLeftContentScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.innerBodyLeftContentImage = QLabel()
    self.innerBodyLeftContentImage.setStyleSheet('background-color: #F6F6F6; border: 1px solid #CCCCCC')
    # 设置图片显示属性
    self.innerBodyLeftContentImage.setAlignment(Qt.AlignCenter)
    self.innerBodyLeftContentImage.setMinimumSize(200, 150)  # 设置合理的最小尺寸
    self.innerBodyLeftContentImage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.innerBodyLeftContentImage.setScaledContents(False)  # 保持纵横比
    self.innerBodyLeftContentImage.setText('原始图像')

    self.innerBodyLeftContentScroll.setWidget(self.innerBodyLeftContentImage)
    self.innerBodyLeftContentLayout.addWidget(self.innerBodyLeftContentScroll)
    self.innerBodyLeftContent.setLayout(self.innerBodyLeftContentLayout)
    self.innerBodyLeftLayout.addWidget(self.innerBodyLeftTitle)
    self.innerBodyLeftLayout.addWidget(self.innerBodyLeftContent)
    self.innerBodyLeft.setLayout(self.innerBodyLeftLayout)

    # 分割线
    self.innerBodySplit = PanelSplitWidget()

    self.innerBodyRight = QWidget()
    self.innerBodyRightLayout = QGridLayout()
    self.innerBodyRightLayout.setContentsMargins(0, 0, 0, 0)
    self.innerBodyRightLayout.setSpacing(0)
    self.innerBodyRightTitle = QWidget()
    self.innerBodyRightTitleLayout = QHBoxLayout()
    self.innerBodyRightTitleLayout.setContentsMargins(10, 0, 10, 0)
    self.innerBodyRightName = QLabel()
    self.innerBodyRightName.setObjectName('innerBodyRightName')
    self.innerBodyRightName.setFixedWidth(70)
    self.innerBodyRightName.setText('检测结果')
    self.innerBodyRightName.setFont(QFont('Microsoft YaHei'))
    self.innerBodyRightTime = QLabel()
    self.innerBodyRightTime.setObjectName('innerBodyRightTime')
    self.innerBodyRightTime.setFont(QFont('Microsoft YaHei'))
    self.innerBodyRightTitleLayout.addWidget(self.innerBodyRightName)
    self.innerBodyRightTitleLayout.addWidget(self.innerBodyRightTime)
    self.innerBodyRightTitle.setLayout(self.innerBodyRightTitleLayout)
    self.innerBodyRightTitle.setFixedHeight(30)
    self.innerBodyRight.setObjectName('imgRight')

    self.innerBodyRightContent = QWidget()
    self.innerBodyRightContentLayout = QGridLayout()
    self.innerBodyRightContentLayout.setContentsMargins(self.imgPadding, 0, self.imgPadding, self.imgPadding)

    self.innerBodyRightContentScroll = QScrollArea()
    self.innerBodyRightContentScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    self.innerBodyRightContentScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    self.innerBodyRightContentImage = QLabel()
    self.innerBodyRightContentImage.setStyleSheet('background-color: #F6F6F6; border: 1px solid #CCCCCC')
    # 设置图片显示属性
    self.innerBodyRightContentImage.setAlignment(Qt.AlignCenter)
    self.innerBodyRightContentImage.setMinimumSize(200, 150)  # 设置合理的最小尺寸
    self.innerBodyRightContentImage.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.innerBodyRightContentImage.setScaledContents(False)  # 保持纵横比
    self.innerBodyRightContentImage.setText('检测结果')

    self.innerBodyRightContentScroll.setWidget(self.innerBodyRightContentImage)
    self.innerBodyRightContentLayout.addWidget(self.innerBodyRightContentScroll)
    self.innerBodyRightContent.setLayout(self.innerBodyRightContentLayout)
    self.innerBodyRightLayout.addWidget(self.innerBodyRightTitle)
    self.innerBodyRightLayout.addWidget(self.innerBodyRightContent)
    self.innerBodyRight.setLayout(self.innerBodyRightLayout)

    self.innerBodyLayout.addWidget(self.innerBodyLeft)
    self.innerBodyLayout.addWidget(self.innerBodySplit)
    self.innerBodyLayout.addWidget(self.innerBodyRight)
    self.innerBody.setLayout(self.innerBodyLayout)
    
    self.innerPanelLayout.addWidget(self.innerTitle)
    self.innerPanelLayout.addWidget(self.innerBody)
    self.innerPanel.setLayout(self.innerPanelLayout)

    # 图片面板设置阴影
    renderShadow(self, self.innerPanel, 0, 0, 10, QColor(200, 200, 200))

    self.imgPanelLayout.addWidget(self.innerPanel)
    self.setLayout(self.imgPanelLayout)


  # 更新图片面板
  def updateImgPanel(self, imgArr, dir):
    try:
      imgBytes = bytes(imgArr)
      img = QImage().fromData(imgBytes)
      pixmap = QPixmap.fromImage(img)
      print(f"Image loading: {img.isNull()}, size: {img.width()}x{img.height()}")
      
      # 更新原始图片
      if dir == 'left':
        self.innerBodyLeftTime.setText(str(datetime.now()))
        self.innerBodyLeftContentImage.setPixmap(pixmap)
        print(f"Left image set, pixmap size: {pixmap.width()}x{pixmap.height()}")
      # 更新结果图片
      else:
        self.innerBodyRightTime.setText(str(datetime.now()))
        self.innerBodyRightContentImage.setPixmap(pixmap)
        print(f"Right image set, pixmap size: {pixmap.width()}x{pixmap.height()}")
        
    except Exception as e:
      print(f"更新图片面板失败: {e}")
  
  # 直接设置左侧图像
  def setLeftImage(self, pixmap):
    try:
      self.innerBodyLeftTime.setText(str(datetime.now()))
      # 获取容器尺寸进行缩放
      if not pixmap.isNull():
        # 获取滚动区域的尺寸（更准确）
        scroll_size = self.innerBodyLeftContentScroll.size()
        available_width = max(scroll_size.width() - 60, 200)  # 留出足够的边距和滚动条空间
        available_height = max(scroll_size.height() - 60, 150)  # 留出足够的边距
        
        # 计算缩放比例，确保图像不超出窗口
        scale_w = available_width / pixmap.width()
        scale_h = available_height / pixmap.height()
        scale = min(scale_w, scale_h)  # 取较小值，确保不超出窗口
        
        # 限制缩放范围，避免过大或过小
        scale = max(0.1, min(scale, 2.0))  # 最小0.1倍，最大2倍
        
        # 应用缩放
        scaled_pixmap = pixmap.scaled(
          int(pixmap.width() * scale), 
          int(pixmap.height() * scale),
          Qt.KeepAspectRatio, 
          Qt.SmoothTransformation
        )
        self.innerBodyLeftContentImage.setPixmap(scaled_pixmap)
        # 调整QLabel尺寸以适应图像
        self.innerBodyLeftContentImage.resize(scaled_pixmap.size())
          
        print(f"Left image set directly, original: {pixmap.width()}x{pixmap.height()}, scaled: {scaled_pixmap.width()}x{scaled_pixmap.height()}, scale: {scale:.2f}, available: {available_width}x{available_height}")
    except Exception as e:
      print(f"设置左侧图像失败: {e}")
  
  # 直接设置右侧图像
  def setRightImage(self, pixmap):
    try:
      self.innerBodyRightTime.setText(str(datetime.now()))
      # 获取容器尺寸进行缩放
      if not pixmap.isNull():
        # 获取滚动区域的尺寸（更准确）
        scroll_size = self.innerBodyRightContentScroll.size()
        available_width = max(scroll_size.width() - 60, 200)  # 留出足够的边距和滚动条空间
        available_height = max(scroll_size.height() - 60, 150)  # 留出足够的边距
        
        # 计算缩放比例，确保图像不超出窗口
        scale_w = available_width / pixmap.width()
        scale_h = available_height / pixmap.height()
        scale = min(scale_w, scale_h)  # 取较小值，确保不超出窗口
        
        # 限制缩放范围，避免过大或过小
        scale = max(0.1, min(scale, 2.0))  # 最小0.1倍，最大2倍
        
        # 应用缩放
        scaled_pixmap = pixmap.scaled(
          int(pixmap.width() * scale), 
          int(pixmap.height() * scale),
          Qt.KeepAspectRatio, 
          Qt.SmoothTransformation
        )
        self.innerBodyRightContentImage.setPixmap(scaled_pixmap)
        # 调整QLabel尺寸以适应图像
        self.innerBodyRightContentImage.resize(scaled_pixmap.size())
          
        print(f"Right image set directly, original: {pixmap.width()}x{pixmap.height()}, scaled: {scaled_pixmap.width()}x{scaled_pixmap.height()}, scale: {scale:.2f}, available: {available_width}x{available_height}")
    except Exception as e:
      print(f"设置右侧图像失败: {e}")


  # 清空图片面板
  def clearImgPanel(self, dir):
    if dir == 'left':
      self.innerBodyLeftTime.setText('')
      self.innerBodyLeftContentImage.clear()
      self.innerBodyLeftContentImage.setText('原始图像')
    else:
      self.innerBodyRightTime.setText('')
      self.innerBodyRightContentImage.clear()
      self.innerBodyRightContentImage.setText('检测结果')


  def paintEvent(self, e):
    # 调整左右两侧图片的尺寸
    try:
      # 调整左侧图像尺寸（不要设置过大的最小尺寸）
      left_width = max(int(self.innerBodyLeftContent.width() - 2 * self.imgPadding), 200)
      left_height = max(int(self.innerBodyLeftContent.height() - 2 * self.imgPadding), 150)
      self.innerBodyLeftContentImage.setMinimumSize(left_width, left_height)
      
      # 调整右侧图像尺寸（不要设置过大的最小尺寸）
      right_width = max(int(self.innerBodyRightContent.width() - 2 * self.imgPadding), 200)
      right_height = max(int(self.innerBodyRightContent.height() - 2 * self.imgPadding), 150)
      self.innerBodyRightContentImage.setMinimumSize(right_width, right_height)
      
    except Exception as e:
      print(f"paintEvent 调整尺寸失败: {e}")
  
  def resizeEvent(self, event):
    """窗口尺寸改变时重新调整图像显示"""
    super().resizeEvent(event)
    # 延迟调整，等待布局完成
    QTimer.singleShot(100, self.adjustImageSizes)
  
  def adjustImageSizes(self):
    """调整图像尺寸以适应容器"""
    try:
      # 重新设置左侧图像
      if hasattr(self.innerBodyLeftContentImage, 'pixmap') and self.innerBodyLeftContentImage.pixmap():
        pixmap = self.innerBodyLeftContentImage.pixmap()
        if not pixmap.isNull():
          self.setLeftImageWithoutTime(pixmap)
      
      # 重新设置右侧图像
      if hasattr(self.innerBodyRightContentImage, 'pixmap') and self.innerBodyRightContentImage.pixmap():
        pixmap = self.innerBodyRightContentImage.pixmap()
        if not pixmap.isNull():
          self.setRightImageWithoutTime(pixmap)
          
    except Exception as e:
      print(f"调整图像尺寸失败: {e}")
  
  def setLeftImageWithoutTime(self, pixmap):
    """设置左侧图像（不更新时间）"""
    try:
      if not pixmap.isNull():
        scroll_size = self.innerBodyLeftContentScroll.size()
        available_width = max(scroll_size.width() - 60, 200)
        available_height = max(scroll_size.height() - 60, 150)
        
        scale_w = available_width / pixmap.width()
        scale_h = available_height / pixmap.height()
        scale = min(scale_w, scale_h)  # 取较小值，确保不超出窗口
        
        # 限制缩放范围
        scale = max(0.1, min(scale, 2.0))
        
        scaled_pixmap = pixmap.scaled(
          int(pixmap.width() * scale), 
          int(pixmap.height() * scale),
          Qt.KeepAspectRatio, 
          Qt.SmoothTransformation
        )
        self.innerBodyLeftContentImage.setPixmap(scaled_pixmap)
        self.innerBodyLeftContentImage.resize(scaled_pixmap.size())
    except Exception as e:
      print(f"设置左侧图像失败: {e}")
  
  def setRightImageWithoutTime(self, pixmap):
    """设置右侧图像（不更新时间）"""
    try:
      if not pixmap.isNull():
        scroll_size = self.innerBodyRightContentScroll.size()
        available_width = max(scroll_size.width() - 60, 200)
        available_height = max(scroll_size.height() - 60, 150)
        
        scale_w = available_width / pixmap.width()
        scale_h = available_height / pixmap.height()
        scale = min(scale_w, scale_h)  # 取较小值，确保不超出窗口
        
        # 限制缩放范围
        scale = max(0.1, min(scale, 2.0))
        
        scaled_pixmap = pixmap.scaled(
          int(pixmap.width() * scale), 
          int(pixmap.height() * scale),
          Qt.KeepAspectRatio, 
          Qt.SmoothTransformation
        )
        self.innerBodyRightContentImage.setPixmap(scaled_pixmap)
        self.innerBodyRightContentImage.resize(scaled_pixmap.size())
    except Exception as e:
      print(f"设置右侧图像失败: {e}")