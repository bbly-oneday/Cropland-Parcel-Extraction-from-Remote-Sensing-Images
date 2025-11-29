"""
耕地质量时空立方体项目配置文件
"""
import os
from datetime import datetime


class Config:
    """项目配置类"""
    
    # 数据路径配置
    DATA_DIR = os.getenv('DATA_DIR', '/workspace/data')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/workspace/output')
    TEMP_DIR = os.getenv('TEMP_DIR', '/workspace/temp')
    
    # 时空分辨率配置
    SPATIAL_RESOLUTION = float(os.getenv('SPATIAL_RESOLUTION', '0.01'))  # 空间分辨率（度）
    TEMPORAL_RESOLUTION = os.getenv('TEMPORAL_RESOLUTION', 'D')  # 时间分辨率：D-日，W-周，M-月
    
    # 数据字段配置
    REQUIRED_COLUMNS = [
        'longitude',
        'latitude', 
        'date',
        'soil_ph',
        'organic_matter',
        'nitrogen',
        'phosphorus', 
        'potassium'
    ]
    
    # 质量指标配置
    QUALITY_INDICATORS = [
        'soil_ph',
        'organic_matter',
        'nitrogen',
        'phosphorus',
        'potassium'
    ]
    
    # 数据验证配置
    MIN_PH = 4.0
    MAX_PH = 9.0
    MIN_ORGANIC_MATTER = 0.1
    MAX_ORGANIC_MATTER = 10.0
    MIN_NITROGEN = 50
    MAX_NITROGEN = 300
    MIN_PHOSPHORUS = 5
    MAX_PHOSPHORUS = 50
    MIN_POTASSIUM = 50
    MAX_POTASSIUM = 400
    
    # 聚合方法配置
    AGGREGATION_METHOD = os.getenv('AGGREGATION_METHOD', 'mean')  # mean, median, max, min
    
    # 输出格式配置
    OUTPUT_FORMAT = os.getenv('OUTPUT_FORMAT', 'json')  # json, netcdf, hdf5
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        for dir_path in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
            os.makedirs(dir_path, exist_ok=True)


# 初始化配置
Config.create_directories()