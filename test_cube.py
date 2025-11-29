"""
耕地质量时空立方体测试脚本
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目路径
sys.path.append('/workspace')

from main import SoilQualityCube, generate_sample_data
from config import Config


def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 生成示例数据
    sample_data_path = '/workspace/test_data.csv'
    df = generate_sample_data(sample_data_path)
    print(f"✓ 生成示例数据: {df.shape}")
    
    # 创建立方体实例
    cube = SoilQualityCube(spatial_resolution=0.02, temporal_resolution='D')
    print("✓ 创建立方体实例")
    
    # 聚合数据到立方体
    result = cube.aggregate_data_to_cube(df)
    print(f"✓ 数据聚合完成")
    print(f"  - 空间网格: {result['spatial_grid']['lon_steps']} x {result['spatial_grid']['lat_steps']}")
    print(f"  - 时间步数: {result['temporal_grid']['time_steps']}")
    print(f"  - 质量指标: {result['quality_indicators']}")
    
    # 测试获取立方体切片
    if 'soil_ph' in result['quality_indicators']:
        ph_slice = cube.get_cube_slice('soil_ph', time_idx=0)
        print(f"✓ 获取切片成功: {ph_slice.shape}")
    
    # 测试保存功能
    output_path = '/workspace/test_output.json'
    cube.save_cube(output_path)
    print(f"✓ 保存立方体到: {output_path}")
    
    # 测试加载功能
    new_cube = SoilQualityCube()
    new_cube.load_cube(output_path)
    print("✓ 加载立方体成功")
    
    print("✓ 所有基本功能测试通过！\n")


def test_config_usage():
    """测试配置文件使用"""
    print("=== 测试配置文件使用 ===")
    
    print(f"✓ 数据目录: {Config.DATA_DIR}")
    print(f"✓ 输出目录: {Config.OUTPUT_DIR}")
    print(f"✓ 临时目录: {Config.TEMP_DIR}")
    print(f"✓ 空间分辨率: {Config.SPATIAL_RESOLUTION}")
    print(f"✓ 时间分辨率: {Config.TEMPORAL_RESOLUTION}")
    print(f"✓ 必需字段: {Config.REQUIRED_COLUMNS}")
    print(f"✓ 质量指标: {Config.QUALITY_INDICATORS}")
    
    print("✓ 配置文件测试通过！\n")


def test_data_validation():
    """测试数据验证"""
    print("=== 测试数据验证 ===")
    
    # 创建测试数据
    n_samples = 100
    np.random.seed(42)
    
    longitudes = np.random.uniform(116.0, 117.0, n_samples)
    latitudes = np.random.uniform(39.0, 40.0, n_samples)
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = np.random.choice(date_range, n_samples)
    
    # 生成质量指标数据
    soil_ph = np.random.normal(6.5, 0.8, n_samples)
    organic_matter = np.random.normal(2.5, 0.5, n_samples)
    nitrogen = np.random.normal(120, 20, n_samples)
    phosphorus = np.random.normal(25, 5, n_samples)
    potassium = np.random.normal(150, 30, n_samples)
    
    df = pd.DataFrame({
        'longitude': longitudes,
        'latitude': latitudes,
        'date': dates,
        'soil_ph': soil_ph,
        'organic_matter': organic_matter,
        'nitrogen': nitrogen,
        'phosphorus': phosphorus,
        'potassium': potassium
    })
    
    # 测试加载数据
    cube = SoilQualityCube()
    try:
        # 验证必需字段
        required_cols = set(Config.REQUIRED_COLUMNS)
        actual_cols = set(df.columns)
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(f"缺少字段: {missing}")
        print("✓ 数据字段验证通过")
    except Exception as e:
        print(f"✗ 数据字段验证失败: {e}")
        return
    
    # 验证数据类型
    try:
        df['date'] = pd.to_datetime(df['date'])
        print("✓ 日期格式验证通过")
    except Exception as e:
        print(f"✗ 日期格式验证失败: {e}")
        return
    
    print("✓ 数据验证测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("开始运行耕地质量时空立方体测试...\n")
    
    test_config_usage()
    test_data_validation()
    test_basic_functionality()
    
    print("🎉 所有测试完成！")


if __name__ == "__main__":
    run_all_tests()