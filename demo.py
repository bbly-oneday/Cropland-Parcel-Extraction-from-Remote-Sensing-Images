"""
耕地质量时空立方体演示程序
使用较小的数据集以避免内存问题
"""
import numpy as np
import pandas as pd
from datetime import datetime
import json
from config import Config


def generate_small_sample_data(output_file: str = '/workspace/demo_soil_data.csv'):
    """
    生成小规模示例耕地质量数据
    
    Args:
        output_file: 输出文件路径
    """
    np.random.seed(42)
    
    # 生成示例数据 - 减少数据量
    n_samples = 200  # 减少样本数量
    longitudes = np.random.uniform(116.2, 116.8, n_samples)  # 缩小范围
    latitudes = np.random.uniform(39.2, 39.8, n_samples)
    
    # 生成时间序列 - 缩短时间范围
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)  # 只有3个月的数据
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = np.random.choice(date_range, n_samples)
    
    # 生成质量指标数据
    soil_ph = np.random.normal(6.5, 0.8, n_samples)
    organic_matter = np.random.normal(2.5, 0.5, n_samples)  # 有机质含量 (%)
    nitrogen = np.random.normal(120, 20, n_samples)  # 氮含量 (mg/kg)
    phosphorus = np.random.normal(25, 5, n_samples)  # 磷含量 (mg/kg)
    potassium = np.random.normal(150, 30, n_samples)  # 钾含量 (mg/kg)
    
    # 创建DataFrame
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
    
    # 保存到CSV文件
    df.to_csv(output_file, index=False)
    print(f"小规模示例数据已生成并保存到: {output_file}")
    return df


def demo_cube_creation():
    """
    演示耕地质量时空立方体的创建过程
    """
    print("开始创建耕地质量时空立方体演示...")
    
    # 生成小规模示例数据
    sample_data_path = '/workspace/demo_soil_data.csv'
    df = generate_small_sample_data(sample_data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"空间范围: 经度 {df['longitude'].min():.3f}-{df['longitude'].max():.3f}, "
          f"纬度 {df['latitude'].min():.3f}-{df['latitude'].max():.3f}")
    
    # 导入SoilQualityCube类
    from main import SoilQualityCube
    
    # 创建时空立方体实例，使用较粗的分辨率以减少内存使用
    cube = SoilQualityCube(spatial_resolution=0.02, temporal_resolution='D')
    
    # 聚合数据到立方体
    result = cube.aggregate_data_to_cube(df)
    
    print(f"时空立方体创建完成!")
    print(f"空间网格: {result['spatial_grid']['lon_steps']} x {result['spatial_grid']['lat_steps']}")
    print(f"时间步数: {result['temporal_grid']['time_steps']}")
    print(f"质量指标: {result['quality_indicators']}")
    
    # 获取特定指标的某个时间切片
    if 'soil_ph' in result['quality_indicators']:
        ph_slice = cube.get_cube_slice('soil_ph', time_idx=0)
        print(f"土壤pH在第一个时间点的切片形状: {ph_slice.shape}")
    
    # 保存立方体
    output_path = '/workspace/demo_soil_quality_cube.json'
    cube.save_cube(output_path)
    print(f"时空立方体已保存到: {output_path}")
    
    # 演示如何加载立方体
    new_cube = SoilQualityCube()
    new_cube.load_cube(output_path)
    print("时空立方体已从文件加载完成!")
    
    print("\n演示完成！")


if __name__ == "__main__":
    demo_cube_creation()