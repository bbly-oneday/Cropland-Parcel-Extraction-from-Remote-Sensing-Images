"""
耕地质量数据汇聚成时空立方体主程序
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
import os

from config import Config


class SoilQualityCube:
    """
    耕地质量时空立方体类
    用于将耕地质量数据汇聚成时空立方体格式
    """
    
    def __init__(self, spatial_resolution: float = None, temporal_resolution: str = None):
        """
        初始化时空立方体
        
        Args:
            spatial_resolution: 空间分辨率（度），如果为None则使用配置文件中的值
            temporal_resolution: 时间分辨率（D-日，W-周，M-月），如果为None则使用配置文件中的值
        """
        self.spatial_resolution = spatial_resolution or Config.SPATIAL_RESOLUTION
        self.temporal_resolution = temporal_resolution or Config.TEMPORAL_RESOLUTION
        self.data_cube = {}
        self.spatial_grid = {}
        self.temporal_grid = {}
        self.quality_indicators = []
    
    def load_soil_data(self, data_file: str) -> pd.DataFrame:
        """
        加载耕地质量数据
        
        Args:
            data_file: 数据文件路径
            
        Returns:
            DataFrame: 耕地质量数据
        """
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        elif data_file.endswith('.xlsx'):
            df = pd.read_excel(data_file)
        else:
            raise ValueError("不支持的数据文件格式，仅支持CSV和XLSX")
        
        # 验证必要字段
        missing_cols = [col for col in Config.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要字段: {missing_cols}")
        
        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def create_spatial_grid(self, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> Dict:
        """
        创建空间网格
        
        Args:
            min_lon: 最小经度
            max_lon: 最大经度
            min_lat: 最小纬度
            max_lat: 最大纬度
            
        Returns:
            Dict: 空间网格信息
        """
        lon_steps = int((max_lon - min_lon) / self.spatial_resolution) + 1
        lat_steps = int((max_lat - min_lat) / self.spatial_resolution) + 1
        
        grid = {
            'min_lon': min_lon,
            'max_lon': max_lon,
            'min_lat': min_lat,
            'max_lat': max_lat,
            'lon_steps': lon_steps,
            'lat_steps': lat_steps,
            'resolution': self.spatial_resolution
        }
        
        self.spatial_grid = grid
        return grid
    
    def create_temporal_grid(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        创建时间网格
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 时间网格信息
        """
        if self.temporal_resolution == 'D':
            time_steps = (end_date - start_date).days + 1
        elif self.temporal_resolution == 'W':
            time_steps = ((end_date - start_date).days // 7) + 1
        elif self.temporal_resolution == 'M':
            months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
            time_steps = months
        else:
            raise ValueError("不支持的时间分辨率")
        
        grid = {
            'start_date': start_date,
            'end_date': end_date,
            'resolution': self.temporal_resolution,
            'time_steps': time_steps
        }
        
        self.temporal_grid = grid
        return grid
    
    def aggregate_data_to_cube(self, df: pd.DataFrame) -> Dict:
        """
        将耕地质量数据聚合到时空立方体
        
        Args:
            df: 耕地质量数据DataFrame
            
        Returns:
            Dict: 包含时空立方体数据的字典
        """
        # 获取空间和时间范围
        min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
        min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
        start_date, end_date = df['date'].min(), df['date'].max()
        
        # 创建时空网格
        self.create_spatial_grid(min_lon, max_lon, min_lat, max_lat)
        self.create_temporal_grid(start_date, end_date)
        
        # 获取质量指标列（使用配置文件中定义的指标）
        quality_cols = [col for col in Config.QUALITY_INDICATORS if col in df.columns]
        self.quality_indicators = quality_cols
        
        # 初始化立方体数据结构
        cube_data = {}
        for indicator in quality_cols:
            cube_data[indicator] = np.full(
                (self.spatial_grid['lat_steps'], self.spatial_grid['lon_steps'], self.temporal_grid['time_steps']), 
                np.nan
            )
        
        # 将数据聚合到立方体中
        for _, row in df.iterrows():
            # 计算空间索引
            lon_idx = int((row['longitude'] - self.spatial_grid['min_lon']) / self.spatial_resolution)
            lat_idx = int((row['latitude'] - self.spatial_grid['min_lat']) / self.spatial_resolution)
            
            # 计算时间索引
            time_diff = row['date'] - self.temporal_grid['start_date']
            if self.temporal_resolution == 'D':
                time_idx = time_diff.days
            elif self.temporal_resolution == 'W':
                time_idx = time_diff.days // 7
            elif self.temporal_resolution == 'M':
                time_idx = (row['date'].year - self.temporal_grid['start_date'].year) * 12 + \
                          (row['date'].month - self.temporal_grid['start_date'].month)
            
            # 检查索引是否在范围内
            if (0 <= lon_idx < self.spatial_grid['lon_steps'] and 
                0 <= lat_idx < self.spatial_grid['lat_steps'] and 
                0 <= time_idx < self.temporal_grid['time_steps']):
                
                # 聚合数据（这里使用平均值，可根据需要修改）
                for indicator in quality_cols:
                    if pd.notna(row[indicator]):
                        if np.isnan(cube_data[indicator][lat_idx, lon_idx, time_idx]):
                            cube_data[indicator][lat_idx, lon_idx, time_idx] = row[indicator]
                        else:
                            # 如果已有值，可以采用平均或其他聚合方式
                            cube_data[indicator][lat_idx, lon_idx, time_idx] = (
                                cube_data[indicator][lat_idx, lon_idx, time_idx] + row[indicator]
                            ) / 2
        
        self.data_cube = cube_data
        return {
            'cube_data': cube_data,
            'spatial_grid': self.spatial_grid,
            'temporal_grid': self.temporal_grid,
            'quality_indicators': self.quality_indicators
        }
    
    def get_cube_slice(self, indicator: str, time_idx: Optional[int] = None, 
                      lat_range: Optional[Tuple[int, int]] = None, 
                      lon_range: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        获取立方体切片数据
        
        Args:
            indicator: 质量指标名称
            time_idx: 时间索引（None表示所有时间）
            lat_range: 纬度范围 (start, end)
            lon_range: 经度范围 (start, end)
            
        Returns:
            np.ndarray: 立方体切片数据
        """
        if indicator not in self.data_cube:
            raise ValueError(f"指标 {indicator} 不在立方体中")
        
        data = self.data_cube[indicator]
        
        # 应用切片
        if time_idx is not None:
            data = data[:, :, time_idx]
        if lat_range:
            data = data[lat_range[0]:lat_range[1], :]
        if lon_range:
            data = data[:, lon_range[0]:lon_range[1]]
        
        return data
    
    def save_cube(self, output_path: str = None):
        """
        保存时空立方体到文件
        
        Args:
            output_path: 输出文件路径，如果为None则使用配置中的默认路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(Config.OUTPUT_DIR, f'soil_quality_cube_{timestamp}.json')
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 根据文件扩展名选择输出格式
        if output_path.endswith('.json'):
            self._save_as_json(output_path)
        elif output_path.endswith('.nc'):  # NetCDF格式
            self._save_as_netcdf(output_path)
        else:
            # 默认使用JSON格式
            self._save_as_json(output_path)
    
    def _save_as_json(self, output_path: str):
        """
        以JSON格式保存立方体
        
        Args:
            output_path: 输出文件路径
        """
        # 保存立方体数据和元数据
        output_data = {
            'data_cube': {k: v.tolist() for k, v in self.data_cube.items()},  # 转换为列表以便JSON序列化
            'spatial_grid': self.spatial_grid,
            'temporal_grid': self.temporal_grid,
            'quality_indicators': self.quality_indicators,
            'spatial_resolution': self.spatial_resolution,
            'temporal_resolution': self.temporal_resolution
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_as_netcdf(self, output_path: str):
        """
        以NetCDF格式保存立方体（需要安装netcdf4包）
        
        Args:
            output_path: 输出文件路径
        """
        try:
            from netCDF4 import Dataset
            import numpy as np
            
            # 创建NetCDF文件
            with Dataset(output_path, 'w', format='NETCDF4') as nc:
                # 创建维度
                lat_dim = nc.createDimension('latitude', self.spatial_grid['lat_steps'])
                lon_dim = nc.createDimension('longitude', self.spatial_grid['lon_steps'])
                time_dim = nc.createDimension('time', self.temporal_grid['time_steps'])
                
                # 创建坐标变量
                lat_var = nc.createVariable('latitude', 'f4', ('latitude',))
                lon_var = nc.createVariable('longitude', 'f4', ('longitude',))
                time_var = nc.createVariable('time', 'i4', ('time',))
                
                # 设置坐标变量的值
                lat_var[:] = np.linspace(
                    self.spatial_grid['min_lat'], 
                    self.spatial_grid['max_lat'], 
                    self.spatial_grid['lat_steps']
                )
                lon_var[:] = np.linspace(
                    self.spatial_grid['min_lon'], 
                    self.spatial_grid['max_lon'], 
                    self.spatial_grid['lon_steps']
                )
                time_var[:] = range(self.temporal_grid['time_steps'])
                
                # 为每个质量指标创建变量
                for indicator, data in self.data_cube.items():
                    var = nc.createVariable(indicator, 'f4', ('latitude', 'longitude', 'time'), 
                                           fill_value=np.nan, zlib=True)
                    var[:, :, :] = data
                    var.long_name = f"{indicator} measurement"
                    var.units = "varies"  # 根据具体指标设置单位
                
                # 添加全局属性
                nc.description = 'Soil Quality Spatio-Temporal Cube'
                nc.spatial_resolution = self.spatial_resolution
                nc.temporal_resolution = self.temporal_resolution
                nc.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
        except ImportError:
            print("警告: 未安装netCDF4包，无法保存为NetCDF格式。请运行: pip install netcdf4")
            # 回退到JSON格式
            json_path = output_path.replace('.nc', '.json')
            print(f"保存为JSON格式: {json_path}")
            self._save_as_json(json_path)
    
    def load_cube(self, input_path: str):
        """
        从文件加载时空立方体
        
        Args:
            input_path: 输入文件路径
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 恢复数据立方体
        self.data_cube = {k: np.array(v) for k, v in data['data_cube'].items()}
        self.spatial_grid = data['spatial_grid']
        self.temporal_grid = data['temporal_grid']
        self.quality_indicators = data['quality_indicators']
        self.spatial_resolution = data['spatial_resolution']
        self.temporal_resolution = data['temporal_resolution']


def generate_sample_data(output_file: str = '/workspace/sample_soil_data.csv'):
    """
    生成示例耕地质量数据
    
    Args:
        output_file: 输出文件路径
    """
    np.random.seed(42)
    
    # 生成示例数据
    n_samples = 1000
    longitudes = np.random.uniform(116.0, 117.0, n_samples)  # 北京附近
    latitudes = np.random.uniform(39.0, 40.0, n_samples)
    
    # 生成时间序列
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
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
    print(f"示例数据已生成并保存到: {output_file}")
    return df


def main():
    """
    主函数 - 演示耕地质量时空立方体的创建过程
    """
    print("开始创建耕地质量时空立方体...")
    
    # 生成示例数据
    sample_data_path = '/workspace/sample_soil_data.csv'
    df = generate_sample_data(sample_data_path)
    
    print(f"数据形状: {df.shape}")
    print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"空间范围: 经度 {df['longitude'].min():.3f}-{df['longitude'].max():.3f}, "
          f"纬度 {df['latitude'].min():.3f}-{df['latitude'].max():.3f}")
    
    # 创建时空立方体实例
    cube = SoilQualityCube(spatial_resolution=0.02, temporal_resolution='D')
    
    # 聚合数据到立方体
    result = cube.aggregate_data_to_cube(df)
    
    print(f"时空立方体创建完成!")
    print(f"空间网格: {result['spatial_grid']['lon_steps']} x {result['spatial_grid']['lat_steps']}")
    print(f"时间步数: {result['temporal_grid']['time_steps']}")
    print(f"质量指标: {result['quality_indicators']}")
    
    # 获取特定指标的某个时间切片
    ph_slice = cube.get_cube_slice('soil_ph', time_idx=0)
    print(f"土壤pH在第一个时间点的切片形状: {ph_slice.shape}")
    
    # 保存立方体
    output_path = '/workspace/soil_quality_cube.json'
    cube.save_cube(output_path)
    print(f"时空立方体已保存到: {output_path}")
    
    # 演示如何加载立方体
    new_cube = SoilQualityCube()
    new_cube.load_cube(output_path)
    print("时空立方体已从文件加载完成!")


if __name__ == "__main__":
    main()