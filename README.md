# 耕地质量时空立方体工程

本项目实现了将耕地质量数据汇聚成时空立方体的功能，用于分析和可视化耕地质量的时空变化。

## 功能特性

- 支持多种耕地质量指标（pH值、有机质、氮磷钾含量等）
- 可配置的空间和时间分辨率
- 支持JSON和NetCDF格式输出
- 提供数据加载、聚合、切片和保存功能
- 配置文件管理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```python
from main import SoilQualityCube

# 创建立方体实例
cube = SoilQualityCube()

# 加载数据
df = cube.load_soil_data('your_data.csv')

# 聚合数据到立方体
result = cube.aggregate_data_to_cube(df)

# 保存立方体
cube.save_cube('output_cube.json')
```

### 配置参数

项目使用 `config.py` 文件进行参数配置，包括：

- 数据路径
- 空间和时间分辨率
- 质量指标定义
- 数据验证范围

### 数据格式要求

输入数据需要包含以下字段：

- `longitude`: 经度
- `latitude`: 纬度
- `date`: 日期
- `soil_ph`: 土壤pH值
- `organic_matter`: 有机质含量
- `nitrogen`: 氮含量
- `phosphorus`: 磷含量
- `potassium`: 钾含量

## 项目结构

```
/workspace/
├── main.py          # 主程序文件
├── config.py        # 配置文件
├── requirements.txt # 依赖包列表
├── README.md        # 项目说明
├── data/            # 数据目录
├── output/          # 输出目录
└── temp/            # 临时目录
```

## 输出格式

- JSON格式：便于查看和调试
- NetCDF格式：适合科学计算和大数据量存储

## 示例数据

运行 `python main.py` 可以生成示例数据并演示完整的立方体构建过程。
