import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)


class Configure(object):

    # 数据集存放根目录
    root_data_path = '../input/'

    # 数据子集存放根目录
    root_sub_data_path = '../input/subset12/'

    # 数据集多分类转二分类存放目录
    root_multi2binary_path = '../input/multi2binary/'

    # 特征数据集路径
    root_feature_path = root_data_path + 'data.csv'

    # 模型输出信息
    root_model_info_path = '../models/info/'

    # stacking 单模型输出路径
    root_model_stacking_path = '../models/__models__/'

    # stacking 数据集存放路径
    root_stacking_path = root_data_path + 'data_stacking.csv'
