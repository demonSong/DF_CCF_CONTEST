import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import time

import pandas as pd
import numpy as np
from conf.configure import Configure

def lgbm_importance(clf):
    split_importance = clf.feature_importance(importance_type='split')
    gains_importance = clf.feature_importance(importance_type='gain')

    all_split_times = np.sum(split_importance)
    split_importance = [val / all_split_times for val in split_importance]

    feature_names = clf.feature_name()

    return split_importance, gains_importance, feature_names

def get_weights(clf):
    strTime = time.strftime('%m%d%H%M',time.localtime(time.time()))

    split_importance, gains_importance, feature_names = lgbm_importance(clf)
    sp_df = pd.DataFrame({
        'feature_name':feature_names,
        'split_importance':split_importance
    })
    sp_df.to_csv(Configure.root_model_info_path + 'split_importance_{}.csv'.format(strTime), index = False)

    ga_df = pd.DataFrame({
        'feature_name': feature_names,
        'gain_importance': gains_importance
    })

    ga_df.to_csv(Configure.root_model_info_path + 'gain_importance_{}.csv'.format(strTime), index=False)