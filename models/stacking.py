"""
@author: DemonSong
@time: 20180219
"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import numpy as np

class Ensemble(object):

    # 参数 传入 Ensemble 之前 已经设置完成
    def __init__(self, n_folds, base_model, seed = 0, kBest = False, k = 2, metric='rmse'):
        self.n_folds = n_folds
        self.base_models = base_model
        self.seed = seed
        self.metric = metric
        self.kBest = kBest
        self.k = k


    def _metric(self, label, pred):
        if self.metric == 'rmse':
            return round(np.sqrt(metrics.mean_squared_error(label, pred)), 5)
        elif self.metric == 'auc':
            return round(metrics.roc_auc_score(label, pred), 5)

    # 得到 训练集的一维输出，和测试集n_folds个模型的平均
    def fit_predict(self, X, y, T, predict_proba = False):

        # 如果是Xgboost 需要进行转换？
        X = np.array(X) # 训练集输入
        y = np.array(y) # 训练集输出
        T = np.array(T) # 测试集输入

        if self.kBest:
            select = SelectKBest(f_regression, k=self.k)
            X = select.fit_transform(X, y)
            T = select.transform(T)

        # folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed).split(X)) # shuffle 数据集样本在K_folds之前将被打乱一次
        folds = list(StratifiedKFold(n_splits=self.n_folds, random_state=self.seed, shuffle=True).split(X, y))

        S_train = np.zeros((X.shape[0], 1)) # 训练样本数 * 模型个数
        S_test = np.zeros((T.shape[0], 1))  # 测试集样本数 * 模型个数
        S_test_n = np.zeros((T.shape[0], len(folds))) # 测试集样本数 * n_folds

        CVRESULT = []
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train = X[train_idx] # 训练集特征
            y_train = y[train_idx] # 训练集标签

            X_holdout = X[test_idx] # 待预测的输入
            y_holdout = y[test_idx]

            self.base_models.fit(X_train, y_train)
            if predict_proba:
                y_hold_out_pred = self.base_models.predict_proba(X_holdout)[:, -1]  # 如果是二元分类
            else:
                y_hold_out_pred = self.base_models.predict(X_holdout)

            # 验证集 结果输出
            cvAUC = self._metric(y_holdout, y_hold_out_pred)
            print(j, ' k-folds : auc ', cvAUC)
            CVRESULT.append(cvAUC)
            S_train[test_idx] = y_hold_out_pred.reshape(-1, 1) #


            # 保存测试集上的k-folds的结果
            if predict_proba:
                y_test_pred = self.base_models.predict_proba(T)[:, -1]
            else:
                y_test_pred = self.base_models.predict(T)

            S_test_n[:, j] = y_test_pred

        print('mean auc: ', np.mean(CVRESULT), 'std auc: ', np.std(CVRESULT))
        S_test[:] = S_test_n.mean(1).reshape(-1, 1) # 取均值
        return S_train, S_test, round(np.mean(CVRESULT), 5), round(np.std(CVRESULT), 5)

