# 2018“云移杯- 景区口碑评价分值预测 (baseline 0.53362)

## 前言
实在太忙，找实习，天池，华为等比赛都放在一块了，该方案初赛第9，进入复赛之后就先放下了。此处记录从春节到3月份关于NLP的学习感悟，供大家交流学习。

## 任务
根据每个用户的评论，预测他们对景区的情感值（1~5）。

## 思路
1. 分类问题：通过分类器学习评论与情感值的复杂映射关系。
2. 回归问题：情感值实际是有先后等级关系，因此可以采用回归大法，直接预测。

注意：分类可以采用softmax多分的手段，实测效果很差。因此，我最终还是采用了回归大法。

## feature
- tf-idf
- doc2vec
- bag of words

## model
- xgboost bagging regression
- lightgbm bagging regression
- random forest classification
- ridge
- lasso
- lstm

## ensemble
stacking
