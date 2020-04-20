"""
@author:
@file: run.py
@time: 2020/4/15 17:55
@desc: 推荐系统算法评估
"""
import random
import numpy as np
import pandas as pd
from numpy import mat
from algorithm.pkg import load_data, svd_recommend, rmse, mae, mse


users_foods, foods, users = load_data()
original_users_foods = users_foods
data_info = []
# 数据集划分 训练与测试,随机选择20%的数据做为验证集
for u in range(users_foods.shape[0]):
    for i in range(users_foods.shape[1]):
        r = users_foods.iloc[u, i]
        if r > 0:
            data_info.append((r, u, i))
random.seed(20)
test_idx = random.sample(range(0, len(data_info)), int(0.2 * len(data_info)))
test_truth_rate = []
test_truth_idx = []
for idx in test_idx:
    test_truth_rate.append(data_info[idx][0])
    test_truth_idx.append(str(data_info[idx][1]) + ',' + str(data_info[idx][2]))
    # 修改元数据，测试测试数据所处位置的得分置为0
    users_foods.iloc[data_info[idx][1], data_info[idx][2]] = 0
# 构造测试集数据label
test_truth_df = pd.DataFrame({'test_truth_idx': test_truth_idx, 'test_truth_rate': test_truth_rate})

# 包含训练数据、测试数据、待预测数据
users_foods = mat(users_foods)

# 训练与评估
total_mae, total_mse, total_rmse = [], [], []
for user_id in range(len(users)):
    _, test_result_ = svd_recommend(user_item=users_foods, user_id=user_id, top_n=3, test_truth=test_truth_df)
    sub_mae = mae(target=test_result_['test_truth_rate'], prediction=test_result_['predict_truth_score'])
    sub_mse = mse(target=test_result_['test_truth_rate'], prediction=test_result_['predict_truth_score'])
    sub_rmse = rmse(target=test_result_['test_truth_rate'], prediction=test_result_['predict_truth_score'])
    total_mae.append(sub_mae)
    total_mse.append(sub_mse)
    total_rmse.append(sub_rmse)
    # 每个用户下的测试集误差
    # print('user_id:', user_id, '\t', 'sub_mae:', sub_mae, '\t', 'sub_mse:', sub_mse, '\t', 'sub_rmse:', sub_rmse, '\t')
    final_recommend_result = pd.merge(_, foods, left_on='item_no', right_on='food_no', how='left'). \
        drop(['item_no', 'food_no'], axis=1)

print('\n 基于SVD的协同过滤推荐算法模型误差如下：\n',
      'mae:', np.mean(total_rmse), '\n',
      'mse:', np.mean(total_mse), '\n',
      'rmse:', np.mean(total_rmse), '\n')
