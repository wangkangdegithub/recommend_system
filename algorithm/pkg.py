"""
@author:
@file: pkg.py
@time: 2020/4/19 17:51
@desc: 工具函数
"""

import os
import numpy as np
import pandas as pd
from math import sqrt
from algorithm.svd import svd_rating


def mae(target, prediction):
    """
    平均绝对误差
    :return:mae
    """
    mae = np.mean(np.absolute(target - prediction))
    return mae


def mse(target, prediction):
    """
    均方误差
    :return:mse
    """
    mse = np.sum(target - prediction) ** 2 / len(target)

    return mse


def rmse(target, prediction):
    """
    均方根误差
    :return:rmse
    """
    rmse = sqrt(mse(target, prediction))
    return rmse


def load_data():
    """
    加载训练数据
    :return: user_item评分矩阵，用户id表、物品id表
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    data = pd.read_excel(os.path.join(base_path, 'data/用户给菜品打分表.xlsx')).reset_index(drop=True)
    foods = pd.DataFrame(data.iloc[0, 1::]).reset_index(drop=True)
    foods.rename(columns={0: 'food_name'}, inplace=True)
    foods['food_no'] = foods.index
    users = pd.DataFrame(data.iloc[1::, 0]).reset_index(drop=True)
    users.columns = ['user_name']
    users['user_no'] = users.index
    users_foods = data.iloc[1::, 1::].reset_index(drop=True).fillna(0)

    return users_foods, foods, users


def svd_recommend(user_item, user_id, top_n, test_truth):
    """
    推荐系统主函数
    :param user_item: 训练数据集
    :param user_id: 用户编号
    :param top_n: 推荐产品个数
    :return: top_n个推荐结果
    """
    predict_score_df = pd.DataFrame()
    unrated_item_idx = np.nonzero(user_item[user_id, :] == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unrated_item_idx) == 0:
        return '已对所有物品进行评分'
    # 物品的编号和评分值
    item_idx_list, item_scores_list = [], []
    for item_idx in unrated_item_idx:
        predict_score = svd_rating(user_item, user_id, item_idx)
        # 与测试集真实值进行比较
        predict_score_df = predict_score_df.append(pd.DataFrame({'predict_truth_idx': [str(user_id) + ',' + str(item_idx)],
                                                                 'predict_truth_score': [predict_score]}))
        item_scores_list.append(predict_score)
        item_idx_list.append(item_idx)
    # 推荐评分预测算法在测试集上的预测值与真实值对比
    test_contrasted_result = pd.merge(test_truth, predict_score_df, how='inner', left_on='test_truth_idx', right_on='predict_truth_idx')
    current_user_test_item_no = [int(no.split(',')[-1]) for no in test_contrasted_result['test_truth_idx']]
    # 排除掉测试集数据，即为对用户实际未打分产品的分值预测结果
    # 对未打分物品的预测结果,包括测试集与真正未打分数据
    rate_result = pd.DataFrame({'item_no': item_idx_list, 'predict_score': item_scores_list})
    # 除测试集外，真正未打分物品的分值
    final_rate_result = rate_result[~rate_result['item_no'].isin(current_user_test_item_no)]
    recommended_result = final_rate_result.sort_values(by='predict_score', ascending=False).reset_index(drop=True).head(top_n)
    return recommended_result, test_contrasted_result


def sat_recommend(user_item, user_id, top_n):
    """
    对于有所有评分的用户，按已有分值top-n推荐
    :param user_item:user_item评分矩阵
    :param user_id:待推荐物品的用户id
    :param top_n:推荐物品数量
    :return:推荐物品的物品id
    """
    nums = user_item.iloc[user_id, :].tolist()
    recommend_item_idx = []
    Inf = -999
    for i in range(top_n):
        recommend_item_idx.append(nums.index(max(nums)))
        nums[nums.index(max(nums))] = Inf

    return recommend_item_idx
