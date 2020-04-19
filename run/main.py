import os
import numpy as np
import pandas as pd
from numpy import nonzero, mat
from algorithm.svd import svd_rating


def recommend(user_item, user_id, top_n):
    """
    推荐系统主函数
    :param user_item: 训练数据集
    :param user_id: 用户编号
    :param top_n: 推荐产品个数
    :return: top_n个推荐结果
    """
    unrated_item_idx = nonzero(user_item[user_id, :] == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unrated_item_idx) == 0:
        return '已对所有物品进行评分'
    # 物品的编号和评分值
    item_idx_list, item_scores_list = [], []
    for item_idx in unrated_item_idx:
        predict_score = svd_rating(user_item, user_id, item_idx)
        item_scores_list.append(predict_score)
        item_idx_list.append(item_idx)
    recommend_result = pd.DataFrame({'item_no': item_idx_list, 'predict_score': item_scores_list})
    recommend_result = recommend_result.sort_values(by='predict_score', ascending=False).reset_index(drop=True)
    return recommend_result


base_path = os.path.dirname(os.path.dirname(__file__))
data = pd.read_excel(os.path.join(base_path, 'data/用户给菜品打分表.xlsx')).reset_index(drop=True)
foods = pd.DataFrame(data.iloc[0, 1::]).reset_index(drop=True)
foods.rename(columns={0: 'food_name'}, inplace=True)
foods['food_no'] = foods.index
users = pd.DataFrame(data.iloc[1::, 0]).reset_index(drop=True)
users.rename(columns={0: 'user_name'}, inplace=True)
users['users_no'] = users.index
users_foods = data.iloc[1::, 1::].reset_index(drop=True).fillna(0)
users_foods = mat(users_foods)

recommend_result = recommend(user_item=users_foods, user_id=0, top_n=5)
recommend_result = pd.merge(recommend_result, foods, left_on='item_no', right_on='food_no', how='left'). \
    drop(['item_no', 'food_no'], axis=1)
print(recommend_result)
