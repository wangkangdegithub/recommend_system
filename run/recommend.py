"""
@author:
@file: recommend.py
@time: 2020/4/20 10:05
@desc: 推荐系统主函数
"""
import os
import sys
import random
import pandas as pd
from numpy import mat

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithm.pkg import load_data, svd_recommend, sat_recommend


def main(single_step=True):
    """
    推荐系统主函数
    :param single_step: 推荐方式标志位，为真：单点交互执行；为假：批量执行
    :return: 推荐结果
    """
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

    '''执行推荐'''
    if single_step is True:
        while True:
            user_name = input('推荐学生姓名:')
            try:
                user_id = users.loc[users['user_name'] == user_name, 'user_no'].values[0]
            except:
                print('无此学生，请重新输入~')
                continue
            predict_result, _ = svd_recommend(user_item=users_foods, user_id=user_id, top_n=3, test_truth=test_truth_df)
            if len(predict_result) > 0:
                final_recommend_result = pd.merge(predict_result, foods, left_on='item_no', right_on='food_no', how='left'). \
                    drop(['item_no', 'food_no'], axis=1)
                print('给 {} 同学的推荐结果：'.format(user_name))
                print(final_recommend_result, '\n')
            else:
                recommend_item_idx = sat_recommend(user_item=original_users_foods, user_id=user_id, top_n=3)
                final_recommend_result = foods[foods['food_no'].isin(recommend_item_idx)]['food_name'].tolist()
                print('给 {} 同学的推荐结果：'.format(user_name))
                print(final_recommend_result, '\n')
    else:
        for user_id in range(len(users)):
            user_name = users.loc[users['user_no'] == user_id, 'user_name'].values[0]
            predict_result, _ = svd_recommend(user_item=users_foods, user_id=user_id, top_n=3, test_truth=test_truth_df)
            if len(predict_result) > 0:
                final_recommend_result = pd.merge(predict_result, foods, left_on='item_no', right_on='food_no', how='left'). \
                    drop(['item_no', 'food_no'], axis=1)
                print('给 {} 同学的推荐结果：'.format(user_name))
                print(final_recommend_result, '\n')
            else:
                recommend_item_idx = sat_recommend(user_item=original_users_foods, user_id=user_id, top_n=3)
                final_recommend_result = foods[foods['food_no'].isin(recommend_item_idx)]['food_name'].tolist()
                print('给 {} 同学的推荐结果：'.format(user_name))
                print(final_recommend_result, '\n')


if __name__ == '__main__':
    print('启动推荐系统...')
    main()
