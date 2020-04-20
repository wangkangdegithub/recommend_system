"""
@author:
@file: run.py
@time: 2020/4/18 13:26
@desc: SVD算法
"""

from numpy import mat, eye
from numpy import linalg as la


def cos_sim(victor_a, victor_b):
    """
    计算向量余弦相似度
    :param victor_a:
    :param victor_b:
    :return: 归一化0~1
    """
    num = float(victor_a * victor_b.T)
    denom = la.norm(victor_a) * la.norm(victor_b)
    # 归一化0~1
    return 0.5 + 0.5 * (num / denom)


def choice_length(sigma, loop_num):
    """
    选择sigma向量长度,通常保留矩阵90 % 的能量，就可以得到重要的特征并取出噪声。
    :param sigma: sigma的值
    :param loop_num: 循环次数，默认为20次
    :return:最优sigma向量长度
    """
    # 总方差的集合（总能量值）
    sigma_square = sigma ** 2
    sigma_sum = sum(sigma_square)
    for sigma_length in range(1, loop_num + 1):
        sigma_square_sum = sum(sigma_square[:sigma_length])
        if sigma_square_sum / sigma_sum > 0.9:
            # print('sigma最优向量长度：', sigma_length)
            return sigma_length
        else:
            continue


def svd_rating(user_item, user_id, item_idx):
    """
    基于SVD的评分估计,对给定用户给定物品构建了一个评分估计值
    :param user_item:
    :param user_id:
    :param item_idx: 待打分的item_idx
    :return: 预测得分
    """
    # 奇异值分解
    u, sigma, vt = la.svd(user_item)
    # 选择最优的sigma向量长度
    sigma_length = choice_length(sigma=sigma, loop_num=len(user_item))
    # 构造产品矩阵向量
    # 方法一
    # item_victors = mat(pd.DataFrame(vt.T).iloc[:, :sigma_length])
    # 方法二
    diagonal_sigma = mat(eye(sigma_length) * sigma[: sigma_length])
    diagonal_sigma_inverse = diagonal_sigma.I
    item_victors = user_item.T * u[:, :sigma_length] * diagonal_sigma_inverse

    sim_total, rat_sim_total = 0, 0
    total_items = user_item.shape[1]

    # 基于已打分物品，求用户user_id 对所有未打分产品的预测得分
    for idx in range(total_items):
        user_rating = user_item[user_id, idx]
        if user_rating == 0 or idx == item_idx:
            continue
        # 求出用户未打分物品与已打分物品之间的相似度
        similarity = cos_sim(victor_a=item_victors[item_idx, :], victor_b=item_victors[idx, :])
        # 基于相似度及已打分值，对未打分物品进行分值预测
        rat_sim_total += similarity * user_rating
        # 对相似度不断累加求和
        sim_total += similarity
    if sim_total == 0:
        return 0
    else:
        return float(rat_sim_total / sim_total)
