import numpy as np
import pandas as pd

from auto_balance_params_search import svm_grid_search
from auto_balance_score import k_score

"""参数确定与交叉验证_在SMOTEENN()平衡数据的基础上"""

# 加载数据
data = pd.read_csv("data/data_cleaned.csv")

# 定义SVM模型的参数
params_of_model = {
    "classifier__kernel": ["rbf", "linear", "poly", "sigmoid"],
    "classifier__C": list(np.arange(1, 10, 0.1)),
    "classifier__gamma": [0.01, 0.1, 1, 10],
}

# 调用函数
grid_search = svm_grid_search(data, params_of_model)

# 输出最佳参数和得分
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


# 定义SVM模型和参数
params_of_model = grid_search.best_params_

# 调用函数
mean_score, std_score = k_score(data,  params_of_model)

# 输出平均得分和标准差
print("Mean score:", mean_score)
print("Standard deviation:", std_score)
