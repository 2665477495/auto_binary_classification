import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from skopt.space import Real, Categorical, Integer
from advance_search import model_optimization
from auto_balance_score import k_score

# 生成虚拟数据集
data = pd.read_csv('data/data_cleaned.csv')

# 定义模型
model = RandomForestClassifier()

# 定义优化器
optimizer = "bayes"

# 定义模型参数
params_of_model = {
    "classifier__n_estimators": Integer(10, 200),
    "classifier__max_depth": Integer(1, 50),
    "classifier__min_samples_split": Integer(2, 20),
    "classifier__min_samples_leaf": Integer(1, 20),
    "classifier__max_features": Categorical(["sqrt", "log2"]),
}

# 调用函数
best_params, best_score = model_optimization(data, model, optimizer, params_of_model)

# 输出最佳参数和得分
print("Best parameters:", best_params)
print("Best score:", best_score)

# params_of_model = best_params
# # 调用函数
# mean_score, std_score = k_score(data,  params_of_model)
#
# # 输出平均得分和标准差
# print("Mean score:", mean_score)
# print("Standard deviation:", std_score)


