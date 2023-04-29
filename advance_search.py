import numpy as np
import pandas as pd
from functools import partial
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from geneticalgorithm import geneticalgorithm as ga
from simanneal import Annealer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer




# 定义函数
def model_optimization(data, model, optimizer, params_of_model):

    # 将数据拆分为特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 定义流水线
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("sampler", SMOTEENN()),
            ("classifier", model),
        ]
    )

    # 使用k折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 算法
    # 遗传算法
    def ga_fitness_function(params, *args):
        pipeline.set_params(**params)
        scores = cross_val_score(pipeline, X, y, scoring='f1_macro', cv=kfold, n_jobs=-1)
        return -np.mean(scores)

    def ga_optimization(X, y, pipeline, kfold, params_of_model):
        ga_algorithm = ga(function=ga_fitness_function, dimension=len(params_of_model), variable_type='real',
                          variable_boundaries=params_of_model, function_timeout=10000,
                          algorithm_parameters={'max_num_iteration': 50, 'population_size': 100,
                                                'mutation_probability': 0.1, 'elit_ratio': 0.01,
                                                'crossover_probability': 0.5, 'parents_portion': 0.3,
                                                'crossover_type': 'uniform', 'max_iteration_without_improv': None})
        ga_algorithm.run()
        return ga_algorithm.best_variable


    # 模拟退火
    class SimulatedAnnealing(Annealer):

        def __init__(self, state, X, y, pipeline, kfold):
            self.X = X
            self.y = y
            self.pipeline = pipeline
            self.kfold = kfold
            super(SimulatedAnnealing, self).__init__(state)  # important!

        def move(self):
            # 变异操作
            pass

        def energy(self):
            # 计算适应度
            pass

    def sa_optimization(X, y, pipeline, kfold, params_of_model):
        init_state = [1] * len(params_of_model)
        sa = SimulatedAnnealing(init_state, X, y, pipeline, kfold)
        best_params, best_fitness = sa.anneal()

        return best_params

    if optimizer == "bayes":
        # 贝叶斯优化
        opt = BayesSearchCV(pipeline, params_of_model, scoring='f1_macro', cv=kfold, n_jobs=-1)
        opt.fit(X, y)
        return opt.best_params_, opt.best_score_

    elif optimizer == "ga":
        # 遗传算法优化
        best_params = ga_optimization(X, y, pipeline, kfold, params_of_model)
        pipeline.set_params(**best_params)
        best_score = -ga_fitness_function(best_params)
        return best_params, best_score


    elif optimizer == "sa":
        # 模拟退火优化
        best_params = sa_optimization(X, y, pipeline, kfold, params_of_model)
        pipeline.set_params(**best_params)
        best_score = -sa.energy()
        return best_params, best_score

    else:
        raise ValueError("Invalid optimizer specified.")

