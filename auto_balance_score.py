import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def remove_prefix_from_params(params, prefix="classifier__"):
    return {key.replace(prefix, ""): value for key, value in params.items()}


def k_score(data, model):
    # 拆分
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 流水线
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("sampler", SMOTEENN()),
            ("classifier", model),
        ]
    )

    # 使用k折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold = 1
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 使用 SMOTEENN 平衡训练数据
        X_train_resampled, y_train_resampled = SMOTEENN().fit_resample(X_train, y_train)

        # 计算每个标签的数据占比
        counter = Counter(y_train_resampled)
        total_samples = sum(counter.values())
        label_ratios = {label: count / total_samples for label, count in counter.items()}

        print(f"Fold {fold}: Label ratios: {label_ratios}")

        # 在流水线中使用平衡后的训练数据
        pipeline.fit(X_train_resampled, y_train_resampled)
        fold += 1

    # 使用交叉验证拟合数据
    scores = cross_val_score(pipeline, X, y, scoring='f1_macro', cv=kfold, n_jobs=-1)

    return np.mean(scores), np.std(scores)
