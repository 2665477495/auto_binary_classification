from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm_grid_search(data, params_of_model):
    # 将数据拆分为特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 定义流水线
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("sampler", SMOTEENN()),
            ("classifier", SVC()),
        ]
    )

    # 使用k折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 定义GridSearchCV
    grid_search = GridSearchCV(pipeline, params_of_model, scoring='f1_macro', cv=kfold, n_jobs=-1)

    # 使用GridSearchCV拟合数据
    grid_search.fit(X, y)

    return grid_search


