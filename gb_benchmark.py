import datetime
from contextlib import contextmanager

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn import metrics
import numpy as np
import gc
import json
import time
import warnings
from typing import List

from data_source.heart_disease import data_heart_disease
from data_source.home_credit import data_home_credit
from data_source.csc_hw1_spring19 import data_csc_hw1_spring19

warnings.simplefilter(action='ignore', category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    delta = time.time() - t0
    print(f"{title} - done in {str(datetime.timedelta(seconds=delta))}")
    return delta


def calc_stat(array):
    return {'mean': array.mean(), 'std': array.std()}


class ClassifierWrapper:
    def __init__(self, title, params):
        self.model = None
        self.title = title
        self.params = {} if params is None else params

    def get_title(self):
        return self.title

    def preprocessed(self, X, y):
        return X, y

    def create(self):
        pass

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        pass

    def clear(self):
        # del self.model
        self.model = None
        gc.collect()


def auc_clf(X_orig, y_orig, classifier: ClassifierWrapper, num_folds=10, verbose=False):
    if verbose:
        print(f"Benchmark AUC classifier for {classifier.get_title()}")
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    learn_time_a = np.empty([num_folds])
    predict_time_a = np.empty([num_folds])
    auc_a = np.empty([num_folds])
    X, y = classifier.preprocessed(X_orig, y_orig)
    for n_fold, (train_index, test_index) in enumerate(folds.split(X, y)):
        if verbose:
            print(f"Fold ({n_fold}/{num_folds})")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        classifier.create()
        start_learn = time.time()
        classifier.fit(X_train, y_train)
        end_learn = time.time()
        start_predict = time.time()
        preds_class = classifier.predict_proba(X_test)
        end_predict = time.time()
        auc = metrics.roc_auc_score(y_test, preds_class)
        learn_time_a[n_fold] = end_learn - start_learn
        predict_time_a[n_fold] = end_predict - start_predict
        auc_a[n_fold] = auc
        classifier.clear()
    return {'target': 'auc',
            'model': classifier.get_title(),
            'auc': calc_stat(auc_a),
            'learn_sec': calc_stat(learn_time_a),
            'predict_sec': calc_stat(predict_time_a),
            'dataset_rows': len(X),
            'dataset_columns': len(X.columns),
            'num_folds': num_folds
            }


def auc_clf_series(X, y, classifiers: List[ClassifierWrapper], num_folds=10, verbose=False):
    return [auc_clf(X, y, c, num_folds=num_folds, verbose=verbose) for c in classifiers]


class SklearnClassifier(ClassifierWrapper):
    def __init__(self, params=None):
        super().__init__('sklearn', params)
        self.X_mean = None

    def create(self):
        self.model = ensemble.GradientBoostingClassifier(**self.params)

    def preprocessed(self, X, y):
        if not np.isfinite(X).all().all():
            print("Copy data without NaN for sklearn")
            X2 = X.copy()
            X2 = X2.dropna(axis=1, how='all')
            X2.fillna(X2.mean(), inplace=True)
            return X2, y
        else:
            return X, y

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class LightgbmClassifier(ClassifierWrapper):
    def __init__(self, params=None):
        super().__init__('lightgbm', params)

    def create(self):
        self.params['eval_metric'] = 'auc'
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y, eval_metric='auc')

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class XgboostClassifier(ClassifierWrapper):
    def __init__(self, params=None):
        super().__init__('xgboost', params)

    def create(self):
        self.params['eval_metric'] = 'auc'
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y, eval_metric='auc')

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


class CatboostClassifier(ClassifierWrapper):
    def __init__(self, params=None):
        super().__init__('catboost', params)

    def create(self):
        self.params['eval_metric'] = 'AUC'
        self.model = cb.CatBoostClassifier(**self.params)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]


def make_classifiers(params=None):
    if params is not None:
        def strict_get(field):
            if field not in params:
                raise Exception(f"Field '{field}' not found in params.")
            else:
                return params[field]

        sklearn_p = strict_get('sklearn')
        lightgbm_p = strict_get('lightgbm')
        xgboost_p = strict_get('xgboost')
        catboost_p = strict_get('catboost')
    else:
        sklearn_p = lightgbm_p = xgboost_p = catboost_p = {}
    return [
        SklearnClassifier(params=sklearn_p),
        LightgbmClassifier(params=lightgbm_p),
        XgboostClassifier(params=xgboost_p),
        CatboostClassifier(params=catboost_p)
    ]


def main(debug=False):
    start_time_marker = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    result_file = f'gb_benchmark_result/gb_bench_{start_time_marker}.json'
    if debug:
        num_rows = 1000
        num_folds = 2
    else:
        num_rows = None
        num_folds = 10

    small_data_params_clf = {
        'sklearn': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        'lightgbm': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        'xgboost': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        'catboost': {'iterations': 100, 'max_depth': 4, 'verbose': False}
    }

    plan = [
        ("heart-disease", data_heart_disease, small_data_params_clf),
        ("home-credit", data_home_credit, small_data_params_clf),
        ("csc_hw1_spring19", data_csc_hw1_spring19, small_data_params_clf)
    ]

    bench_result = []
    for data_name, data_fun, params in plan:
        gc.collect()
        print(f"Load dataset:{data_name}")
        X, y = data_fun(num_rows)
        print(f"Make classifierss for {data_name}")
        classifiers = make_classifiers(params)
        print(f"Benchmark classifiers for {data_name}")
        clf_result = auc_clf_series(X, y, classifiers, num_folds=num_folds, verbose=True)
        bench_result.append({
            'dataset': data_name,
            'test': 'classification',
            'params': params,
            'results': clf_result
        })
        del X, y
        # Write temp file
        with open(result_file + "_.tmp", 'w') as outfile:
            json.dump(bench_result, outfile)

    if debug:
        print('Result:')
        for x in bench_result:
            print(f'dataset:{x["dataset"]}')
            for x in x['results']:
                print(x['model'], x['auc'], 'time:', x['learn_sec']['mean'], x['predict_sec']['mean'])

    # Final file
    with open(result_file, 'w') as outfile:
        json.dump(bench_result, outfile)


if __name__ == "__main__":
    with timer("Full benchmark run"):
        main(debug=True)
