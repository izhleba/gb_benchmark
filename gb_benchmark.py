import datetime
from contextlib import contextmanager
import argparse
import catboost as cb
import lightgbm as lgb
import sys
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import ensemble
from sklearn import metrics
import numpy as np
import gc
import json
import time
import warnings

from data_source.fraud_detection import data_fraud_detection
from data_source.heart_disease import data_heart_disease
from data_source.home_credit import data_home_credit
from data_source.csc_hw1_spring19 import data_csc_hw1_spring19
from data_source.home_credit_cat import data_home_credit_cat

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
    def __init__(self, params):
        self.model = None
        self.params = {} if params is None else params

    def preprocessed(self, X, y, cat_features):
        return X, y

    def create(self):
        pass

    def fit(self, X, y, cat_features):
        pass

    def predict_proba(self, X, cat_features):
        pass

    def clear(self):
        # del self.model
        self.model = None
        gc.collect()


def auc_clf(X_orig, y_orig, classifier: ClassifierWrapper, cat_features=[], num_folds=10, verbose=False):
    folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    learn_time_a = np.empty([num_folds])
    predict_time_a = np.empty([num_folds])
    auc_a = np.empty([num_folds])
    X, y = classifier.preprocessed(X_orig, y_orig, cat_features)
    try:
        for n_fold, (train_index, test_index) in enumerate(folds.split(X, y)):
            if verbose:
                print(f"Fold ({n_fold+1}/{num_folds})")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            classifier.create()
            start_learn = time.time()
            classifier.fit(X_train, y_train, cat_features)
            end_learn = time.time()
            start_predict = time.time()
            preds_class = classifier.predict_proba(X_test, cat_features)
            end_predict = time.time()
            auc = metrics.roc_auc_score(y_test, preds_class)
            learn_time_a[n_fold] = end_learn - start_learn
            predict_time_a[n_fold] = end_predict - start_predict
            auc_a[n_fold] = auc
            classifier.clear()
        return {'target': 'auc',
                'fail': False,
                'fail_msg': None,
                'auc': calc_stat(auc_a),
                'learn_sec': calc_stat(learn_time_a),
                'predict_sec': calc_stat(predict_time_a),
                'dataset_rows': len(X),
                'dataset_columns': len(X.columns),
                'num_folds': num_folds
                }
    except Exception as e:
        print("Warning!", str(e), sys.exc_info()[0])
        return {'target': 'auc',
                'fail': True,
                'fail_msg': str(e),
                'fail_msg2': str(sys.exc_info()[0]),
                'auc': None,
                'learn_sec': None,
                'predict_sec': None,
                'dataset_rows': len(X),
                'dataset_columns': len(X.columns),
                'num_folds': num_folds
                }


class SklearnClf(ClassifierWrapper):
    def __init__(self, params=None):
        super().__init__(params)
        self.X_mean = None

    def create(self):
        self.model = ensemble.GradientBoostingClassifier(**self.params)

    def preprocessed(self, X, y, cat_features):
        if not np.isfinite(X).all().all():
            print("Copy data without NaN for sklearn")
            X2 = X.copy()
            X2 = X2.replace([np.inf, -np.inf], np.nan)
            X2 = X2.dropna(axis=1, how='all')
            X2.fillna(X2.mean(), inplace=True)
            return X2, y
        else:
            return X, y

    def fit(self, X, y, cat_features):
        if len(cat_features) > 0:
            print("Sklearn no cat_features support")
        else:
            self.model.fit(X, y)

    def predict_proba(self, X, cat_features):
        return self.model.predict_proba(X)[:, 1]


class LightgbmClf(ClassifierWrapper):
    def create(self):
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X, y, cat_features):
        if len(cat_features) > 0:
            print(f"Lightgbm with cat features {cat_features}")
            self.model.fit(X, y, eval_metric='auc', categorical_feature=cat_features)
        else:
            self.model.fit(X, y, eval_metric='auc')

    def predict_proba(self, X, cat_features):
        return self.model.predict_proba(X)[:, 1]


class XgboostClf(ClassifierWrapper):
    def create(self):
        self.params['eval_metric'] = 'auc'
        self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X, y, cat_features):
        if len(cat_features) > 0:
            print("Sklearn no cat_features support")
        else:
            self.model.fit(X, y, eval_metric='auc')

    def predict_proba(self, X, cat_features):
        return self.model.predict_proba(X)[:, 1]


class CatboostClf(ClassifierWrapper):
    def create(self):
        self.params['eval_metric'] = 'AUC'
        self.model = cb.CatBoostClassifier(**self.params)

    def fit(self, X, y, cat_features):
        if len(cat_features) > 0:
            print(f"Catboost with cat features {cat_features}")
            self.model.fit(X, y, cat_features)
        else:
            self.model.fit(X, y)

    def predict_proba(self, X, cat_features):
        return self.model.predict_proba(X)[:, 1]


def main(debug=False):
    print(f"Debug: {debug}")
    start_time_marker = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    result_file = f'gb_benchmark_result/gb_bench_{start_time_marker}.json'
    if debug:
        num_rows = 100000
        num_folds = 2
    else:
        num_rows = None
        num_folds = 10

    clf_config = {
        'sklearn': (SklearnClf, [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05},
        ]),
        'lightgbm': (LightgbmClf, [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05},
            {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05},
        ]),
        'xgboost': (XgboostClf, [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'exact'},
            {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'exact'}
        ]),
        'xgb_hist': (XgboostClf, [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 1000, 'max_depth': 5, 'learning_rate': 0.05, 'tree_method': 'hist'},
            {'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.05, 'tree_method': 'hist'}
        ]),
        'catboost': (CatboostClf, [
            {'iterations': 100, 'max_depth': 3, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 7, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 3, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 7, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 3, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 5, 'learning_rate': 0.05, 'verbose': False},
            {'iterations': 100, 'max_depth': 7, 'learning_rate': 0.05, 'verbose': False}
        ])
    }

    plan = [
        # ("heart-disease", data_heart_disease, clf_config),
        ("home-credit", data_home_credit, clf_config),
        ("home-credit-cat", data_home_credit_cat, clf_config)
        # ("csc_hw1_spring19", data_csc_hw1_spring19, params_list_cls),
        # ("fraud_detection",data_fraud_detection,params_list_cls)
    ]

    bench_result = []
    for data_name, data_fun, config in plan:
        gc.collect()
        print(f"Load dataset:{data_name}")
        X, y, cat_features = data_fun(num_rows)
        config_items = config.items()
        for i, (clf_name, (clf_class, params_list)) in enumerate(config_items):
            for j, params in enumerate(params_list):
                print(f"Clf: {clf_name} {i+1}/{len(config_items)}. Params: {j+1}/{len(params_list)}")
                classifier = clf_class(params=params)
                clf_result = auc_clf(X, y, classifier, cat_features=cat_features, num_folds=num_folds, verbose=True)
                bench_result.append({
                    'dataset': data_name,
                    'test': 'classification',
                    'params': params,
                    'name': clf_name,
                    'result': clf_result
                })

            # Write temp file
            print("Savepoint")
            with open(result_file + "_.tmp", 'w+') as outfile:
                json.dump(bench_result, outfile)
        print("Delete X,y")
        del X, y

    # if debug:
    if True:
        print('Result:')
        for x in bench_result:
            result_ = x['result']
            if result_['fail'] == False:
                auc = "auc: %.4f ±%.4f" % (result_['auc']['mean'], result_['auc']['std'])
                learn = "learn: %.4f ±%.4f" % (result_['learn_sec']['mean'], result_['learn_sec']['std'])
                pred = "pred: %.4f ±%.4f" % (result_['predict_sec']['mean'], result_['predict_sec']['std'])
                print(x["dataset"], x['name'], auc, learn, pred, x['params'])

    # Final file
    with open(result_file, 'w') as outfile:
        json.dump(bench_result, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    with timer("Full benchmark run"):
        main(debug=args.debug)
