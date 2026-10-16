"""Preprocessing transformers and pipeline factory."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer,
                                   PowerTransformer, OneHotEncoder, FunctionTransformer)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class SignedLogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(X) * np.log1p(np.abs(X))
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip extreme values to fitted quantiles to make downstream transforms stable."""
    def __init__(self, lower=0.5, upper=99.5):
        self.lower = lower
        self.upper = upper
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper, axis=0)
        # avoid zero-width
        self.upper_ = np.where(self.upper_ <= self.lower_, self.lower_ + 1.0, self.upper_)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)

def make_preprocessor(numeric_cols, categorical_cols, transform_kind):
    num_steps = [("imputer", SimpleImputer(strategy="median")), ("clip", Winsorizer())]
    if transform_kind == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif transform_kind == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif transform_kind == "log_standard":
        num_steps.append(("log", SignedLogTransformer()))
        num_steps.append(("scaler", StandardScaler()))
    elif transform_kind == "quantile":
        num_steps.append(("scaler", QuantileTransformer(output_distribution="normal", random_state=42)))
    elif transform_kind == "power":
        num_steps.append(("scaler", PowerTransformer(method="yeo-johnson", standardize=True)))
    else:
        raise ValueError(transform_kind)

    transformers = [("num", Pipeline(num_steps), numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_model(model_name, class_weight=True):
    if model_name == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if class_weight else None, n_jobs=-1)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=150, class_weight="balanced" if class_weight else None,
                                     random_state=42, n_jobs=-1)
    if model_name == "xgboost":
        return XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=19, random_state=42,
                             eval_metric="logloss", n_jobs=-1, verbosity=0)
    if model_name == "lightgbm":
        return LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced" if class_weight else None,
                              random_state=42, n_jobs=-1, verbose=-1)
    raise ValueError(model_name)


TRANSFORMS = ["standard", "robust", "log_standard", "quantile", "power"]
MODELS = ["logistic", "random_forest", "xgboost", "lightgbm"]
