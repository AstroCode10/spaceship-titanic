from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

#Outlier Remover
class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, up_percent=0.95, low_percent=0.05):
        self.up_percent = up_percent
        self.low_percent = low_percent

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=getattr(X, "columns", None))
        self.upper_bounds_ = X.quantile(self.up_percent)
        self.lower_bounds_ = X.quantile(self.low_percent)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=getattr(X, "columns", None)).copy()

        # Only clip columns that exist in X
        cols_to_use = [c for c in self.upper_bounds_.index if c in X.columns]
        X[cols_to_use] = X[cols_to_use].clip(
            lower=self.lower_bounds_[cols_to_use],
            upper=self.upper_bounds_[cols_to_use],
            axis=1
        )
        return X

#Log Transformer
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()

        if self.cols is not None:
            # Only use columns present in X
            cols_to_use = [c for c in self.cols if c in X.columns]
            X[cols_to_use] = np.log1p(X[cols_to_use])
        else:
            X[:] = np.log1p(X)

        return X