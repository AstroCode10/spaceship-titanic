import pandas as pd
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from transformers import OutlierRemover, LogTransformer
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import shap

#Functions
#EDA Functions
def cat_value_counts(df, cols):
    for col in cols:
        print(df[col].value_counts().rename(f"{col}_count").to_frame())

def transported_rate_by_cat(df, cols):
    for col in cols:
        print(
            df.groupby(col)["Transported"]
            .mean()
            .sort_values(ascending=False)
            .to_frame(name=f"{col}_transported_rate")
        )

def missingness_vs_target(df, cols):
    for col in cols:
        print(
            df.assign(is_missing=df[col].isna())
            .groupby("is_missing")["Transported"]
            .mean()
            .to_frame(name="transported_rate")
        )


#General Functions
#Parsing Cabin
def parse_cabin(df):
    cabin = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["CabinSide"] = cabin[2]

    return df

#Extracting group
def add_group_sizes(df):
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("size")

    return df

#Making numeric transformer
def make_num_transformer(log_cols):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log_transform", LogTransformer(cols=log_cols)),
        ("outlier_rmv", OutlierRemover())
    ])

#Making categorical transformer
def make_cat_transformer():
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent"))
    ])


#Main code
def main():
    set_config(transform_output="pandas")

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    #Defining the unnecessary columns to drop
    DROP_COLS = ["PassengerId", "Cabin", "Name", "GroupId"]

    #Getting features and target column
    X = train_df.drop(columns=["Transported"])
    y = train_df["Transported"]

    X = parse_cabin(X)
    X = add_group_sizes(X)

    X = X.drop(columns=DROP_COLS)

    #Extracting column data types
    num_cols = X.select_dtypes(include=["int64", "float64"])
    cat_cols = X.select_dtypes(include=["category", "object"])

    #Explorative Data Analysis
    #cat_value_counts(X, cat_cols)
    #transported_rate_by_cat(pd.concat([X,y], axis=1), cat_cols)
    #missingness_vs_target(pd.concat([X,y], axis=1), cat_cols)

    #fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    #axes = axes.flatten()

    #for ax, col in zip(axes, num_cols):
    #    ax.hist(X[col].dropna(), bins=30)
    #    ax.set_title(f"Distribution of {col}")

    #plt.tight_layout()
    #plt.show()

    #df_corr = pd.concat(
    #    [X[num_cols], y.astype(int).rename("Transported")],
    #    axis=1
    #)
    #corr = df_corr.corr()

    #plt.figure(figsize=(6, 4))
    #plt.imshow(corr, cmap="coolwarm")
    #plt.colorbar()
    #plt.xticks(np.arange(len(num_cols)), num_cols, rotation=45)
    #plt.yticks(np.arange(len(num_cols)), num_cols)
    #plt.title("Correlation Matrix")
    #plt.show()


    #Preprocessing
    num_pipeline = make_num_transformer(["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"])
    X[num_cols.columns] = num_pipeline.fit_transform(X[num_cols.columns])

    cat_pipeline = make_cat_transformer()
    X[cat_cols.columns] = cat_pipeline.fit_transform(X[cat_cols.columns])

    cat_indices = [X.columns.get_loc(col) for col in cat_cols.columns.tolist()]

    #Nested CV with CatBoost
    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    outer_scores = []

    #for train_idx, val_idx in outer_cv.split(X, y):
    #    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    #    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    #    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    #    val_pool = Pool(X_val, y_val, cat_features=cat_indices)

    #    params = {
    #        "loss_function": "Logloss",
    #        "eval_metric": "AUC",
    #        "iterations": 500,
    #        "depth": 6,
    #        "learning_rate": 0.05,
    #        "l2_leaf_reg": 5,
    #        "random_seed": 42,
    #        "verbose": 100
    #   }

    #    model = CatBoostClassifier(**params)
    #    model.fit(train_pool)

    #    y_pred = model.predict_proba(val_pool)[:, 1]
    #    outer_scores.append(roc_auc_score(y_val, y_pred))


    #print(f"AUC per fold: {outer_scores}")
    #print(f"Mean AUC: {np.mean(outer_scores)}")
    #print(f"Std AUC: {np.std(outer_scores)}")

    #Locking in final model
    final_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1500,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5,
        "random_seed": 42,
        "verbose": 100
    }

    final_model = CatBoostClassifier(**final_params)
    full_pool = Pool(X, y, cat_features=cat_indices)
    final_model.fit(full_pool)


    #Implementing joblib to serialise the model
    artifact = {
        "model": final_model,
        "num_pipeline": num_pipeline,
        "cat_pipeline": cat_pipeline,
        "feature_columns": X.columns.tolist(),
        "cat_indices": cat_indices
    }

    joblib.dump(artifact, "spaceship_model.joblib")

if "__main__" == __name__:
    main()
