import pandas as pd
import joblib
from catboost import Pool
from sklearn import set_config


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


#Main code
set_config(transform_output="pandas")

def main():
    test_df = pd.read_csv("test.csv")
    test_copy = test_df.copy()

    artifact = joblib.load("spaceship_model.joblib")
    model = artifact["model"]
    num_pipeline = artifact["num_pipeline"]
    cat_pipeline = artifact["cat_pipeline"]
    cols = artifact["feature_columns"]
    cat_indices = artifact["cat_indices"]

    test_copy = parse_cabin(test_copy)
    test_copy = add_group_sizes(test_copy)
    test_copy = test_copy.drop(columns=["PassengerId", "Cabin", "Name", "GroupId"])

    num_cols = test_copy.select_dtypes(include=["int64", "float64"])
    cat_cols = test_copy.select_dtypes(include=["category", "object"])

    test_copy[num_cols.columns] = num_pipeline.transform(test_copy[num_cols.columns])
    test_copy[cat_cols.columns] = cat_pipeline.transform(test_copy[cat_cols.columns])

    test_copy = test_copy[cols]

    test_pool = Pool(test_copy, cat_features=cat_indices)
    preds = model.predict(test_pool)

    submission = pd.DataFrame({
        "PassengerId": pd.read_csv("test.csv")["PassengerId"],
        "Transported": preds
    })


    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
