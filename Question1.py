import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm, preprocessing

database_path = "./MarketBasket.db"


def test():
    connection = sqlite3.connect(database_path)
    use_local_buffer = True
    # use local buffer to speed up
    df_history_order_products_path = "./df_history_order_products.pickle"
    if os.path.isfile(df_history_order_products_path) and use_local_buffer:
        df_history_order_products = pd.read_pickle(df_history_order_products_path)
    else:
        sql = '''
                            SELECT * FROM history_order_products ORDER BY user_id, order_id;
                            '''
        df_history_order_products = pd.read_sql(sql, connection)
        if use_local_buffer:
            df_history_order_products.to_pickle(df_history_order_products_path)

    # use local buffer to speed up
    df_user_products_train_path = "./df_user_products_train.pickle"
    if os.path.isfile(df_user_products_train_path) and use_local_buffer:
        df_user_products_train = pd.read_pickle(df_user_products_train_path)
    else:
        sql = '''
                            SELECT * FROM user_products_train ORDER BY user_id, order_id;
                            '''
        df_user_products_train = pd.read_sql(sql, connection)
        if use_local_buffer:
            df_user_products_train.to_pickle(df_user_products_train_path)

    # use local buffer to speed up
    df_orders_train_path = "./df_orders_train.pickle"
    if os.path.isfile(df_orders_train_path) and use_local_buffer:
        df_orders_train = pd.read_pickle(df_orders_train_path)
    else:
        sql = '''
                            SELECT user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order FROM orders 
                            WHERE eval_set='train' ORDER BY user_id, order_id;
                                '''
        df_orders_train = pd.read_sql(sql, connection)
        if use_local_buffer:
            df_orders_train.to_pickle(df_orders_train_path)

    # fill nan
    df_orders_train.fillna(0, inplace=True)
    df_history_order_products.fillna(0, inplace=True)
    df_user_products_train.fillna(0, inplace=True)

    df_features = pd.DataFrame()
    """
    start extract features
    """
    # There are many features that we are going to extract respectively and are shown below
    # The amount of purchase times of a specific product
    df_product_all_purchase_number = df_history_order_products.groupby(["product_id"]).size().reset_index(
        name='product_all_purchase_number').sort_values(by="product_id")

    # validate the extracted features match the real data
    print(df_history_order_products.loc[df_history_order_products['product_id'] == 23])

    # a product was being purchased by how many customers
    df_product_all_user_number = df_history_order_products.groupby(
        ["product_id", "user_id"]).size().reset_index().groupby("product_id").size().reset_index(
        name="product_all_user_number").sort_values(by="product_id")

    # The customer-level features
    # The amount of history orders of a specific customer
    df_user_order_number = df_history_order_products.groupby(["user_id", "order_id"]).size().reset_index().groupby(
        "user_id").size().reset_index(name="user_order_number")

    # The interval time from last order of a customer
    df_user_days_since_prior_order = pd.DataFrame()

    # The average time between the two most recent orders of a customer
    df_user_avg_time_between_order = \
        df_history_order_products.groupby(["user_id", "order_id"])[
            "days_since_prior_order"].mean().reset_index().groupby(
            "user_id")["days_since_prior_order"].mean().reset_index(name="user_avg_time_between_order")

    # The amount of different products a customer has purchased
    df_user_all_products = df_history_order_products.groupby("user_id")["product_id"].size().reset_index(
        name="user_all_products")

    # The amount of different departments a customer has purchased
    df_user_all_categories = df_history_order_products.groupby("user_id")["product_id"].nunique().reset_index(
        name="user_all_categories")

    # The average amount of products one order a customer
    df_user_order_avg_product_number = \
        df_history_order_products.groupby(["user_id", "order_id"]).size().reset_index().groupby("user_id")[
            0].mean().reset_index(name="user_order_avg_product_number")

    # The customer-product-level features
    # The amount of a specific product that a customer has purchased
    df_user_product_number = df_history_order_products.groupby(["user_id", "product_id"]).size().reset_index(
        name="user_product_number")

    # The average purchase time during the day of history orders of a customer(The time in the day when purchased)
    df_user_product_avg_hour_of_day = df_history_order_products.groupby(["user_id", "product_id"])[
        "order_hour_of_day"].mean().reset_index(name="user_product_avg_hour_of_day")

    # The average purchase time during the day of history orders of a customer(The day in the week)
    df_user_product_avg_dow = df_history_order_products.groupby(["user_id", "product_id"])[
        "order_dow"].mean().reset_index(name="user_product_avg_dow")

    # The mean of add_to_cart order of a customer
    df_user_product_avg_add_to_chart_order = df_history_order_products.groupby(["user_id", "product_id"])[
        "add_to_cart_order"].mean().reset_index(name="user_product_avg_add_to_chart_order")

    # generate the feature dataframe
    df_features["user_id"] = df_user_product_number["user_id"].values
    df_features["product_id"] = df_user_product_number["product_id"].values
    df_features = pd.merge(df_features, df_user_products_train[["user_id", "product_id", "reordered"]], how="left",
                           on=["user_id", "product_id"])
    df_features.fillna(0, inplace=True)
    df_features.rename(columns={"reordered": "label"}, inplace=True)
    # df_features.sort_values(by=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_product_all_purchase_number, how="left", on=["product_id"])
    df_features = pd.merge(df_features, df_product_all_user_number, how="left", on=["product_id"])
    df_features = pd.merge(df_features, df_user_product_number, how="left", on=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_user_product_avg_hour_of_day, how="left", on=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_user_product_avg_dow, how="left", on=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_user_product_avg_add_to_chart_order, how="left",
                           on=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_user_order_number, how="left", on=["user_id"])
    # df_features = pd.merge(df_features, df_user_days_since_prior_order, how="left", on=["user_id", "product_id"])
    # df_features = pd.merge(df_features, df_user_product_number_divide_user_order_number, how="left", on=["user_id", "product_id"])
    # df_features = pd.merge(df_features, df_user_avg_time_between_order, how="left", on=["user_id", "product_id"])
    df_features = pd.merge(df_features, df_user_all_products, how="left", on=["user_id"])
    df_features = pd.merge(df_features, df_user_all_categories, how="left", on=["user_id"])
    df_features = pd.merge(df_features, df_user_order_avg_product_number, how="left", on=["user_id"])
    df_features = pd.merge(df_features, df_orders_train, how="left", on=["user_id"])

    # split train and validation set
    x_train, x_validation, y_train, y_validation = train_test_split(
        df_features.drop(["user_id", "product_id", 'label'], axis=1), df_features["label"], train_size=0.9,
        shuffle=True)

    n_feature = x_train.to_numpy()[:9]  # select the front 9 component to compare with the validation set
    # and decides whether those components are efficient enough to the model
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(n_feature)

    pca = PCA(n_components=9)
    pca.fit(X_minMax)
    print(pca.singular_values_)

    x_feature = x_validation.to_numpy()
    min_max_scaler = preprocessing.MinMaxScaler()
    x_minMax = min_max_scaler.fit_transform(x_feature)

    pca = PCA(n_components=14)
    pca.fit(x_minMax)

    # print(pca.explained_variance_ratio_)
    print(pca.singular_values_)

    # xgboost classifier
    d_train = xgb.DMatrix(x_train, y_train)
    xgb_params = {  # the parameters in the xgboost classifier and we could modify and fine-tune those to fit our model
        "objective": "reg:logistic"
        , "learning_rate": 0.1
        , "eval_metric": "logloss"
        , "eta": 0.1
        , "max_depth": 6
        , "min_child_weight": 10
        , "gamma": 0.70
        , "subsample": 0.76
        , "colsample_bytree": 0.95
        , "alpha": 2e-05
        , "lambda": 10
    }
    watchlist = [(d_train, "train")]
    bst = xgb.train(params=xgb_params, dtrain=d_train, num_boost_round=80, evals=watchlist, verbose_eval=10)
    xgb.plot_importance(bst)
    d_validation = xgb.DMatrix(x_validation)
    predictions = (bst.predict(d_validation) > 0.21).astype(int)
    # print("Accuracy of xgboost : ", accuracy_score(y_validation, predictions))
    # print("Precision of xgboost : ", precision_score(y_validation, predictions))
    # print("Recall of xgboost : ", recall_score(y_validation, predictions))
    # print("F1_score of xgboost : ", f1_score(y_validation, predictions))

    # LDA classifier
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train)
    predictions = lda.predict(x_validation)
    # print("Accuracy of LDA : ", accuracy_score(y_validation, predictions))
    # print("Precision of LDA : ", precision_score(y_validation, predictions))
    # print("Recall of LDA : ", recall_score(y_validation, predictions))
    # print("F1 score of LDA : ", f1_score(y_validation, predictions))

    # SVM is too slow
    # SVM classifier with linear kernel
    # clf = svm.LinearSVC()
    # clf.fit(x_train, y_train)
    # predictions = clf.predict(x_validation)
    # print("Accuracy of SVM with linear kernel : ", accuracy_score(y_validation, predictions))

    # From the results of LDA and XGBoost classifier, we could see that the accuracy of LDA is a little
    # higher than the XGBoost, we assume that LDA is better suited for low-dimensional, small scale
    # classification tasks while XGBoost is better suited for high-dimensional. large scale classification
    # tasks. Therefore, the dimensionality, features and scale in our problem are relatively low like only
    # 2 dimensions at most and about 10 features which may results in the better performance in LDA classifier


if __name__ == "__main__":
    test()
