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
from sklearn import svm

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
    # 商品对应的特征
    # 某一个商品被购买的次数
    df_product_all_purchase_number = df_history_order_products.groupby(["product_id"]).size().reset_index(
        name='product_all_purchase_number').sort_values(by="product_id")
    # 某一个商品被多少个人购买过
    df_product_all_user_number = df_history_order_products.groupby(
        ["product_id", "user_id"]).size().reset_index().groupby("product_id").size().reset_index(
        name="product_all_user_number").sort_values(by="product_id")

    # 顾客对应的特征
    # 某一个顾客历史订单数量
    df_user_order_number = df_history_order_products.groupby(["user_id", "order_id"]).size().reset_index().groupby(
        "user_id").size().reset_index(name="user_order_number")
    # 某一个顾客上次订单相隔的时间
    df_user_days_since_prior_order = pd.DataFrame()
    # 某一个顾客相邻两次历史订单平均时间
    df_user_avg_time_between_order = \
    df_history_order_products.groupby(["user_id", "order_id"])["days_since_prior_order"].mean().reset_index().groupby(
        "user_id")["days_since_prior_order"].mean().reset_index(name="user_avg_time_between_order")
    # 某一个顾客所有买过的商品数
    df_user_all_products = df_history_order_products.groupby("user_id")["product_id"].size().reset_index(
        name="user_all_products")
    # 某一个顾客所有买过的商品种类数
    df_user_all_categories = df_history_order_products.groupby("user_id")["product_id"].nunique().reset_index(
        name="user_all_categories")
    # 某一个顾客平均一个订单的商品数
    df_user_order_avg_product_number = \
    df_history_order_products.groupby(["user_id", "order_id"]).size().reset_index().groupby("user_id")[
        0].mean().reset_index(name="user_order_avg_product_number")

    # 顾客和商品对应的特征
    # 某一个顾客购买某一个商品的次数
    df_user_product_number = df_history_order_products.groupby(["user_id", "product_id"]).size().reset_index(
        name="user_product_number")
    # 某一个顾客购买某一个商品的历史平均购买时间（购买时在一天的小时）
    df_user_product_avg_hour_of_day = df_history_order_products.groupby(["user_id", "product_id"])[
        "order_hour_of_day"].mean().reset_index(name="user_product_avg_hour_of_day")
    # 某一个顾客购买某一个商品的历史平均购买时间（购买时周几）
    df_user_product_avg_dow = df_history_order_products.groupby(["user_id", "product_id"])[
        "order_dow"].mean().reset_index(name="user_product_avg_dow")
    # 某一个顾客购买某一个商品的加入购物车顺序的平均数
    df_user_product_avg_add_to_chart_order = df_history_order_products.groupby(["user_id", "product_id"])[
        "add_to_cart_order"].mean().reset_index(name="user_product_avg_add_to_chart_order")

    # 生成 feature dataframe
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

    # xgboost classifier
    d_train = xgb.DMatrix(x_train, y_train)
    xgb_params = {
        "objective": "reg:logistic"
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
    print("Accuracy of xgboost : ", accuracy_score(y_validation, predictions))
    print("Precision of xgboost : ", precision_score(y_validation, predictions))
    print("Recall of xgboost : ", recall_score(y_validation, predictions))
    print("F1_score of xgboost : ", f1_score(y_validation, predictions))

    # LDA classifier
    lda = LDA(n_components=1)
    lda.fit(x_train, y_train)
    predictions = lda.predict(x_validation)
    print("Accuracy of LDA : ", accuracy_score(y_validation, predictions))
    print("Precision of LDA : ", precision_score(y_validation, predictions))
    print("Recall of LDA : ", recall_score(y_validation, predictions))
    print("F1 score of LDA : ", f1_score(y_validation, predictions))

    # SVM is too slow
    # SVM classifier with linear kernel
    # clf = svm.LinearSVC()
    # clf.fit(x_train, y_train)
    # predictions = clf.predict(x_validation)
    # print("Accuracy of SVM with linear kernel : ", accuracy_score(y_validation, predictions))


if __name__ == "__main__":
    test()



