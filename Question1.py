import sqlite3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm, preprocessing

database_path = "./MarketBasket.db"


def train_xgboost(data_train, label_train, data_validation, label_validation):
    grid_search_parameter = {
        "objective": ["reg:logistic"]
        , "eval_metric": ["logloss"]
        , "learning_rate": [0.1, 0.05]
        , "min_child_weight": [8, 10, 12]
        , "gamma": [0.7]
        , "subsample": [0.5, 0.7, 0.9]
        , "colsample_bytree": [0.9]
        , "lambda": [10]
        , "n_estimators": [80]
        , "scale_pos_weight": [9, 10]
    }
    # grid_search_parameter = {
    #     "objective": ["reg:logistic"]
    #     , "eval_metric": ["logloss"]
    #     , "learning_rate": [0.1]
    #     , "min_child_weight": [8, 10]
    #     , "gamma": [0.7]
    #     , "subsample": [0.5]
    #     , "colsample_bytree": [0.9]
    #     , "lambda": [10]
    #     , "n_estimators": [80]
    #     , "scale_pos_weight": [9]
    # }
    # Init Grid Search
    search_classifier = xgb.XGBRegressor()
    def my_score_function(label, predictions):
        best_f1_score = 0
        for threshold in np.arange(0.1, 1.0, 0.1):
            buffer_predictions_label = (predictions > threshold).astype(int)
            buffer_score = f1_score(label, buffer_predictions_label)
            if buffer_score > best_f1_score:
                best_f1_score = buffer_score
        return best_f1_score
    my_f1_score = make_scorer(my_score_function)
    grid_search = GridSearchCV(search_classifier, grid_search_parameter, n_jobs=-1, cv=3, scoring=my_f1_score, )
    # Fit
    grid_search.fit(data_train, label_train)
    # optimal_parameter = {
    #     "objective": "reg:logistic"
    #     , "eval_metric": "logloss"
    #     , "learning_rate": 0.1
    #     , "max_depth": 6
    #     , "min_child_weight": 10
    #     , "gamma": 0.70
    #     , "subsample": 0.76
    #     , "colsample_bytree": 0.95
    #     , "alpha": 2e-05
    #     , "lambda": 10
    #     , "n_estimators": 80
    # }
    # use optimal parameter to create a new classifier
    optimal_parameter = grid_search.best_params_
    print("Optimal xgboost parameters : ", optimal_parameter)
    xgb_classifier = xgb.XGBRegressor(**optimal_parameter)
    # train
    evaluation = [(data_train, label_train)]
    xgb_classifier.fit(data_train, label_train, eval_set=evaluation, verbose=10)
    best_f1_score = 0
    best_threshold = 0
    # d_validation = xgb.DMatrix(data_validation, label_validation)
    predictions = (xgb_classifier.predict(data_validation) > 0.0).astype(int)
    for threshold in np.arange(0.1, 1.0, 0.1):
        buffer_predictions = (xgb_classifier.predict(data_validation) > threshold).astype(int)
        buffer_score = f1_score(label_validation, buffer_predictions)
        if buffer_score > best_f1_score:
            predictions = buffer_predictions
            best_threshold = threshold
            best_f1_score = buffer_score
    print("Accuracy of xgboost : ", accuracy_score(label_validation, predictions))
    print("Precision of xgboost : ", precision_score(label_validation, predictions))
    print("Recall of xgboost : ", recall_score(label_validation, predictions))
    print("F1_score of xgboost : ", f1_score(label_validation, predictions))
    print("best threshold : \n", best_threshold)
    return [xgb_classifier, best_threshold]


def xgboost_submit(xgb_classifier_list, df_test, path="./xgboost_submission.csv"):
    xgb_classifier = xgb_classifier_list[0]
    best_threshold = xgb_classifier_list[1]
    predictions = (xgb_classifier.predict(df_test.drop(["user_id", "product_id", "order_id"], axis=1).to_numpy()) > best_threshold).astype(int)
    df_submit = df_test[["order_id", "user_id", "product_id"]].copy()
    df_submit["predictions"] = predictions
    submit_csv(df_submit, path)


def train_lda(data_train, label_train, data_validation, label_validation):
    lda = LDA(n_components=1)
    lda.fit(data_train, label_train)
    predictions = lda.predict(data_validation)
    print("Accuracy of LDA : ", accuracy_score(label_validation, predictions))
    print("Precision of LDA : ", precision_score(label_validation, predictions))
    print("Recall of LDA : ", recall_score(label_validation, predictions))
    print("F1 score of LDA : ", f1_score(label_validation, predictions))
    return lda


def _help_submit(x):
    x["products"] = ' '.join(str(i) for i in x["product_id"][x["predictions"] == 1])
    return x.iloc[0]


def submit_csv(df_test, path="./submission.csv"):
    df_submit = df_test.groupby("user_id").apply(lambda x : _help_submit(x))
    df_submit = df_submit[["order_id", "products"]].sort_values(["order_id"])
    df_submit.to_csv(path, index=None)


def lda_submit(lda, df_test, path="./lda_submission.csv"):
    normalized_test = preprocessing.normalize(df_test.drop(["user_id", "product_id", "order_id"], axis=1).to_numpy(), axis=1)
    predictions = lda.predict(normalized_test)
    df_submit = df_test[["order_id", "user_id", "product_id"]].copy()
    df_submit["predictions"] = predictions
    submit_csv(df_submit, path)


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

    # use local buffer to speed up
    df_orders_test_path = "./df_orders_test.pickle"
    if os.path.isfile(df_orders_test_path) and use_local_buffer:
        df_orders_test = pd.read_pickle(df_orders_test_path)
    else:
        sql = '''
                            SELECT order_id, user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order FROM orders 
                            WHERE eval_set='test' ORDER BY user_id, order_id;
                                '''
        df_orders_test = pd.read_sql(sql, connection)
        if use_local_buffer:
            df_orders_test.to_pickle(df_orders_test_path)

    print("Finish loading data")
    # fill nan
    df_orders_train.fillna(0, inplace=True)
    df_history_order_products.fillna(0, inplace=True)
    df_user_products_train.fillna(0, inplace=True)
    df_orders_test.fillna(0, inplace=True)

    df_features_path = "./df_features.pickle"
    if os.path.isfile(df_features_path) and use_local_buffer:
        df_features = pd.read_pickle(df_features_path)
    else:
        df_features = pd.DataFrame()
        """
        start extract features
        """
        # There are many features that we are going to extract respectively and are shown below
        # The product-level features
        # The amount of purchase times of a specific product
        df_product_all_purchase_number = df_history_order_products.groupby(["product_id"]).size().reset_index(
            name='product_all_purchase_number').sort_values(by="product_id")

        # validate the extracted features match the real data
        # print(df_history_order_products.loc[df_history_order_products['product_id'] == 23])

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
        df_buffer = df_history_order_products.groupby(["user_id", "product_id"]).head(1).reset_index()
        df_features["user_id"] = df_buffer["user_id"].values
        df_features["product_id"] = df_buffer["product_id"].values
        df_features["eval_set"] = df_buffer["eval_set"].values
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
        df_features = pd.merge(df_features, df_user_avg_time_between_order, how="left", on=["user_id"])
        df_features = pd.merge(df_features, df_user_all_products, how="left", on=["user_id"])
        df_features = pd.merge(df_features, df_user_all_categories, how="left", on=["user_id"])
        df_features = pd.merge(df_features, df_user_order_avg_product_number, how="left", on=["user_id"])
        if use_local_buffer:
            df_features.to_pickle(df_features_path)



    # split feature dataframe into train and test

    df_features_train = df_features.loc[df_features['eval_set'] == "train"]
    df_features_test = df_features.loc[df_features['eval_set'] == "test"].drop(["eval_set"], axis=1)

    df_features_test = pd.merge(df_features_test, df_orders_test, how="left", on=["user_id"])
    df_features_train = pd.merge(df_features_train, df_orders_train, how="left", on=["user_id"])

    # add label to train features
    df_features_train = pd.merge(df_features_train, df_user_products_train[["user_id", "product_id", "reordered"]], how="left",
                           on=["user_id", "product_id"])
    df_features_train.fillna(0, inplace=True)
    df_features_train.rename(columns={"reordered": "label"}, inplace=True)
    print("Finish extracting features")
    # split train and validation set
    x_train, x_validation, y_train, y_validation = train_test_split(
        df_features_train.drop(["user_id", "product_id", 'label', "eval_set"], axis=1), df_features_train["label"], train_size=0.9,
        shuffle=True)
    # do pca
    normalized_x_train = preprocessing.normalize(x_train, axis=1)
    normalized_x_validation = preprocessing.normalize(x_validation, axis=1)
    # pca = PCA(n_components=normalized_x_train.shape[1])
    # pca.fit(normalized_x_train)
    # with np.printoptions(precision=2, suppress=True):
    #     print(pca.explained_variance_ratio_*100)
    # # find components with total ratio >= 0.95
    # cumulated_ratio = 0
    # selected_component_number = 0
    # for i in range(len(pca.explained_variance_ratio_)):
    #     cumulated_ratio += pca.explained_variance_ratio_[i]
    #     if cumulated_ratio >= 0.99:
    #         selected_component_number = i + 1
    #         break
    # # plot figures:
    # buffer_transformed_x = pca.transform(normalized_x_train)
    # plt.scatter(buffer_transformed_x[y_train == 0, 0], buffer_transformed_x[y_train == 0, 1], c="Blue", alpha=0.3)
    # plt.scatter(buffer_transformed_x[y_train == 1, 0], buffer_transformed_x[y_train == 1, 1], c="Red", alpha=0.3)
    # plt.legend(["not repurchased", "repurchased"], loc="lower right")
    # plt.show()
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(buffer_transformed_x[y_train == 0, 0], buffer_transformed_x[y_train == 0, 1], buffer_transformed_x[y_train == 0, 2], c="Blue", alpha=0.3)
    # ax.scatter3D(buffer_transformed_x[y_train == 1, 0], buffer_transformed_x[y_train == 1, 1], buffer_transformed_x[y_train == 1, 2], c="Red", alpha=0.3)
    # plt.legend(["not repurchased", "repurchased"], loc="lower right")
    # plt.show()
    # # select the components
    # pca = PCA(n_components=selected_component_number)
    # pca.fit(normalized_x_train)
    # with np.printoptions(precision=2, suppress=True):
    #     print(pca.explained_variance_ratio_*100)
    # selected_normalized_x_train = pca.transform(normalized_x_train)
    # selected_normalized_x_validation = pca.transform(normalized_x_validation)

    xgb_classifier_list = train_xgboost(x_train, y_train, x_validation, y_validation)
    xgboost_submit(xgb_classifier_list, df_features_test)
    #lda = train_lda(normalized_x_train, y_train, normalized_x_validation, y_validation)
    #lda_submit(lda, df_features_test)



if __name__ == "__main__":
    test()
