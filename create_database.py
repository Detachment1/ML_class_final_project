import sqlite3
import pandas as pd
import os

database_path = "./MarketBasket.db"
# create a folder called data and the data folder should be in the parent directory of create_database.py
# copy all the csv files into the data folder and then run create_database.py
if not os.path.exists(database_path):
    # read data from csv files
    aisles = pd.read_csv(
        r"../data/aisles.csv")
    departments = pd.read_csv(
        r"../data/departments.csv")
    order_products_prior = pd.read_csv(
        r"../data/order_products__prior.csv")
    order_products_train = pd.read_csv(
        r"../data/order_products__train.csv")
    orders = pd.read_csv(
        r"../data/orders.csv")
    products = pd.read_csv(
        r"../data/products.csv")
    # create the database
    connection = sqlite3.connect(database_path)
    # create tables
    # if the table exists, replace it
    aisles.to_sql(name="aisles", con=connection, if_exists="replace", index=False)
    departments.to_sql(name="departments", con=connection, if_exists="replace", index=False)
    order_products_prior.to_sql(name="order_products_prior", con=connection, if_exists="replace", index=False)
    order_products_train.to_sql(name="order_products_train", con=connection, if_exists="replace", index=False)
    orders.to_sql(name="orders", con=connection, if_exists="replace", index=False)
    products.to_sql(name="products", con=connection, if_exists="replace", index=False)
    # close connection
    connection.close()


