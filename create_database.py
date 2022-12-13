import sqlite3
import pandas as pd
import os

database_path = "./MarketBasket.db"
# create a folder called data and the data folder should be in the parent directory of create_database.py
# copy all the csv files into the data folder and then run create_database.py


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
# create or connect the database
connection = sqlite3.connect(database_path)
# create tables
# if the table exists, replace it
aisles.to_sql(name="aisles", con=connection, if_exists="replace", index=False)
departments.to_sql(name="departments", con=connection, if_exists="replace", index=False)
order_products_prior.to_sql(name="order_products_prior", con=connection, if_exists="replace", index=False)
order_products_train.to_sql(name="order_products_train", con=connection, if_exists="replace", index=False)
orders.to_sql(name="orders", con=connection, if_exists="replace", index=False)
products.to_sql(name="products", con=connection, if_exists="replace", index=False)


# create fact tables
curser = connection.cursor()
# drop fact table f_products if exists
sql = '''
        DROP TABLE IF EXISTS f_products
'''
curser.execute(sql)
# create fact table f_products
sql = '''
        CREATE TABLE IF NOT EXISTS f_products AS
            SELECT product_id, product_name, a.aisle, d.department FROM products 
                LEFT JOIN aisles a ON products.aisle_id = a.aisle_id 
                LEFT JOIN departments d ON products.department_id = d.department_id 
                    ORDER BY product_id
'''
curser.execute(sql)
# drop fact table f_order_products_prior if exists
sql = '''
        DROP TABLE IF EXISTS f_order_products_prior
'''
curser.execute(sql)
# create fact table f_order_products_prior
sql = '''
        CREATE TABLE IF NOT EXISTS f_order_products_prior AS
            SELECT orders.order_id AS order_id, user_id, eval_set, order_dow, order_hour_of_day, 
            days_since_prior_order, reordered, product_name, aisle, department FROM orders
                LEFT JOIN order_products_prior opp on orders.order_id = opp.order_id
                LEFT JOIN f_products fp ON fp.product_id = opp.product_id
                    ORDER BY order_id
    '''
curser.execute(sql)
# drop fact table f_order_products_train if exists
sql = '''
        DROP TABLE IF EXISTS f_order_products_train
'''
curser.execute(sql)
# create fact table f_order_products_train
sql = '''
            CREATE TABLE IF NOT EXISTS f_order_products_train AS
                SELECT orders.order_id AS order_id, user_id, eval_set, order_dow, order_hour_of_day, 
                days_since_prior_order, reordered, product_name, aisle, department FROM orders
                    LEFT JOIN order_products_train opt on orders.order_id = opt.order_id
                    LEFT JOIN f_products fp ON fp.product_id = opt.product_id
                        ORDER BY order_id
        '''
curser.execute(sql)
# drop table history_order_products if exists
sql = '''
        DROP TABLE IF EXISTS history_order_products 
'''
curser.execute(sql)
# create table history_order_products
sql = '''
            CREATE TABLE IF NOT EXISTS history_order_products AS
                SELECT a.order_id as order_id, user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order, product_id, add_to_cart_order, reordered FROM 
                (SELECT a.user_id as user_id, order_id, order_number, order_dow, order_hour_of_day, days_since_prior_order, a.eval_set as eval_set  FROM (SELECT user_id, eval_set FROM orders where eval_set='train' or eval_set='test') a
                LEFT JOIN (SELECT * FROM orders where eval_set='prior') b
                on a.user_id = b.user_id) a
                LEFT JOIN order_products_prior opp
                on a.order_id = opp.order_id;
        '''
curser.execute(sql)
# drop table user_products_train if exists
sql = '''
        DROP TABLE IF EXISTS user_products_train
'''
curser.execute(sql)
# create table user_products_train
sql = '''
            CREATE TABLE IF NOT EXISTS user_products_train AS
                SELECT o.order_id as order_id, user_id, order_number, order_dow, order_hour_of_day, days_since_prior_order, product_id, add_to_cart_order, reordered from orders o
                left join order_products_train opt
                on o.order_id = opt.order_id
                where eval_set='train' and reordered=1;
        '''
curser.execute(sql)
# create indexes on the table : f_order_products_prior
sql = '''
            CREATE INDEX f_order_products_prior_order_hour_of_day_eval_set ON f_order_products_prior (order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_prior_department_eval_set ON f_order_products_prior (department, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_prior_aisle_eval_set ON f_order_products_prior (aisle, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_prior_aisle_order_hour_of_day_eval_set ON f_order_products_prior (aisle, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_prior_department_order_hour_of_day_eval_set ON f_order_products_prior (department, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_prior_product_name_order_hour_of_day_eval_set ON f_order_products_prior (product_name, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
# create fact table f_order_products_train
sql = '''
            CREATE TABLE IF NOT EXISTS f_order_products_train AS
                SELECT orders.order_id AS order_id, user_id, eval_set, order_dow, order_hour_of_day, 
                days_since_prior_order, reordered, product_name, aisle, department FROM orders
                    LEFT JOIN order_products_train opt on orders.order_id = opt.order_id
                    LEFT JOIN f_products fp ON fp.product_id = opt.product_id
                        ORDER BY order_id
        '''
curser.execute(sql)
# create indexes on the table : f_order_products_train
sql = '''
            CREATE INDEX f_order_products_train_order_hour_of_day_eval_set ON f_order_products_train (order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_train_department_eval_set ON f_order_products_train (department, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_train_aisle_eval_set ON f_order_products_train (aisle, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_train_aisle_order_hour_of_day_eval_set ON f_order_products_train (aisle, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_train_department_order_hour_of_day_eval_set ON f_order_products_train (department, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
sql = '''
            CREATE INDEX f_order_products_train_product_name_order_hour_of_day_eval_set ON f_order_products_train (product_name, order_hour_of_day, eval_set);
        '''
curser.execute(sql)
# refresh the database to use optimal plan
sql = '''
            ANALYZE;
        '''
curser.execute(sql)
# close connection
connection.close()


