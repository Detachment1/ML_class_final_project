import pandas as pd
import sqlite3

# test code
database_path = "./MarketBasket.db"
connection = sqlite3.connect(database_path)
sql = '''
        select * from orders
'''
df = pd.read_sql(sql, connection)

if __name__ == "__main__":
    print("write code here")
