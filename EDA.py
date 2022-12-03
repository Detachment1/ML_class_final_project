import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go


database_path = "./MarketBasket.db"


if __name__ == "__main__":
    connection = sqlite3.connect(database_path)
    sql = '''
            SELECT COUNT(*) AS number, department FROM f_order_products_prior WHERE eval_set='prior'
        GROUP BY department ORDER BY number DESC
    '''

    df = pd.read_sql(sql, connection)
    # plt.plot(df["department"], df["number"])
    # plt.show()
    fig = px.bar(x=df["department"], y=df["number"], title="Purchase number over department",
                 labels={'x': 'Department', 'y': 'Purchase number'})
    fig.write_html('department_number.html', auto_open=True)

    sql = '''
             SELECT COUNT(*) AS number, aisle FROM f_order_products_prior WHERE eval_set='prior'
        GROUP BY aisle ORDER BY number DESC
        '''

    df = pd.read_sql(sql, connection)
    # plt.plot(df["aisle"], df["number"])
    # plt.show()
    fig = px.bar(x=df["aisle"], y=df["number"], title="Purchase number over aisle",
                 labels={'x': 'Aisle', 'y': 'Purchase number'})
    fig.write_html('aisle_number.html', auto_open=True)

    sql = '''
                 SELECT COUNT(*) AS number, order_hour_of_day FROM f_order_products_prior WHERE eval_set='prior'
            GROUP BY order_hour_of_day ORDER BY order_hour_of_day;
            '''

    df = pd.read_sql(sql, connection)
    # plt.plot(df["order_hour_of_day"], df["number"])
    # plt.show()
    fig = px.bar(x=df["order_hour_of_day"], y=df["number"], title="Purchase number over order hour of day",
                 labels={'x': 'Order hour of day', 'y': 'Purchase number'})
    fig.write_html('order_hour_of_day_number.html', auto_open=True)

    df_aisle_time_dict = {}
    for i in range(24):
        sql = '''
                         SELECT COUNT(*) AS number, aisle FROM f_order_products_prior WHERE eval_set='prior' and order_hour_of_day = {}
            GROUP BY aisle ORDER BY number DESC  LIMIT 5;
                    '''.format(i)
        df = pd.read_sql(sql, connection)
        for index, row in df.iterrows():
            a = row["aisle"]
            n = row["number"]
            if a not in df_aisle_time_dict:
                df_aisle_time_dict[a] = [0]*24
            else:
                df_aisle_time_dict[a][i] = n
    data_list = []
    keys = []
    colors = px.colors.qualitative.Plotly
    for key, value in df_aisle_time_dict.items():
        keys.append(key)
        data_list.append(value)
    data_array = np.array(data_list)
    sort_index = np.argsort(data_array, axis=0)
    figure_bar_list = []
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            figure_bar_list.append(go.Bar(name=keys[sort_index[i][j]],
                                          x=[j], y=[data_array[sort_index[i][j]][j]],
                                          marker=dict(color=colors[sort_index[i][j]]),
                                          legendgroup=keys[sort_index[i][j]],
                                          showlegend=j == 0
                                          ))
    fig = go.Figure(figure_bar_list)
    fig.update_layout(barmode='stack', title="Top 5 aisle over hour of day",
                      xaxis_title="Hour of day",
                      yaxis_title="Purchase numbers",
                      legend_title="Aisle Name")
    fig.write_html('order_hour_of_day_aisle_number.html', auto_open=True)

    df_department_time_dict = {}
    for i in range(24):
        sql = '''
                             SELECT COUNT(*) AS number, department FROM f_order_products_prior WHERE eval_set='prior' and order_hour_of_day = {}
                GROUP BY department ORDER BY number DESC  LIMIT 5;
                        '''.format(i)
        df = pd.read_sql(sql, connection)
        for index, row in df.iterrows():
            a = row["department"]
            n = row["number"]
            if a not in df_department_time_dict:
                df_department_time_dict[a] = [0] * 24
            else:
                df_department_time_dict[a][i] = n
    data_list = []
    keys = []
    colors = px.colors.qualitative.Plotly
    for key, value in df_department_time_dict.items():
        keys.append(key)
        data_list.append(value)
    data_array = np.array(data_list)
    sort_index = np.argsort(data_array, axis=0)
    figure_bar_list = []
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            figure_bar_list.append(go.Bar(name=keys[sort_index[i][j]],
                                          x=[j], y=[data_array[sort_index[i][j]][j]],
                                          marker=dict(color=colors[sort_index[i][j]]),
                                          legendgroup=keys[sort_index[i][j]],
                                          showlegend=j == 0
                                          ))
    fig = go.Figure(figure_bar_list)
    fig.update_layout(barmode='stack', title="Top 5 department over hour of day",
                      xaxis_title="Hour of day",
                      yaxis_title="Purchase numbers",
                      legend_title="Department Name")
    fig.write_html('order_hour_of_day_department_number.html', auto_open=True)

    df_product_name_time_dict = {}
    for i in range(24):
        sql = '''
                        SELECT COUNT(*) AS number, product_name FROM f_order_products_prior WHERE eval_set='prior' and order_hour_of_day = {}
                GROUP BY product_name ORDER BY number DESC  LIMIT 5;
                            '''.format(i)
        df = pd.read_sql(sql, connection)
        for index, row in df.iterrows():
            a = row["product_name"]
            n = row["number"]
            if a not in df_product_name_time_dict:
                df_product_name_time_dict[a] = [0] * 24
            else:
                df_product_name_time_dict[a][i] = n
    data_list = []
    keys = []
    colors = px.colors.qualitative.Plotly
    for key, value in df_product_name_time_dict.items():
        keys.append(key)
        data_list.append(value)
    data_array = np.array(data_list)
    sort_index = np.argsort(data_array, axis=0)
    figure_bar_list = []
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            figure_bar_list.append(go.Bar(name=keys[sort_index[i][j]],
                                          x=[j], y=[data_array[sort_index[i][j]][j]],
                                          marker=dict(color=colors[sort_index[i][j]]),
                                          legendgroup=keys[sort_index[i][j]],
                                          showlegend=j == 0
                                          ))
    fig = go.Figure(figure_bar_list)
    fig.update_layout(barmode='stack', title="Top 5 product name over hour of day",
                      xaxis_title="Hour of day",
                      yaxis_title="Purchase numbers",
                      legend_title="Product Name")
    fig.write_html('order_hour_of_day_product_name_number.html', auto_open=True)