
import pandas as pd
import numpy as np

def log_return(list_stock_prices): # Stock prices are estimated through wap values
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def compute_wap(book_pd):
    wap = (book_pd['bid_price1'] * book_pd['ask_size1'] + book_pd['ask_price1'] * book_pd['bid_size1']) / (book_pd['bid_size1']+ book_pd['ask_size1'])
    return wap

def realized_volatility_per_time_id(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    
    # Estimate stock price per time point
    df_book_data['wap'] = compute_wap(df_book_data)
    
    # Compute log return from wap values per time_id
    df_book_data['log_return'] = df_book_data.groupby(['time_id'])['wap'].apply(log_return)
    df_book_data = df_book_data[~df_book_data['log_return'].isnull()]
    
    # Compute the square root of the sum of log return squared to get realized volatility
    df_realized_vol_per_stock =  pd.DataFrame(df_book_data.groupby(['time_id'])['log_return'].agg(realized_volatility)).reset_index()
    
    # Formatting
    df_realized_vol_per_stock = df_realized_vol_per_stock.rename(columns = {'log_return':prediction_column_name})
    stock_id = file_path.split('=')[1]
    df_realized_vol_per_stock['row_id'] = df_realized_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')
    
    return df_realized_vol_per_stock[['row_id',prediction_column_name]]

def past_realized_volatility_per_stock(list_file,prediction_column_name):
    df_past_realized = pd.DataFrame()
    for file in list_file:
        df_past_realized = pd.concat([df_past_realized,
                                     realized_volatility_per_time_id(file,prediction_column_name)])
    return df_past_realized
