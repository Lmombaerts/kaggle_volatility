
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

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

def stupidForestPrediction(book_path_train,prediction_column_name,train_targets_pd,book_path_test):
    naive_predictions_train = past_realized_volatility_per_stock(list_file=book_path_train,prediction_column_name=prediction_column_name)
    df_joined_train = train_targets_pd.merge(naive_predictions_train[['row_id','pred']], on = ['row_id'], how = 'left')
    
    X = np.array(df_joined_train['pred']).reshape(-1,1)
    y = np.array(df_joined_train['target']).reshape(-1,)

    regr = RandomForestRegressor(random_state=0)
    regr.fit(X, y)
    
    naive_predictions_test = past_realized_volatility_per_stock(list_file=book_path_test,prediction_column_name='target')
    yhat = regr.predict(np.array(naive_predictions_test['target']).reshape(-1,1))
    
    updated_predictions = naive_predictions_test.copy()
    updated_predictions['target'] = yhat
    
    return updated_predictions
