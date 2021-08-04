import pandas as pd
import numpy as np
import math
import os

from scipy.interpolate import interp1d
import time 

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from information_measures import *
from joblib import Parallel, delayed

#from arch import arch_model

def rmspe(y_true, y_pred):
    return  (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))

def log_return(list_stock_prices): # Stock prices are estimated through wap values
    return np.log(list_stock_prices).diff() 

def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return**2))

def compute_wap(book_pd):
    wap = (book_pd['bid_price1'] * book_pd['ask_size1'] + book_pd['ask_price1'] * book_pd['bid_size1']) / (book_pd['bid_size1']+ book_pd['ask_size1'])
    return wap

def realized_volatility_from_book_pd(book_stock_time):
    
    wap = compute_wap(book_stock_time)
    returns = log_return(wap)
    volatility = realized_volatility(returns)
    
    return volatility

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

def garch_fit_predict_volatility(returns_series, N=10000):
    model = arch_model(returns_series * N, p=1, q=1)
    model_fit = model.fit(update_freq=0, disp='off')
    yhat = model_fit.forecast(horizon=600, reindex=False)

    pred_volatility = np.sqrt(np.sum(yhat.variance.values)) / N

    return pred_volatility

def garch_volatility_per_time_id(file_path, prediction_column_name):
    # read the data
    df_book_data = pd.read_parquet(file_path) 

    # calculate the midprice (not the WAP)  
    df_book_data['midprice'] =(df_book_data['bid_price1'] + df_book_data['ask_price1'])/2

    # leave only WAP for now
    df_book_data = df_book_data[['time_id', 'seconds_in_bucket', 'midprice']]
    df_book_data = df_book_data.sort_values('seconds_in_bucket')

    # make the book updates evenly spaced
    df_book_data_evenly = pd.DataFrame({'time_id':np.repeat(df_book_data['time_id'].unique(), 600), 
                                        'second':np.tile(range(0,600), df_book_data['time_id'].nunique())})
    df_book_data_evenly['second'] = df_book_data_evenly['second'].astype(np.int16)
    df_book_data_evenly = df_book_data_evenly.sort_values('second')


    df_book_data_evenly = pd.merge_asof(df_book_data_evenly,
                           df_book_data,
                           left_on='second',right_on='seconds_in_bucket',
                           by = 'time_id')

    # Ordering for easier use
    df_book_data_evenly = df_book_data_evenly[['time_id', 'second', 'midprice']]
    df_book_data_evenly = df_book_data_evenly.sort_values(['time_id','second']).reset_index(drop=True)


    # calculate log returns 
    df_book_data_evenly['log_return'] = df_book_data_evenly.groupby(['time_id'])['midprice'].apply(log_return)
    df_book_data_evenly = df_book_data_evenly[~df_book_data_evenly['log_return'].isnull()]


    # fit GARCH(1, 1) and predict the volatility of returns
    df_garch_vol_per_stock =  \
        pd.DataFrame(df_book_data_evenly.groupby(['time_id'])['log_return'].agg(garch_fit_predict_volatility)).reset_index()
    df_garch_vol_per_stock = df_garch_vol_per_stock.rename(columns = {'log_return':prediction_column_name})

    # add row_id column to the data
    stock_id = file_path.split('=')[1]
    df_garch_vol_per_stock['row_id'] = df_garch_vol_per_stock['time_id'].apply(lambda x:f'{stock_id}-{x}')

    # return the result
    return df_garch_vol_per_stock[['row_id', prediction_column_name]]

def garch_volatility_per_stock(list_file, prediction_column_name):
    df_garch_predicted = pd.DataFrame()
    for file in list_file:
        df_garch_predicted = pd.concat([df_garch_predicted,
                                     garch_volatility_per_time_id(file, prediction_column_name)])
    return df_garch_predicted

def entropy_from_book(book_stock_time,last_min):
    
    if last_min < 10:
        book_stock_time = book_stock_time[book_stock_time['seconds_in_bucket'] >= (600-last_min*60)]
        if book_stock_time.empty == True or book_stock_time.shape[0] < 3:
            return 0
        
    wap = compute_wap(book_stock_time)
    t_init = book_stock_time['seconds_in_bucket']
    t_new = np.arange(np.min(t_init),np.max(t_init)) 
    
    # Closest neighbour interpolation (no changes in wap between lines)
    nearest = interp1d(t_init, wap, kind='nearest')
    resampled_wap = nearest(t_new)
    
    # Compute sample entropy
    # sampleEntropy = nolds.sampen(resampled_wap)
    sampleEntropy = sampen(resampled_wap)
    
    return sampleEntropy

def entropy_from_wap(wap,seconds,last_seconds):
    
    if last_seconds < 600:
        idx = np.where(seconds >= last_seconds)[0]
        if len(idx) < 3:
            return 0
        else:
            wap = wap[idx]
            seconds = seconds[idx]
    
    # Closest neighbour interpolation (no changes in wap between lines)
    t_new = np.arange(np.min(seconds),np.max(seconds))
    nearest = interp1d(seconds, wap, kind='nearest')
    resampled_wap = nearest(t_new)
    
    # Compute sample entropy
    # sampleEntropy = nolds.sampen(resampled_wap)
    sampleEntropy = sampen(resampled_wap)
    # sampleEntropy = ApEn_new(resampled_wap,3,0.001)
    
    return sampleEntropy

def linearFit(book_stock_time, last_min):
    
    if last_min < 10:
        book_stock_time = book_stock_time[book_stock_time['seconds_in_bucket'] >= (600-last_min*60)]
        if book_stock_time.empty == True or book_stock_time.shape[0] < 2:
            return 0
        
    wap = np.array(compute_wap(book_stock_time))
    t_init = book_stock_time['seconds_in_bucket']
    
    return (wap[-1] - wap[0])/(np.max(t_init) - np.min(t_init))

def wapStat(book_stock_time, last_min):
    
    if last_min < 10:
        book_stock_time = book_stock_time[book_stock_time['seconds_in_bucket'] >= (600-last_min*60)]
        if book_stock_time.empty == True or book_stock_time.shape[0] < 2:
            return 0
        
    wap = compute_wap(book_stock_time)
    t_init = book_stock_time['seconds_in_bucket']
    t_new = np.arange(np.min(t_init),np.max(t_init)) 
    
    # Closest neighbour interpolation (no changes in wap between lines)
    nearest = interp1d(t_init, wap, kind='nearest')
    resampled_wap = nearest(t_new)
    
    return np.std(resampled_wap)


def entropy_Prediction(book_path_train,prediction_column_name,train_targets_pd,book_path_test,all_stocks_ids,test_file):
    
    # Compute features
    book_features_encoded_test = computeFeatures_1(book_path_test,'test',test_file,all_stocks_ids) 
    
    book_features_encoded_train = computeFeatures_1(book_path_train,'train',train_targets_pd,all_stocks_ids)
    
    X = book_features_encoded_train.drop(['row_id','target','stock_id'],axis=1)
    y = book_features_encoded_train['target']
    
    # Modeling
    catboost_default = CatBoostRegressor(verbose=0)
    catboost_default.fit(X,y)
    
    # Predict
    X_test = book_features_encoded_test.drop(['row_id','stock_id'],axis=1)
    yhat = catboost_default.predict(X_test)
    
    # Formatting
    yhat_pd = pd.DataFrame(yhat,columns=['target'])
    predictions = pd.concat([test_file,yhat_pd],axis=1)
    
    return predictions


def computeFeatures_1(book_path,prediction_column_name,train_targets_pd,all_stocks_ids):
    
    book_all_features = pd.DataFrame()
    encoder = np.eye(len(all_stocks_ids))

    stocks_id_list, row_id_list = [], []
    volatility_list, entropy2_list = [], []
    linearFit_list, linearFit5_list, linearFit2_list = [], [], []
    wap_std_list, wap_std5_list, wap_std2_list = [], [], []

    for file in book_path:
        start = time.time()

        book_stock = pd.read_parquet(file)
        stock_id = file.split('=')[1]
        print('stock id computing = ' + str(stock_id))
        stock_time_ids = book_stock['time_id'].unique()
        for time_id in stock_time_ids:     

            # Access book data at this time + stock
            book_stock_time = book_stock[book_stock['time_id'] == time_id]

            # Create feature matrix
            stocks_id_list.append(stock_id)
            row_id_list.append(str(f'{stock_id}-{time_id}'))
            volatility_list.append(realized_volatility_from_book_pd(book_stock_time=book_stock_time))
            entropy2_list.append(entropy_from_book(book_stock_time=book_stock_time,last_min=2))
            linearFit_list.append(linearFit(book_stock_time=book_stock_time,last_min=10))
            linearFit5_list.append(linearFit(book_stock_time=book_stock_time,last_min=5))
            linearFit2_list.append(linearFit(book_stock_time=book_stock_time,last_min=2))
            wap_std_list.append(wapStat(book_stock_time=book_stock_time,last_min=10))
            wap_std5_list.append(wapStat(book_stock_time=book_stock_time,last_min=5))
            wap_std2_list.append(wapStat(book_stock_time=book_stock_time,last_min=2))

        print('Computing one stock entropy took', time.time() - start, 'seconds for stock ', stock_id)

    # Merge targets
    stocks_id_pd = pd.DataFrame(stocks_id_list,columns=['stock_id'])
    row_id_pd = pd.DataFrame(row_id_list,columns=['row_id'])
    volatility_pd = pd.DataFrame(volatility_list,columns=['volatility'])
    entropy2_pd = pd.DataFrame(entropy2_list,columns=['entropy2'])
    linearFit_pd = pd.DataFrame(linearFit_list,columns=['linearFit_coef'])
    linearFit5_pd = pd.DataFrame(linearFit5_list,columns=['linearFit_coef5'])
    linearFit2_pd = pd.DataFrame(linearFit2_list,columns=['linearFit_coef2'])
    wap_std_pd = pd.DataFrame(wap_std_list,columns=['wap_std'])
    wap_std5_pd = pd.DataFrame(wap_std5_list,columns=['wap_std5'])
    wap_std2_pd = pd.DataFrame(wap_std2_list,columns=['wap_std2'])

    book_all_features = pd.concat([stocks_id_pd,row_id_pd,volatility_pd,entropy2_pd,linearFit_pd,linearFit5_pd,linearFit2_pd,
                                  wap_std_pd,wap_std5_pd,wap_std2_pd],axis=1)

    # This line makes sure the predictions are aligned with the row_id in the submission file
    book_all_features = train_targets_pd.merge(book_all_features, on = ['row_id'])

    # Add encoded stock
    encoded = list()

    for i in range(book_all_features.shape[0]):
        stock_id = book_all_features['stock_id'][i]
        encoded_stock = encoder[np.where(all_stocks_ids == int(stock_id))[0],:]
        encoded.append(encoded_stock)

    encoded_pd = pd.DataFrame(np.array(encoded).reshape(book_all_features.shape[0],np.array(all_stocks_ids).shape[0]))
    book_all_features_encoded = pd.concat([book_all_features, encoded_pd],axis=1)
    
    return book_all_features_encoded

def calc_wap(df):
    return (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])

def calc_wap2(df):
    return (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])

def calc_wap3(df):
    return (df['bid_price2'] * df['bid_size2'] + df['ask_price2'] * df['ask_size2']) / (df['bid_size2'] + df['ask_size2'])

def calc_wap4(df):
    return (df['bid_price1'] * df['bid_size1'] + df['ask_price1'] * df['ask_size1']) / (df['bid_size1'] + df['ask_size1'])

def mid_price(df):
    return df['bid_price1'] /2 + df['ask_price1'] / 2

def calc_rv_from_wap_numba(values, index):
    log_return = np.diff(np.log(values))
    realized_vol = np.sqrt(np.sum(np.square(log_return[1:])))
    return realized_vol

def load_book_data_by_id(stock_id,datapath,train_test):
    file_to_read = os.path.join(datapath,'book_' + str(train_test) + str('.parquet'),'stock_id=' + str(stock_id))
    df = pd.read_parquet(file_to_read)
    return df

def load_trades_data_by_id(stock_id,datapath,train_test):
    file_to_read = os.path.join(datapath,'trade_' + str(train_test) + str('.parquet'),'stock_id=' + str(stock_id))
    df = pd.read_parquet(file_to_read)
    return df

def entropy_from_df(df):
    
    if df.shape[0] < 3:
        return 0
        
    t_init = df['seconds_in_bucket']
    t_new = np.arange(np.min(t_init),np.max(t_init)) 
    
    # Closest neighbour interpolation (no changes in wap between lines)
    nearest = interp1d(t_init, df['wap'], kind='nearest')
    resampled_wap = nearest(t_new)
    
    # Compute sample entropy
    # sampleEntropy = nolds.sampen(resampled_wap)
    sampleEntropy = sampen(resampled_wap)
    
    return sampleEntropy

def entropy_from_df2(df):
    
    if df.shape[0] < 3:
        return 0
        
    t_init = df['seconds_in_bucket']
    t_new = np.arange(np.min(t_init),np.max(t_init)) 
    
    # Closest neighbour interpolation (no changes in wap between lines)
    nearest = interp1d(t_init, df['wap2'], kind='nearest')
    resampled_wap = nearest(t_new)
    
    # Compute sample entropy
    # sampleEntropy = nolds.sampen(resampled_wap)
    sampleEntropy = sampen(resampled_wap)
    
    return sampleEntropy

def entropy_from_df3(df):
    
    if df.shape[0] < 3:
        return 0
        
    t_init = df['seconds_in_bucket']
    t_new = np.arange(np.min(t_init),np.max(t_init)) 
    
    # Closest neighbour interpolation (no changes in wap between lines)
    nearest = interp1d(t_init, df['wap3'], kind='nearest')
    resampled_wap = nearest(t_new)
    
    # Compute sample entropy
    sampleEntropy = sampen(resampled_wap)
    
    return sampleEntropy

def financial_metrics(df):
    
    wap_imbalance = np.mean(df['wap'] - df['wap2'])
    price_spread = np.mean((df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2))
    bid_spread = np.mean(df['bid_price1'] - df['bid_price2'])  
    ask_spread = np.mean(df['ask_price1'] - df['ask_price2']) # Abs to take
    total_volume = np.mean((df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2']))
    volume_imbalance = np.mean(abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2'])))
    
    return [wap_imbalance,price_spread,bid_spread,ask_spread,total_volume,volume_imbalance]

def financial_metrics_2(df):
    
    wap_imbalance = df['wap'] - df['wap2']
    price_spread = (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1'])/2)
    bid_spread = df['bid_price1'] - df['bid_price2']
    ask_spread = df['ask_price1'] - df['ask_price2'] # Abs to take
    total_volume = (df['ask_size1'] + df['ask_size2']) + (df['bid_size1'] + df['bid_size2'])
    volume_imbalance = abs((df['ask_size1'] + df['ask_size2']) - (df['bid_size1'] + df['bid_size2']))
    
    # New features here
    wap_imbalance_mean = np.mean(wap_imbalance)
    wap_imbalance_sum = np.sum(wap_imbalance)
    wap_imbalance_std = np.std(wap_imbalance)
    wap_imbalance_max = np.max(wap_imbalance)
    wap_imbalance_min = np.min(wap_imbalance)
    
    price_spread_mean = np.mean(price_spread)
    price_spread_sum = np.sum(price_spread)
    price_spread_std = np.std(price_spread)
    price_spread_max = np.max(price_spread)
    price_spread_min = np.min(price_spread)
    
    bid_spread_mean = np.mean(bid_spread)
    bid_spread_sum = np.sum(bid_spread)
    bid_spread_std = np.std(bid_spread)
    bid_spread_max = np.max(bid_spread)
    bid_spread_min = np.min(bid_spread)
    
    ask_spread_mean = np.mean(ask_spread)
    ask_spread_sum = np.sum(ask_spread)
    ask_spread_std = np.std(ask_spread)
    ask_spread_max = np.max(ask_spread)
    ask_spread_min = np.min(ask_spread)
    
    total_volume_mean = np.mean(total_volume)
    total_volume_sum = np.sum(total_volume)
    total_volume_std = np.std(total_volume)
    total_volume_max = np.max(total_volume)
    total_volume_min = np.min(total_volume)
    
    volume_imbalance_mean = np.mean(volume_imbalance)
    volume_imbalance_sum = np.sum(volume_imbalance)
    volume_imbalance_std = np.std(volume_imbalance)
    volume_imbalance_max = np.max(volume_imbalance)
    volume_imbalance_min = np.min(volume_imbalance)
    
    return [wap_imbalance_mean,price_spread_mean,bid_spread_mean,ask_spread_mean,total_volume_mean,volume_imbalance_mean, wap_imbalance_sum,price_spread_sum,bid_spread_sum,ask_spread_sum,total_volume_sum,volume_imbalance_sum, wap_imbalance_std,price_spread_std,bid_spread_std,ask_spread_std,total_volume_std,volume_imbalance_std, wap_imbalance_max,price_spread_max,bid_spread_max,ask_spread_max,total_volume_max,volume_imbalance_max, wap_imbalance_min,price_spread_min,bid_spread_min,ask_spread_min,total_volume_min,volume_imbalance_min]

def other_metrics(df):
    
    if df.shape[0] < 2:
        linearFit = 0
        linearFit2 = 0
        linearFit3 = 0
        std_1 = 0
        std_2 = 0
        std_3 = 0
    else:
        linearFit = (df['wap'].iloc[-1] - df['wap'].iloc[0]) / ((np.max(df['seconds_in_bucket']) - np.min(df['seconds_in_bucket']))) 
        linearFit2 = (df['wap2'].iloc[-1] - df['wap2'].iloc[0]) / ((np.max(df['seconds_in_bucket']) - np.min(df['seconds_in_bucket']))) 
        linearFit3 = (df['wap3'].iloc[-1] - df['wap3'].iloc[0]) / ((np.max(df['seconds_in_bucket']) - np.min(df['seconds_in_bucket']))) 
    
        # Resampling
        t_init = df['seconds_in_bucket']
        t_new = np.arange(np.min(t_init),np.max(t_init)) 

        # Closest neighbour interpolation (no changes in wap between lines)
        nearest = interp1d(t_init, df['wap'], kind='nearest')
        nearest2 = interp1d(t_init, df['wap2'], kind='nearest')
        nearest3 = interp1d(t_init, df['wap3'], kind='nearest')

        std_1 = np.std(nearest(t_new))
        std_2 = np.std(nearest2(t_new))
        std_3 = np.std(nearest3(t_new))
    
    return [linearFit, linearFit2, linearFit3, std_1, std_2, std_3]

def load_book_data_by_id_kaggle(stock_id,train_test):
    df = pd.read_parquet(f'../input/optiver-realized-volatility-prediction/book_{train_test}.parquet/stock_id={stock_id}')
    return df

def load_trades_data_by_id_kaggle(stock_id,train_test):
    df = pd.read_parquet(f'../input/optiver-realized-volatility-prediction/trade_{train_test}.parquet/stock_id={stock_id}')
    return df

def computeFeatures_wEntropy(machine, dataset, all_stocks_ids, datapath):
    
    list_rv, list_rv2, list_rv3 = [], [], []
    list_ent, list_fin, list_fin2 = [], [], []
    list_others, list_others2, list_others3 = [], [], []

    for stock_id in range(127):
        
        start = time.time()
        
        if machine == 'local':
            try:
                book_stock = load_book_data_by_id(stock_id,datapath,dataset)
            except:
                continue
        elif machine == 'kaggle':
            try:
                book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
            except:
                continue
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Calculate wap for the book
        book_stock['wap'] = calc_wap(book_stock)
        book_stock['wap2'] = calc_wap2(book_stock)
        book_stock['wap3'] = calc_wap3(book_stock)

        # Calculate realized volatility
        df_sub = book_stock.groupby('time_id')['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub2 = book_stock.groupby('time_id')['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub3 = book_stock.groupby('time_id')['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub['time_id']]
        df_sub = pd.concat([df_sub,df_sub2['wap2'],df_sub3['wap3']],axis=1)
        df_sub = df_sub.rename(columns={'time_id':'row_id','wap': 'rv', 'wap2': 'rv2', 'wap3': 'rv3'})
        
        # Calculate realized volatility last 5 min
        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_sub_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub_5['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_5['time_id']]
            df_sub_5 = pd.concat([df_sub_5,df_sub2_5['wap2'],df_sub3_5['wap3']],axis=1)
            df_sub_5 = df_sub_5.rename(columns={'time_id':'row_id','wap': 'rv_5', 'wap2': 'rv2_5', 'wap3': 'rv3_5'})
        else: # 0 volatility
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_5'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_5'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_5'])
            df_sub_5 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3],axis=1) 

        # Calculate realized volatility last 2 min
        isEmpty = book_stock.query(f'seconds_in_bucket >= 480').empty
        if isEmpty == False:
            df_sub_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()    
            df_sub_2['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_2['time_id']] 
            df_sub_2 = pd.concat([df_sub_2,df_sub2_2['wap2'],df_sub3_2['wap3']],axis=1)
            df_sub_2 = df_sub_2.rename(columns={'time_id':'row_id','wap': 'rv_2', 'wap2': 'rv2_2', 'wap3': 'rv3_2'})
        else: # 0 volatility
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_2'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_2'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_2'])
            df_sub_2 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3],axis=1) 

        list_rv.append(df_sub)
        list_rv2.append(df_sub_5)
        list_rv3.append(df_sub_2)

        # Calculate other financial metrics from book 
        df_sub_book_feats = book_stock.groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
        df_sub_book_feats = df_sub_book_feats.rename(columns={0:'embedding'})
        df_sub_book_feats[['wap_imbalance','price_spread','bid_spread','ask_spread','total_vol','vol_imbalance']] = pd.DataFrame(df_sub_book_feats.embedding.tolist(), index=df_sub_book_feats.index)
        df_sub_book_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats['time_id']] 
        df_sub_book_feats = df_sub_book_feats.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_sub_book_feats5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
            df_sub_book_feats5 = df_sub_book_feats5.rename(columns={0:'embedding'})
            df_sub_book_feats5[['wap_imbalance5','price_spread5','bid_spread5','ask_spread5','total_vol5','vol_imbalance5']] = pd.DataFrame(df_sub_book_feats5.embedding.tolist(), index=df_sub_book_feats5.index)
            df_sub_book_feats5['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats5['time_id']] 
            df_sub_book_feats5 = df_sub_book_feats5.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['wap_imbalance5']) 
            temp2 = pd.DataFrame([0],columns=['price_spread5'])
            temp3 = pd.DataFrame([0],columns=['bid_spread5'])
            temp4 = pd.DataFrame([0],columns=['ask_spread5'])
            temp5 = pd.DataFrame([0],columns=['total_vol5'])
            temp6 = pd.DataFrame([0],columns=['vol_imbalance5'])
            df_sub_book_feats5 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1) 
            
        list_fin.append(df_sub_book_feats)
        list_fin2.append(df_sub_book_feats5)

        # Compute entropy 
        isEmpty = book_stock.query(f'seconds_in_bucket >= 480').empty
        if isEmpty == False:
            df_ent = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id']).apply(entropy_from_df).to_frame().reset_index().fillna(0)
            df_ent2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id']).apply(entropy_from_df2).to_frame().reset_index().fillna(0)
            df_ent3 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id']).apply(entropy_from_df3).to_frame().reset_index().fillna(0)
            df_ent['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_ent['time_id']]
            df_ent = df_ent.rename(columns={'time_id':'row_id',0:'entropy'})
            df_ent2 = df_ent2.rename(columns={0:'entropy2'}).drop(['time_id'],axis=1)
            df_ent3 = df_ent3.rename(columns={0:'entropy3'}).drop(['time_id'],axis=1)
            df_ent = pd.concat([df_ent,df_ent2,df_ent3],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['entropy']) 
            temp2 = pd.DataFrame([0],columns=['entropy2'])
            temp3 = pd.DataFrame([0],columns=['entropy3'])
            df_ent = pd.concat([times_pd,temp,temp2,temp3],axis=1)
            
        list_ent.append(df_ent)

        # Compute other metrics
        df_others = book_stock.groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
        df_others = df_others.rename(columns={0:'embedding'})
        df_others[['linearFit1_1','linearFit1_2','linearFit1_3','wap_std1_1','wap_std1_2','wap_std1_3']] = pd.DataFrame(df_others.embedding.tolist(), index=df_others.index)
        df_others['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others['time_id']] 
        df_others = df_others.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        list_others.append(df_others)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_others2 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
            df_others2 = df_others2.rename(columns={0:'embedding'})
            df_others2[['linearFit2_1','linearFit2_2','linearFit2_3','wap_std2_1','wap_std2_2','wap_std2_3']] = pd.DataFrame(df_others2.embedding.tolist(), index=df_others2.index)
            df_others2['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others2['time_id']] 
            df_others2 = df_others2.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['linearFit2_1']) 
            temp2 = pd.DataFrame([0],columns=['linearFit2_2'])
            temp3 = pd.DataFrame([0],columns=['linearFit2_3'])
            temp4 = pd.DataFrame([0],columns=['wap_std2_1'])
            temp5 = pd.DataFrame([0],columns=['wap_std2_2'])
            temp6 = pd.DataFrame([0],columns=['wap_std2_3'])
            df_others2 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1)
            
        list_others2.append(df_others2)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 480').empty 
        if isEmpty == False:
            df_others3 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
            df_others3 = df_others3.rename(columns={0:'embedding'})
            df_others3[['linearFit3_1','linearFit3_2','linearFit3_3','wap_std3_1','wap_std3_2','wap_std3_3']] = pd.DataFrame(df_others3.embedding.tolist(), index=df_others3.index)
            df_others3['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others3['time_id']] 
            df_others3 = df_others3.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['linearFit3_1']) 
            temp2 = pd.DataFrame([0],columns=['linearFit3_2'])
            temp3 = pd.DataFrame([0],columns=['linearFit3_3'])
            temp4 = pd.DataFrame([0],columns=['wap_std3_1'])
            temp5 = pd.DataFrame([0],columns=['wap_std3_2'])
            temp6 = pd.DataFrame([0],columns=['wap_std3_3'])
            df_others3 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1)
            
        list_others3.append(df_others3)

        print('Computing one stock took', time.time() - start, 'seconds for stock ', stock_id)

    # Create features dataframe
    df_submission = pd.concat(list_rv)
    df_submission2 = pd.concat(list_rv2)
    df_submission3 = pd.concat(list_rv3)
    df_ent_concat = pd.concat(list_ent)
    df_fin_concat = pd.concat(list_fin)
    df_fin2_concat = pd.concat(list_fin2)
    df_others = pd.concat(list_others)
    df_others2 = pd.concat(list_others2)
    df_others3 = pd.concat(list_others3)

    df_book_features = df_submission.merge(df_submission2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_submission3, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_ent_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin2_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others3, on = ['row_id'], how='left').fillna(0)
    
    # Add encoded stock
    encoder = np.eye(len(all_stocks_ids))
    encoded = list()

    for i in range(df_book_features.shape[0]):
        stock_id = int(df_book_features['row_id'][i].split('-')[0])
        encoded_stock = encoder[np.where(all_stocks_ids == int(stock_id))[0],:]
        encoded.append(encoded_stock)

    encoded_pd = pd.DataFrame(np.array(encoded).reshape(df_book_features.shape[0],np.array(all_stocks_ids).shape[0]))
    df_book_features_encoded = pd.concat([df_book_features, encoded_pd],axis=1)
    
    return df_book_features_encoded

def computeFeatures_july(machine, dataset, all_stocks_ids, datapath):
    
    list_rv, list_rv2, list_rv3 = [], [], []
    list_ent, list_fin, list_fin2 = [], [], []
    list_others, list_others2, list_others3 = [], [], []

    for stock_id in range(127):
        
        start = time.time()
        
        if machine == 'local':
            try:
                book_stock = load_book_data_by_id(stock_id,datapath,dataset)
            except:
                continue
        elif machine == 'kaggle':
            try:
                book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
            except:
                continue
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Calculate wap for the book
        book_stock['wap'] = calc_wap(book_stock)
        book_stock['wap2'] = calc_wap2(book_stock)
        book_stock['wap3'] = calc_wap3(book_stock)

        # Calculate realized volatility
        df_sub = book_stock.groupby('time_id')['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub2 = book_stock.groupby('time_id')['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub3 = book_stock.groupby('time_id')['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub['time_id']]
        df_sub = pd.concat([df_sub,df_sub2['wap2'],df_sub3['wap3']],axis=1)
        df_sub = df_sub.rename(columns={'time_id':'row_id','wap': 'rv', 'wap2': 'rv2', 'wap3': 'rv3'})
        
        # Calculate realized volatility last 5 min
        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_sub_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub_5['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_5['time_id']]
            df_sub_5 = pd.concat([df_sub_5,df_sub2_5['wap2'],df_sub3_5['wap3']],axis=1)
            df_sub_5 = df_sub_5.rename(columns={'time_id':'row_id','wap': 'rv_5', 'wap2': 'rv2_5', 'wap3': 'rv3_5'})
        else: # 0 volatility
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_5'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_5'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_5'])
            df_sub_5 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3],axis=1) 

        # Calculate realized volatility last 2 min
        isEmpty = book_stock.query(f'seconds_in_bucket >= 480').empty
        if isEmpty == False:
            df_sub_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_2 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()    
            df_sub_2['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_2['time_id']] 
            df_sub_2 = pd.concat([df_sub_2,df_sub2_2['wap2'],df_sub3_2['wap3']],axis=1)
            df_sub_2 = df_sub_2.rename(columns={'time_id':'row_id','wap': 'rv_2', 'wap2': 'rv2_2', 'wap3': 'rv3_2'})
        else: # 0 volatility
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_2'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_2'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_2'])
            df_sub_2 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3],axis=1) 

        list_rv.append(df_sub)
        list_rv2.append(df_sub_5)
        list_rv3.append(df_sub_2)

        # Calculate other financial metrics from book 
        df_sub_book_feats = book_stock.groupby(['time_id']).apply(financial_metrics_2).to_frame().reset_index()
        df_sub_book_feats = df_sub_book_feats.rename(columns={0:'embedding'})
        df_sub_book_feats[['wap_imbalance','price_spread','bid_spread','ask_spread','total_vol','vol_imbalance']] = pd.DataFrame(df_sub_book_feats.embedding.tolist(), index=df_sub_book_feats.index)
        df_sub_book_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats['time_id']] 
        df_sub_book_feats = df_sub_book_feats.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_sub_book_feats5 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(financial_metrics_2).to_frame().reset_index()
            df_sub_book_feats5 = df_sub_book_feats5.rename(columns={0:'embedding'})
            df_sub_book_feats5[['wap_imbalance5','price_spread5','bid_spread5','ask_spread5','total_vol5','vol_imbalance5']] = pd.DataFrame(df_sub_book_feats5.embedding.tolist(), index=df_sub_book_feats5.index)
            df_sub_book_feats5['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats5['time_id']] 
            df_sub_book_feats5 = df_sub_book_feats5.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['wap_imbalance5']) 
            temp2 = pd.DataFrame([0],columns=['price_spread5'])
            temp3 = pd.DataFrame([0],columns=['bid_spread5'])
            temp4 = pd.DataFrame([0],columns=['ask_spread5'])
            temp5 = pd.DataFrame([0],columns=['total_vol5'])
            temp6 = pd.DataFrame([0],columns=['vol_imbalance5'])
            df_sub_book_feats5 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1) 
            
        list_fin.append(df_sub_book_feats)
        list_fin2.append(df_sub_book_feats5)

        # Compute other metrics
        df_others = book_stock.groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
        df_others = df_others.rename(columns={0:'embedding'})
        df_others[['linearFit1_1','linearFit1_2','linearFit1_3','wap_std1_1','wap_std1_2','wap_std1_3']] = pd.DataFrame(df_others.embedding.tolist(), index=df_others.index)
        df_others['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others['time_id']] 
        df_others = df_others.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        list_others.append(df_others)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 300').empty
        if isEmpty == False:
            df_others2 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
            df_others2 = df_others2.rename(columns={0:'embedding'})
            df_others2[['linearFit2_1','linearFit2_2','linearFit2_3','wap_std2_1','wap_std2_2','wap_std2_3']] = pd.DataFrame(df_others2.embedding.tolist(), index=df_others2.index)
            df_others2['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others2['time_id']] 
            df_others2 = df_others2.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['linearFit2_1']) 
            temp2 = pd.DataFrame([0],columns=['linearFit2_2'])
            temp3 = pd.DataFrame([0],columns=['linearFit2_3'])
            temp4 = pd.DataFrame([0],columns=['wap_std2_1'])
            temp5 = pd.DataFrame([0],columns=['wap_std2_2'])
            temp6 = pd.DataFrame([0],columns=['wap_std2_3'])
            df_others2 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1)
            
        list_others2.append(df_others2)

        isEmpty = book_stock.query(f'seconds_in_bucket >= 480').empty 
        if isEmpty == False:
            df_others3 = book_stock.query(f'seconds_in_bucket >= 480').groupby(['time_id']).apply(other_metrics).to_frame().reset_index().fillna(0)
            df_others3 = df_others3.rename(columns={0:'embedding'})
            df_others3[['linearFit3_1','linearFit3_2','linearFit3_3','wap_std3_1','wap_std3_2','wap_std3_3']] = pd.DataFrame(df_others3.embedding.tolist(), index=df_others3.index)
            df_others3['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_others3['time_id']] 
            df_others3 = df_others3.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['linearFit3_1']) 
            temp2 = pd.DataFrame([0],columns=['linearFit3_2'])
            temp3 = pd.DataFrame([0],columns=['linearFit3_3'])
            temp4 = pd.DataFrame([0],columns=['wap_std3_1'])
            temp5 = pd.DataFrame([0],columns=['wap_std3_2'])
            temp6 = pd.DataFrame([0],columns=['wap_std3_3'])
            df_others3 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1)
            
        list_others3.append(df_others3)

        print('Computing one stock took', time.time() - start, 'seconds for stock ', stock_id)

    # Create features dataframe
    df_submission = pd.concat(list_rv)
    df_submission2 = pd.concat(list_rv2)
    df_submission3 = pd.concat(list_rv3)
    df_ent_concat = pd.concat(list_ent)
    df_fin_concat = pd.concat(list_fin)
    df_fin2_concat = pd.concat(list_fin2)
    df_others = pd.concat(list_others)
    df_others2 = pd.concat(list_others2)
    df_others3 = pd.concat(list_others3)

    df_book_features = df_submission.merge(df_submission2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_submission3, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_ent_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin2_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_others3, on = ['row_id'], how='left').fillna(0)
    
    # Add encoded stock
    encoder = np.eye(len(all_stocks_ids))
    encoded = list()

    for i in range(df_book_features.shape[0]):
        stock_id = int(df_book_features['row_id'][i].split('-')[0])
        encoded_stock = encoder[np.where(all_stocks_ids == int(stock_id))[0],:]
        encoded.append(encoded_stock)

    encoded_pd = pd.DataFrame(np.array(encoded).reshape(df_book_features.shape[0],np.array(all_stocks_ids).shape[0]))
    df_book_features_encoded = pd.concat([df_book_features, encoded_pd],axis=1)
    
    return df_book_features_encoded

def computeFeatures_newTest_Laurent(machine, dataset, all_stocks_ids, datapath):
    
    list_rv, list_rv2, list_rv3 = [], [], []
    list_ent, list_fin, list_fin2 = [], [], []
    list_others, list_others2, list_others3 = [], [], []

    for stock_id in range(127):
        
        start = time.time()
        
        if machine == 'local':
            try:
                book_stock = load_book_data_by_id(stock_id,datapath,dataset)
            except:
                continue
        elif machine == 'kaggle':
            try:
                book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
            except:
                continue
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Calculate wap for the entire book
        book_stock['wap'] = calc_wap(book_stock)
        book_stock['wap2'] = calc_wap2(book_stock)
        book_stock['wap3'] = calc_wap3(book_stock)
        book_stock['wap4'] = calc_wap2(book_stock)
        book_stock['mid_price'] = calc_wap3(book_stock)

        # Calculate past realized volatility per time_id
        df_sub = book_stock.groupby('time_id')['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub2 = book_stock.groupby('time_id')['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub3 = book_stock.groupby('time_id')['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub4 = book_stock.groupby('time_id')['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub5 = book_stock.groupby('time_id')['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        
        df_sub['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub['time_id']]
        df_sub = df_sub.rename(columns={'time_id':'row_id'})
        
        df_sub = pd.concat([df_sub,df_sub2['wap2'],df_sub3['wap3'], df_sub4['wap4'], df_sub5['mid_price']],axis=1)
        df_sub = df_sub.rename(columns={'wap': 'rv', 'wap2': 'rv2', 'wap3': 'rv3', 'wap4':'rv4','mid_price':'rv5'})
        
        list_rv.append(df_sub)
        
        # Query segments
        bucketQuery480 = book_stock.query(f'seconds_in_bucket >= 480')
        isEmpty480 = bucketQuery480.empty
        
        bucketQuery300 = book_stock.query(f'seconds_in_bucket >= 300')
        isEmpty300 = bucketQuery300.empty
        
        times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
        times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
        times_pd = times_pd.rename(columns={'time_id':'row_id'})
        
        # Calculate past realized volatility per time_id and query subset
        if isEmpty300 == False:
            df_sub_300 = bucketQuery300.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_300 = bucketQuery300.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_300 = bucketQuery300.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_300 = bucketQuery300.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_300 = bucketQuery300.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_300 = pd.concat([times_pd,df_sub_300['wap'],df_sub2_300['wap2'],df_sub3_300['wap3'],df_sub4_300['wap4'],df_sub5_300['mid_price']],axis=1)
            df_sub_300 = df_sub_300.rename(columns={'wap': 'rv_300', 'wap2_300': 'rv2', 'wap3_300': 'rv3', 'wap4':'rv4_300','mid_price':'rv5_300'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_300'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_300'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_300'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_300'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_300'])
            df_sub_300 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 
            
        list_rv2.append(df_sub_300)
        
        # Calculate realized volatility last 2 min
        if isEmpty480 == False:
            df_sub_480 = bucketQuery480.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_480 = bucketQuery480.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_480 = bucketQuery480.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_480 = bucketQuery480.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_480 = bucketQuery480.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_480 = pd.concat([times_pd,df_sub_480['wap'],df_sub2_480['wap2'],df_sub3_480['wap3'],df_sub4_480['wap4'],df_sub5_480['mid_price']],axis=1)
            df_sub_480 = df_sub_480.rename(columns={'wap': 'rv_480', 'wap2_480': 'rv2', 'wap3_480': 'rv3', 'wap4':'rv4_480','mid_price':'rv5_480'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_480'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_480'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_480'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_480'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_480'])
            df_sub_480 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 

        
        list_rv3.append(df_sub_480)

        # Calculate other financial metrics from book 
        df_sub_book_feats = book_stock.groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
        df_sub_book_feats = df_sub_book_feats.rename(columns={0:'embedding'})
        df_sub_book_feats[['wap_imbalance','price_spread','bid_spread','ask_spread','total_vol','vol_imbalance']] = pd.DataFrame(df_sub_book_feats.embedding.tolist(), index=df_sub_book_feats.index)
        df_sub_book_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats['time_id']] 
        df_sub_book_feats = df_sub_book_feats.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        list_fin.append(df_sub_book_feats)
            
        if isEmpty300 == False:
            df_sub_book_feats_300 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={0:'embedding'})
            df_sub_book_feats_300[['wap_imbalance5','price_spread5','bid_spread5','ask_spread5','total_vol5','vol_imbalance5']] = pd.DataFrame(df_sub_book_feats_300.embedding.tolist(), index=df_sub_book_feats_300.index)
            df_sub_book_feats_300['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats_300['time_id']] 
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['wap_imbalance5']) 
            temp2 = pd.DataFrame([0],columns=['price_spread5'])
            temp3 = pd.DataFrame([0],columns=['bid_spread5'])
            temp4 = pd.DataFrame([0],columns=['ask_spread5'])
            temp5 = pd.DataFrame([0],columns=['total_vol5'])
            temp6 = pd.DataFrame([0],columns=['vol_imbalance5'])
            df_sub_book_feats_300 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1) 
            
        list_fin2.append(df_sub_book_feats_300)
        
        print('Computing one stock took', time.time() - start, 'seconds for stock ', stock_id)

    # Create features dataframe
    df_submission = pd.concat(list_rv)
    df_submission2 = pd.concat(list_rv2)
    df_submission3 = pd.concat(list_rv3)
    df_fin_concat = pd.concat(list_fin)
    df_fin2_concat = pd.concat(list_fin2)

    df_book_features = df_submission.merge(df_submission2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_submission3, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin2_concat, on = ['row_id'], how='left').fillna(0)
    
    # Add encoded stock
    encoder = np.eye(len(all_stocks_ids))
    encoded = list()

    for i in range(df_book_features.shape[0]):
        stock_id = int(df_book_features['row_id'][i].split('-')[0])
        encoded_stock = encoder[np.where(all_stocks_ids == int(stock_id))[0],:]
        encoded.append(encoded_stock)

    encoded_pd = pd.DataFrame(np.array(encoded).reshape(df_book_features.shape[0],np.array(all_stocks_ids).shape[0]))
    df_book_features_encoded = pd.concat([df_book_features, encoded_pd],axis=1)
    
    return df_book_features_encoded

def computeFeatures_newTest_Laurent_noCode(machine, dataset, all_stocks_ids, datapath):
    
    list_rv, list_rv2, list_rv3 = [], [], []
    list_ent, list_fin, list_fin2 = [], [], []
    list_others, list_others2, list_others3 = [], [], []

    for stock_id in range(127):
        
        start = time.time()
        
        if machine == 'local':
            try:
                book_stock = load_book_data_by_id(stock_id,datapath,dataset)
            except:
                continue
        elif machine == 'kaggle':
            try:
                book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
            except:
                continue
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Calculate wap for the entire book
        book_stock['wap'] = calc_wap(book_stock)
        book_stock['wap2'] = calc_wap2(book_stock)
        book_stock['wap3'] = calc_wap3(book_stock)
        book_stock['wap4'] = calc_wap2(book_stock)
        book_stock['mid_price'] = calc_wap3(book_stock)

        # Calculate past realized volatility per time_id
        df_sub = book_stock.groupby('time_id')['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub2 = book_stock.groupby('time_id')['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub3 = book_stock.groupby('time_id')['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub4 = book_stock.groupby('time_id')['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub5 = book_stock.groupby('time_id')['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        
        df_sub['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub['time_id']]
        df_sub = df_sub.rename(columns={'time_id':'row_id'})
        
        df_sub = pd.concat([df_sub,df_sub2['wap2'],df_sub3['wap3'], df_sub4['wap4'], df_sub5['mid_price']],axis=1)
        df_sub = df_sub.rename(columns={'wap': 'rv', 'wap2': 'rv2', 'wap3': 'rv3', 'wap4':'rv4','mid_price':'rv5'})
        
        list_rv.append(df_sub)
        
        # Query segments
        bucketQuery480 = book_stock.query(f'seconds_in_bucket >= 480')
        isEmpty480 = bucketQuery480.empty
        
        bucketQuery300 = book_stock.query(f'seconds_in_bucket >= 300')
        isEmpty300 = bucketQuery300.empty
        
        times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
        times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
        times_pd = times_pd.rename(columns={'time_id':'row_id'})
        
        # Calculate past realized volatility per time_id and query subset
        if isEmpty300 == False:
            df_sub_300 = bucketQuery300.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_300 = bucketQuery300.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_300 = bucketQuery300.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_300 = bucketQuery300.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_300 = bucketQuery300.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_300 = pd.concat([times_pd,df_sub_300['wap'],df_sub2_300['wap2'],df_sub3_300['wap3'],df_sub4_300['wap4'],df_sub5_300['mid_price']],axis=1)
            df_sub_300 = df_sub_300.rename(columns={'wap': 'rv_300', 'wap2_300': 'rv2', 'wap3_300': 'rv3', 'wap4':'rv4_300','mid_price':'rv5_300'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_300'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_300'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_300'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_300'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_300'])
            df_sub_300 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 
            
        list_rv2.append(df_sub_300)
        
        # Calculate realized volatility last 2 min
        if isEmpty480 == False:
            df_sub_480 = bucketQuery480.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_480 = bucketQuery480.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_480 = bucketQuery480.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_480 = bucketQuery480.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_480 = bucketQuery480.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_480 = pd.concat([times_pd,df_sub_480['wap'],df_sub2_480['wap2'],df_sub3_480['wap3'],df_sub4_480['wap4'],df_sub5_480['mid_price']],axis=1)
            df_sub_480 = df_sub_480.rename(columns={'wap': 'rv_480', 'wap2_480': 'rv2', 'wap3_480': 'rv3', 'wap4':'rv4_480','mid_price':'rv5_480'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_480'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_480'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_480'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_480'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_480'])
            df_sub_480 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 

        
        list_rv3.append(df_sub_480)

        # Calculate other financial metrics from book 
        df_sub_book_feats = book_stock.groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
        df_sub_book_feats = df_sub_book_feats.rename(columns={0:'embedding'})
        df_sub_book_feats[['wap_imbalance','price_spread','bid_spread','ask_spread','total_vol','vol_imbalance']] = pd.DataFrame(df_sub_book_feats.embedding.tolist(), index=df_sub_book_feats.index)
        df_sub_book_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats['time_id']] 
        df_sub_book_feats = df_sub_book_feats.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        list_fin.append(df_sub_book_feats)
            
        if isEmpty300 == False:
            df_sub_book_feats_300 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(financial_metrics).to_frame().reset_index()
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={0:'embedding'})
            df_sub_book_feats_300[['wap_imbalance5','price_spread5','bid_spread5','ask_spread5','total_vol5','vol_imbalance5']] = pd.DataFrame(df_sub_book_feats_300.embedding.tolist(), index=df_sub_book_feats_300.index)
            df_sub_book_feats_300['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats_300['time_id']] 
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
            times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
            times_pd = times_pd.rename(columns={'time_id':'row_id'})
            temp = pd.DataFrame([0],columns=['wap_imbalance5']) 
            temp2 = pd.DataFrame([0],columns=['price_spread5'])
            temp3 = pd.DataFrame([0],columns=['bid_spread5'])
            temp4 = pd.DataFrame([0],columns=['ask_spread5'])
            temp5 = pd.DataFrame([0],columns=['total_vol5'])
            temp6 = pd.DataFrame([0],columns=['vol_imbalance5'])
            df_sub_book_feats_300 = pd.concat([times_pd,temp,temp2,temp3,temp4,temp5,temp6],axis=1) 
            
        list_fin2.append(df_sub_book_feats_300)
        
        print('Computing one stock took', time.time() - start, 'seconds for stock ', stock_id)

    # Create features dataframe
    df_submission = pd.concat(list_rv)
    df_submission2 = pd.concat(list_rv2)
    df_submission3 = pd.concat(list_rv3)
    df_fin_concat = pd.concat(list_fin)
    df_fin2_concat = pd.concat(list_fin2)

    df_book_features = df_submission.merge(df_submission2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_submission3, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin2_concat, on = ['row_id'], how='left').fillna(0)
    
    return df_book_features

def computeFeatures_newTest_Laurent_wTrades(machine, dataset, all_stocks_ids, datapath):
    
    list_rv, list_rv2, list_rv3 = [], [], []
    list_ent, list_fin, list_fin2 = [], [], []
    list_others, list_others2, list_others3 = [], [], []
    list_trades1, list_trades2 = [], []
    list_vlad_book, list_vlad_trades = [], []
    for stock_id in range(127):
        
        start = time.time()
        
        if machine == 'local':
            try:
                book_stock = load_book_data_by_id(stock_id,datapath,dataset)
                trades_stock = load_trades_data_by_id(stock_id,datapath,dataset)
            except:
                continue
        elif machine == 'kaggle':
            try:
                book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
                trades_stock = load_trades_data_by_id_kaggle(stock_id,dataset)
            except:
                continue
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Calculate wap for the entire book
        book_stock['wap'] = calc_wap(book_stock)
        book_stock['wap2'] = calc_wap2(book_stock)
        book_stock['wap3'] = calc_wap3(book_stock)
        book_stock['wap4'] = calc_wap4(book_stock)
        book_stock['mid_price'] = mid_price(book_stock)

        # Calculate past realized volatility per time_id
        df_sub = book_stock.groupby('time_id')['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub2 = book_stock.groupby('time_id')['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub3 = book_stock.groupby('time_id')['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub4 = book_stock.groupby('time_id')['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        df_sub5 = book_stock.groupby('time_id')['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
        
        df_sub['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub['time_id']]
        df_sub = df_sub.rename(columns={'time_id':'row_id'})
        
        df_sub = pd.concat([df_sub,df_sub2['wap2'],df_sub3['wap3'], df_sub4['wap4'], df_sub5['mid_price']],axis=1)
        df_sub = df_sub.rename(columns={'wap': 'rv', 'wap2': 'rv2', 'wap3': 'rv3', 'wap4':'rv4','mid_price':'rv5'})
        
        list_rv.append(df_sub)
        
        # Query segments
        bucketQuery480 = book_stock.query(f'seconds_in_bucket >= 480')
        isEmpty480 = bucketQuery480.empty
        
        bucketQuery300 = book_stock.query(f'seconds_in_bucket >= 300')
        isEmpty300 = bucketQuery300.empty
        
        times_pd = pd.DataFrame(all_time_ids_byStock,columns=['time_id'])
        times_pd['time_id'] = [f'{stock_id}-{time_id}' for time_id in times_pd['time_id']]
        times_pd = times_pd.rename(columns={'time_id':'row_id'})
        
        # Calculate past realized volatility per time_id and query subset
        if isEmpty300 == False:
            df_sub_300 = bucketQuery300.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_300 = bucketQuery300.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_300 = bucketQuery300.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_300 = bucketQuery300.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_300 = bucketQuery300.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_300 = pd.concat([times_pd,df_sub_300['wap'],df_sub2_300['wap2'],df_sub3_300['wap3'],df_sub4_300['wap4'],df_sub5_300['mid_price']],axis=1)
            df_sub_300 = df_sub_300.rename(columns={'wap': 'rv_300', 'wap2': 'rv2_300', 'wap3': 'rv3_300', 'wap4':'rv4_300','mid_price':'rv5_300'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_300'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_300'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_300'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_300'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_300'])
            df_sub_300 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 
            
        list_rv2.append(df_sub_300)
        
        # Calculate realized volatility last 2 min
        if isEmpty480 == False:
            df_sub_480 = bucketQuery480.groupby(['time_id'])['wap'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub2_480 = bucketQuery480.groupby(['time_id'])['wap2'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub3_480 = bucketQuery480.groupby(['time_id'])['wap3'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub4_480 = bucketQuery480.groupby(['time_id'])['wap4'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            df_sub5_480 = bucketQuery480.groupby(['time_id'])['mid_price'].agg(calc_rv_from_wap_numba, engine='numba').to_frame().reset_index()
            

            df_sub_480 = pd.concat([times_pd,df_sub_480['wap'],df_sub2_480['wap2'],df_sub3_480['wap3'],df_sub4_480['wap4'],df_sub5_480['mid_price']],axis=1)
            df_sub_480 = df_sub_480.rename(columns={'wap': 'rv_480', 'wap2': 'rv2_480', 'wap3': 'rv3_480', 'wap4':'rv4_480','mid_price':'rv5_480'})
            
        else: # 0 volatility
            
            zero_rv = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv_480'])
            zero_rv2 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv2_480'])
            zero_rv3 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv3_480'])
            zero_rv4 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv4_480'])
            zero_rv5 = pd.DataFrame(np.zeros((1,times_pd.shape[0])),columns=['rv5_480'])
            df_sub_480 = pd.concat([times_pd,zero_rv,zero_rv2,zero_rv3,zero_rv4,zero_rv5],axis=1) 

        
        list_rv3.append(df_sub_480)

        # Calculate other financial metrics from book 
        df_sub_book_feats = book_stock.groupby(['time_id']).apply(financial_metrics_2).to_frame().reset_index()
        df_sub_book_feats = df_sub_book_feats.rename(columns={0:'embedding'})
        df_sub_book_feats[['wap_imbalance_mean','price_spread_mean','bid_spread_mean','ask_spread_mean','total_vol_mean','vol_imbalance_mean','wap_imbalance_sum','price_spread_sum','bid_spread_sum','ask_spread_sum','total_vol_sum','vol_imbalance_sum','wap_imbalance_std','price_spread_std','bid_spread_std','ask_spread_std','total_vol_std','vol_imbalance_std','wap_imbalance_max','price_spread_max','bid_spread_max','ask_spread_max','total_vol_max','vol_imbalance_max','wap_imbalance_min','price_spread_min','bid_spread_min','ask_spread_min','total_vol_min','vol_imbalance_min']] = pd.DataFrame(df_sub_book_feats.embedding.tolist(), index=df_sub_book_feats.index)
        df_sub_book_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats['time_id']] 
        df_sub_book_feats = df_sub_book_feats.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        list_fin.append(df_sub_book_feats)
            
        if isEmpty300 == False:
            df_sub_book_feats_300 = book_stock.query(f'seconds_in_bucket >= 300').groupby(['time_id']).apply(financial_metrics_2).to_frame().reset_index()
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={0:'embedding'})
            df_sub_book_feats_300[['wap_imbalance_mean','price_spread_mean','bid_spread_mean','ask_spread_mean','total_vol_mean','vol_imbalance_mean','wap_imbalance_sum','price_spread_sum','bid_spread_sum','ask_spread_sum','total_vol_sum','vol_imbalance_sum','wap_imbalance_std','price_spread_std','bid_spread_std','ask_spread_std','total_vol_std','vol_imbalance_std','wap_imbalance_max','price_spread_max','bid_spread_max','ask_spread_max','total_vol_max','vol_imbalance_max','wap_imbalance_min','price_spread_min','bid_spread_min','ask_spread_min','total_vol_min','vol_imbalance_min']] = pd.DataFrame(df_sub_book_feats_300.embedding.tolist(), index=df_sub_book_feats_300.index)
                
            df_sub_book_feats_300['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_book_feats_300['time_id']] 
            df_sub_book_feats_300 = df_sub_book_feats_300.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)
        else:
            df_sub_book_feats_300 = df_sub_book_feats.copy()
            for col in df_sub_book_feats_300.columns:
                df_sub_book_feats_300[col].values[:] = 0
            
        list_fin2.append(df_sub_book_feats_300)
        
        # Trades features (sum, mean, std, max, min)
        df_sub_trades_feats = trades_stock.groupby(['time_id'])['price','size','order_count'].agg(['sum','mean','std','max','min']).reset_index()
        df_sub_trades_feats['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_trades_feats['time_id']]
        df_sub_trades_feats = df_sub_trades_feats.rename(columns={'time_id':'row_id'})
        
        list_trades1.append(df_sub_trades_feats)
        
        # Query segments
        bucketQuery300_trades = trades_stock.query(f'seconds_in_bucket >= 300')
        isEmpty300_trades = bucketQuery300_trades.empty
        
        if isEmpty300_trades == False:
            df_sub_trades_300 = bucketQuery300_trades.groupby(['time_id'])['price','size','order_count'].agg(['sum','mean','std','max','min']).reset_index()
            df_sub_trades_300['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_sub_trades_300['time_id']]
            df_sub_trades_300 = df_sub_trades_300.rename(columns={'time_id':'row_id'})
        else:
            df_sub_trades_300 = df_sub_trades_feats.copy()
            for col in df_sub_trades_300.columns:
                df_sub_trades_300[col].values[:] = 0
        
        list_trades2.append(df_sub_trades_300)
            
            
        # Fin metrics book
        df_fin_metrics_book = book_stock.groupby(['time_id']).apply(fin_metrics_book_data).to_frame().reset_index()
        df_fin_metrics_book = df_fin_metrics_book.rename(columns={0:'embedding'})
        df_fin_metrics_book[['spread','depth_imb']] = pd.DataFrame(df_fin_metrics_book.embedding.tolist(), index=df_fin_metrics_book.index)
        df_fin_metrics_book['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_fin_metrics_book['time_id']] 
        df_fin_metrics_book = df_fin_metrics_book.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        list_vlad_book.append(df_fin_metrics_book)
        
        # Fin metrics trades
        df_fin_metrics_trades = trades_stock.groupby(['time_id']).apply(fin_metrics_trades_data).to_frame().reset_index()
        df_fin_metrics_trades = df_fin_metrics_trades.rename(columns={0:'embedding'})
        df_fin_metrics_trades[['roll_measure', 'roll_impact', 'mkt_impact', 'amihud']] = pd.DataFrame(df_fin_metrics_trades.embedding.tolist(), index=df_fin_metrics_trades.index)
        df_fin_metrics_trades['time_id'] = [f'{stock_id}-{time_id}' for time_id in df_fin_metrics_trades['time_id']] 
        df_fin_metrics_trades = df_fin_metrics_trades.rename(columns={'time_id':'row_id'}).drop(['embedding'],axis=1)

        list_vlad_trades.append(df_fin_metrics_trades)
        
        
        print('Computing one stock took', time.time() - start, 'seconds for stock ', stock_id)

    # Create features dataframe
    df_submission = pd.concat(list_rv)
    df_submission2 = pd.concat(list_rv2)
    df_submission3 = pd.concat(list_rv3)
    df_fin_concat = pd.concat(list_fin)
    df_fin2_concat = pd.concat(list_fin2)
    df_trades1 = pd.concat(list_trades1) 
    df_trades2 = pd.concat(list_trades2)
    df_vlad_book = pd.concat(list_vlad_book)
    df_vlad_trades = pd.concat(list_vlad_trades)
    
    df_book_features = df_submission.merge(df_submission2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_submission3, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_fin2_concat, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_trades1, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_trades2, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_vlad_book, on = ['row_id'], how='left').fillna(0)
    df_book_features = df_book_features.merge(df_vlad_trades, on = ['row_id'], how='left').fillna(0)
    
    return df_book_features


def computeFeatures_2807(machine, dataset, all_stocks_ids, datapath):
    
    # Create parallel function
    def for_joblib(stock_id):
             
        if machine == 'local':
            book_stock = load_book_data_by_id(stock_id,datapath,dataset)
            trades_stock = load_trades_data_by_id(stock_id,datapath,dataset)
            
        elif machine == 'kaggle':
            book_stock = load_book_data_by_id_kaggle(stock_id,dataset)
            trades_stock = load_trades_data_by_id_kaggle(stock_id,dataset)
        
        # Useful
        all_time_ids_byStock = book_stock['time_id'].unique() 

        # Book stats processing
        book_features_df = book_preprocessor(book_stock, stock_id)
        
        # Trades stats processing
        trades_features_df = trade_preprocessor(trades_stock, stock_id)
        
        df_tmp = pd.merge(book_features_df, trades_features_df, on = 'row_id', how = 'left')
        
        return df_tmp 
    
    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs = -1, verbose = 1)(delayed(for_joblib)(stock_id) for stock_id in all_stocks_ids)
    
    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index = True)
    
    return df

def book_preprocessor(book_stock, stock_id):
    
    # Calculate wap for the entire book
    book_stock['wap'] = calc_wap(book_stock)
    book_stock['wap2'] = calc_wap2(book_stock)
    book_stock['wap3'] = calc_wap3(book_stock)
    book_stock['wap4'] = calc_wap4(book_stock)
    book_stock['mid_price'] = mid_price(book_stock)
        
    # Calculate log returns
    book_stock['log_return1'] = book_stock.groupby(['time_id'])['wap'].apply(log_return)
    book_stock['log_return2'] = book_stock.groupby(['time_id'])['wap2'].apply(log_return)
    book_stock['log_return3'] = book_stock.groupby(['time_id'])['wap3'].apply(log_return)
    book_stock['log_return4'] = book_stock.groupby(['time_id'])['wap4'].apply(log_return)
    book_stock['log_returnMidprice'] = book_stock.groupby(['time_id'])['mid_price'].apply(log_return)
    
    # Wap imbalances
    book_stock['wap_imbalance1'] = book_stock['wap'] - book_stock['wap2']
    book_stock['wap_imbalance2'] = book_stock['wap3'] - book_stock['wap4']
    
    # Spread
    book_stock['price_spread'] = (book_stock['ask_price1'] - book_stock['bid_price1']) / ((book_stock['ask_price1'] + book_stock['bid_price1'])/2)
    book_stock['bid_spread'] = book_stock['bid_price1'] - book_stock['bid_price2']  
    book_stock['ask_spread'] = book_stock['ask_price1'] - book_stock['ask_price2'] 
    book_stock['total_volume'] = (book_stock['ask_size1'] + book_stock['ask_size2']) + (book_stock['bid_size1'] + book_stock['bid_size2'])
    book_stock['volume_imbalance'] = abs((book_stock['ask_size1'] + book_stock['ask_size2']) - (book_stock['bid_size1'] + book_stock['bid_size2']))
    
    # Time Length (for next computation) - this should be done after grouping
    book_stock['time_length'] = book_stock.groupby('time_id')['seconds_in_bucket'].diff().shift(periods=-1)
    final_observations = book_stock.index[np.isnan(book_stock['time_length'])].tolist()
    book_stock.loc[final_observations, 'time_length'] = 600 - book_stock.loc[final_observations, 'seconds_in_bucket']
    
    # Spread
    book_stock['spread'] = book_stock['ask_price1'] - book_stock['bid_price1']
    book_stock['spread'] = book_stock['spread'] * book_stock['time_length'] / 600
    
    # Depth imbalance
    book_stock['depth_imbalance'] = (book_stock['bid_size1']/(book_stock['mid_price']-book_stock['bid_price1']) + \
                                  book_stock['bid_size2']/(book_stock['mid_price']-book_stock['bid_price2'])) / \
                                 (book_stock['ask_size1']/(-book_stock['mid_price']+book_stock['ask_price1']) + \
                                  book_stock['ask_size2']/(-book_stock['mid_price']+book_stock['ask_price2'])) 
    
    book_stock['depth_imbalance'] = book_stock['depth_imbalance'] * book_stock['time_length'] / 600
    
    # Dictionnary for aggregations
    create_feature_dict = {
        'wap': [np.sum, np.mean, np.std],
        'wap2': [np.sum, np.mean, np.std],
        'wap3': [np.sum, np.mean, np.std],
        'wap4': [np.sum, np.mean, np.std],
        'mid_price': [np.sum, np.mean, np.std],
        'log_return1': [np.sum, realized_volatility, np.mean, np.std],
        'log_return2': [np.sum, realized_volatility, np.mean, np.std],
        'log_return3': [np.sum, realized_volatility, np.mean, np.std],
        'log_return4': [np.sum, realized_volatility, np.mean, np.std],
        'log_returnMidprice': [np.sum, realized_volatility, np.mean, np.std],
        'wap_imbalance1': [np.sum, np.mean, np.std],
        'wap_imbalance2': [np.sum, np.mean, np.std],
        'price_spread':[np.sum, np.mean, np.std],
        'bid_spread':[np.sum, np.mean, np.std],
        'ask_spread':[np.sum, np.mean, np.std],
        'total_volume':[np.sum, np.mean, np.std],
        'volume_imbalance':[np.sum, np.mean, np.std],
        'spread' : [np.sum],
        'depth_imbalance' : [np.sum]
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        
        # Group by the window
        df_feature = book_stock[book_stock['seconds_in_bucket'] >= seconds_in_bucket].groupby(['time_id']).agg(create_feature_dict).reset_index()
        
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]
        
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
        
        return df_feature

    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_480 = get_stats_window(seconds_in_bucket = 480, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_120 = get_stats_window(seconds_in_bucket = 120, add_suffix = True)
    
    # Merge all
    df_feature = df_feature.merge(df_feature_480, how = 'left', left_on = 'time_id_', right_on = 'time_id__480')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_120, how = 'left', left_on = 'time_id_', right_on = 'time_id__120')
    
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__480', 'time_id__300', 'time_id__120'], axis = 1, inplace = True)
    
    # Create row_id so we can merge
    df_feature['row_id'] = df_feature['time_id_'].apply(lambda x: f'{stock_id}-{x}')
    df_feature.drop(['time_id_'], axis = 1, inplace = True)
    
    return df_feature


def fin_metrics_trades_data(df):
    
    if 'order_count' not in df.columns:
        sys.exit("Trades data format requred")
        
    df = df.copy()
    
    # compute neccessary cols
    df['log_return'] = log_return(df['price'])
    df['d_price']    = df['price'].diff()
    df['d_price_l1'] = df['d_price'].shift(1)
    
    # compute metrics
    roll_measure = 2 * np.sqrt(np.abs(df['d_price'].cov(df['d_price_l1'])))
    roll_impact = roll_measure / (np.sum(df['price'] * df['size']))
    mkt_impact = np.sum(np.abs(df['d_price'])) / np.sum(df['size'])
    amihud = np.abs(np.sum(df['log_return'])) / np.sum(df['size'])
    
    return [roll_measure, roll_impact, mkt_impact, amihud]

def trade_preprocessor(trades_stock, stock_id):
    
    trades_stock['log_return'] = trades_stock.groupby('time_id')['price'].apply(log_return)
    trades_stock['d_price'] = trades_stock['price'].diff()
    trades_stock['d_price_l1'] = trades_stock['d_price'].shift(1)
    
    # Dictionnary for aggregations
    create_feature_dict = {
        'log_return':[realized_volatility],
        'seconds_in_bucket':[count_unique],
        'size':[np.sum],
        'order_count':[np.sum, np.mean],
    }
    
    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(seconds_in_bucket, add_suffix = False):
        
        # Group by the window
        trades_stock_sub = trades_stock[trades_stock['seconds_in_bucket'] >= seconds_in_bucket]
        df_feature = trades_stock_sub.groupby(['time_id']).agg(create_feature_dict).reset_index()
        
        # Rename columns joining suffix
        df_feature.columns = ['_'.join(col) for col in df_feature.columns]

        if trades_stock_sub.empty == False:
            roll_measure = trades_stock.groupby('time_id').apply(lambda x: 2 * np.sqrt(np.abs(x['d_price'].cov(x['d_price_l1'])))).reset_index(name='roll_measure')
            roll_impact = trades_stock.groupby('time_id').apply(lambda x: (2 * np.sqrt(np.abs(x['d_price'].cov(x['d_price_l1']))))/ (np.sum(x['price'] * x['size']))).reset_index(name='roll_impact')
            mkt_impact = trades_stock.groupby('time_id').apply(lambda x: np.sum(np.abs(x['d_price'])) / np.sum(x['size'])).reset_index(name='mkt_impact')
            amihud = trades_stock.groupby('time_id').apply(lambda x: np.abs(np.sum(x['log_return'])) / np.sum(x['size'])).reset_index(name='amihud')
            traded_volume = trades_stock.groupby('time_id').apply(lambda x: np.sum(x['size'] * x['price'])).reset_index(name='traded_volume')
            avg_trade_size = trades_stock.groupby('time_id').apply(lambda x: np.sum(x['size']) / np.sum(x['order_count'])).reset_index(name='avg_trade_size')
            df_feature = pd.concat([df_feature,roll_measure['roll_measure'],roll_impact['roll_impact'],mkt_impact['mkt_impact'],amihud['amihud'],traded_volume['traded_volume'],avg_trade_size['avg_trade_size']],axis=1)
        else:
            roll_measure = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'roll_measure':0})
            roll_impact = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'roll_impact':0})
            mkt_impact = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'mkt_impact':0})
            amihud = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'amihud':0})
            traded_volume = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'traded_volume':0})
            avg_trade_size = pd.DataFrame({'time_id':trades_stock['time_id'].unique(), 'avg_trade_size':0})
            df_feature = pd.concat([df_feature,roll_measure['roll_measure'],roll_impact['roll_impact'],mkt_impact['mkt_impact'],amihud['amihud'],traded_volume['traded_volume'],avg_trade_size['avg_trade_size']],axis=1)
            
        
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature = df_feature.add_suffix('_' + str(seconds_in_bucket))
            
        return df_feature
    
    # Get the stats for different windows
    df_feature = get_stats_window(seconds_in_bucket = 0, add_suffix = False)
    df_feature_480 = get_stats_window(seconds_in_bucket = 480, add_suffix = True)
    df_feature_300 = get_stats_window(seconds_in_bucket = 300, add_suffix = True)
    df_feature_120 = get_stats_window(seconds_in_bucket = 120, add_suffix = True)
    
    # Merge all
    df_feature = df_feature.merge(df_feature_480, how = 'left', left_on = 'time_id_', right_on = 'time_id__480')
    df_feature = df_feature.merge(df_feature_300, how = 'left', left_on = 'time_id_', right_on = 'time_id__300')
    df_feature = df_feature.merge(df_feature_120, how = 'left', left_on = 'time_id_', right_on = 'time_id__120')
    
    # Drop unnecesary time_ids
    df_feature.drop(['time_id__480', 'time_id__300', 'time_id__120'], axis = 1, inplace = True)
    
    df_feature = df_feature.add_prefix('trade_')
    df_feature['row_id'] = df_feature['trade_time_id_'].apply(lambda x:f'{stock_id}-{x}')
    df_feature.drop(['trade_time_id_'], axis = 1, inplace = True)
    
    return df_feature

def count_unique(series):
    return len(np.unique(series))

def fin_metrics_book_data(df):
    
    if 'bid_price1' not in df.columns:
        sys.exit("Book data format requred")
    
    df = df.copy()
    
    # compute time length
    df['time_length'] = df['seconds_in_bucket'].diff()
    df.time_length = df.time_length.shift(periods=-1)
    df.loc[len(df)-1,'time_length'] = 600 - df['seconds_in_bucket'].iloc[-1]
    
    # quoted spread
    df['spread'] = df['ask_price1'] - df['bid_price1']

    # depth imbalance
    df['mid_price'] = (df['ask_price1'] + df['bid_price1'])/2 # we need the midprice to calculate the imbalance

    df['depth_imbalance'] = (df['bid_size1']/(df['mid_price']-df['bid_price1']) + \
                                  df['bid_size2']/(df['mid_price']-df['bid_price2'])) / \
                                 (df['ask_size1']/(-df['mid_price']+df['ask_price1']) + \
                                  df['ask_size2']/(-df['mid_price']+df['ask_price2'])) 

    # Compute the weighted averages
    spread = np.sum(df['spread'] * df['time_length']) / 600
    depth_imb = np.sum(df['depth_imbalance'] * df['time_length']) / 600
    
    return [spread, depth_imb]

