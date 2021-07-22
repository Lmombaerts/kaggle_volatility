
import pandas as pd
import numpy as np
import math

from scipy.interpolate import interp1d
import time 

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from information_measures import *

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
    
def ApEn_new(U, m, r):
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))

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
