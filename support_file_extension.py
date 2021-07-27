
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
            

            df_sub_300 = pd.concat([times_pd,df_sub_300['wap'],df_sub2_300['wap2'],df_sub3_300['wap3'],df_sub4_300['wap4'],df_sub5_300['mid_range']],axis=1)
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
            

            df_sub_480 = pd.concat([times_pd,df_sub_480['wap'],df_sub2_480['wap2'],df_sub3_480['wap3'],df_sub4_480['wap4'],df_sub5_480['mid_range']],axis=1)
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
