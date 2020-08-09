def encode_nominal_cat(dat_df,cat_feature,method='frequency_encoding',targ_var=None,num_cat=10):
    '''
    encoded nominal categorical features
    
    args:
    dat_df (dataframe) : dataframe with the categorical features
    cat_feature (strings) : name of categorical features
    method (string) : method of encoding for features with greater than num_cat categories
                      method - "frequency_encoding", "target_mean_encoding"," binary_encoding" 
    targ_var (string) : None (default), name of target variable required only when method = "target_mean_encoding"
    num_cat (int) : if no. of categories is <= num_cat then dummy variables are created else encoding method is used as
                    specified in "method"
   
    returns:
    dat_df (dataframe) with transformed categorical features
    '''
    import pandas as pd
    import category_encoders as ce

    if dat_df[cat_feature].nunique() <= num_cat :
        temp_df = pd.get_dummies(dat_df[[cat_feature]],drop_first=True)
        temp_df = temp_df.apply(lambda x: x.astype(int))
        #dat_df = dat_df.drop(cat_feature,axis=1)
        dat_df = pd.concat([dat_df,temp_df],axis=1)
        print(f'{cat_feature} - One Hot Encoding ')
    else:            
        if method == "frequency_encoding":
            freq = dat_df[cat_feature].value_counts()/len(dat_df)
            dat_df[cat_feature+"_encoding"]=dat_df[cat_feature].map(freq)
            dat_df[cat_feature+"_encoding"] = dat_df[cat_feature+"_encoding"].astype(float)
            print(f'{cat_feature} - Frequency Encoding ')
            
        if method == "target_mean_encoding":
            targ_mean = dat_df.groupby(cat_feature)[targ_var].mean()
            dat_df[cat_feature+"_encoding"]=dat_df[cat_feature].map(targ_mean)
            dat_df[cat_feature+"_encoding"] = dat_df[cat_feature+"_encoding"].astype(float)
            print(f'{cat_feature} - Mean Target Encoding ')
            
        if method == "binary_encoding":
            encoder = ce.BinaryEncoder(cols=[cat_feature])
            temp_df = encoder.fit_transform(dat_df[cat_feature])   
            temp_df = temp_df.apply(lambda x: x.astype(int))  
            #dat_df = dat_df.drop(cat_feature,axis=1)
            dat_df = pd.concat([dat_df,temp_df],axis=1)
            print(f'{cat_feature} - Binary Encoding ')
                    
    return dat_df


