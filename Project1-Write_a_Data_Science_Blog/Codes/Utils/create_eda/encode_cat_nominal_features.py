def run_nominal_encoding(df,nominal_cat_features,method='frequency_encoding',targ_var=None,num_cat=10):
    
    from Utils.create_eda import encode_nominal_cat as enc

    print("--------------------------------------------------")    
    print('Encoding nominal categorical features')
    print("--------------------------------------------------")    
    print("--------------------------------------------------")    
    print(nominal_cat_features)
    print("--------------------------------------------------")    

    # just keep the values in column names
    nominal_cat_features = nominal_cat_features[nominal_cat_features.isin(df.columns.values)]

    # nominal categorical feature encoding
    for var in nominal_cat_features:
        df = enc.encode_nominal_cat(df,var,method,targ_var)
        
    return df
