def run_ordinal_encoding(df,ordinal_cat_features, map_list):
    
    from Utils.create_eda import encode_ordinal_cat as eoc

    print("--------------------------------------------------")    
    print('Encoding ordinal categorical features')
    print("--------------------------------------------------")    
    print("--------------------------------------------------")    
    print(ordinal_cat_features)
    print("--------------------------------------------------")    


    # just keep the values in column names
    #ordinal_cat_features = ordinal_cat_features[ordinal_cat_features in df.columns.values]

    if len(ordinal_cat_features) > 0:
        
        # ordinal categorical feature encoding
        for i,var in enumerate(ordinal_cat_features):
            df = eoc.encode_ordinal_cat(df,var,map_list[i])

    return df
