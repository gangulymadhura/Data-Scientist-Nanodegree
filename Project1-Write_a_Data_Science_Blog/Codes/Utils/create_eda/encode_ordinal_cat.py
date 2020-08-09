def encode_ordinal_cat(dat_df,cat_feature,value_map):
    '''
    encoded ordinal categorical features
    
    args:
    dat_df (dataframe) : dataframe with the categorical features
    cat_feature (strings) : name of categorical features
    value_map (dictionary) : map of category to numeric values
   
    returns:
    dat_df (dataframe) with transformed categorical features
    '''
    import pandas as pd
                    
    dat_df[cat_feature+"_encoding"] = dat_df[cat_feature].map(value_map)
    dat_df[cat_feature+"_encoding"] = dat_df[cat_feature+"_encoding"].astype(float)
    print(f'{cat_feature} - Ordinal label Encoding ')
    
    return dat_df


