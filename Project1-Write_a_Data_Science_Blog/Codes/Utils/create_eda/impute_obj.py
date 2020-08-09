def impute_missing_object(col):
    '''
    INPUT 
    dat_df : dataframe where missing values are to be imputed
    cat_varlist : list of object features to be imputed
    
    OUTPUT
    dat_df : dataframe with imputed values
    '''

    if col.isnull().sum()>0:
        col =col.fillna("missing")
    #print(dat_df[var].unique())
        
    return col
