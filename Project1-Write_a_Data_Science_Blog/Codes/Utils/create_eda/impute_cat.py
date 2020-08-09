def impute_missing_cat(col):
    '''
    INPUT 
    dat_df : dataframe where missing values are to be imputed
    cat_varlist : list of categorical features to be imputed
    
    OUTPUT
    dat_df : dataframe with imputed values
    '''
    if col.isnull().sum()>0:
        if "missing" not in col.cat.categories:
            col = col.cat.add_categories("missing")
        col = col.fillna("missing")
        #print(dat_df[var].unique())
        
    return col
