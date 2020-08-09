def drop_missing(dat_df):
    '''
    INPUT 
    dat_df : dataframe from which columns with 100% missing values and constant values are to be removed
    
    OUTPUT
    dat_df : dataframe after dropping columns with 100% missing values and constant values are to be removed
    '''   
    
    miss_var = dat_df.columns[dat_df.isnull().sum() == len(dat_df)]   
    dat_df.drop(miss_var,axis=1,inplace=True)
    print('Columns dropped due to 100% miss rate : \n')
    print(miss_var)

    return dat_df