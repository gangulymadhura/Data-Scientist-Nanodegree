def drop_columns(dat_df,var_list):
    '''
    INPUT
    dat_df : dataframe where duplicate records are searched for
    
    OUTPUT
    dat_df : dataframe without duplicates
    
    '''

    if len(var_list) > 0:
        l= dat_df.shape[1]
        print(f'There are {l} columns in the dataframe')
        dat_df = dat_df.drop(var_list,axis=1)
        print(f'Dropped {l-dat_df.shape[1]} columns')

    return dat_df