def dup_records(dat_df):
    '''
    INPUT
    dat_df : dataframe where duplicate records are searched for
    
    OUTPUT
    dat_df : dataframe without duplicates
    
    '''

    if dat_df.duplicated().sum() > 0:
        print("There are duplicate records")
        l=len(dat_df)
        dat_df= dat_df.drop_duplicates()
        print(f'Dropped {l-len(dat_df)} records')

    return dat_df