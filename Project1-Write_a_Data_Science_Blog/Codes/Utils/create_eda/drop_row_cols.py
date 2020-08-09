def drop_row_cols(df,drop_varlist=None,dat_float_lst=None,dat_int_lst=None,dat_cat_lst=None,dat_obj_lst=None):
    '''
    This function deos the following -
    1. drops unwanted columns from the data
    2. drops exact duplicate rows
    3. drops numeric columns with 0 standard deviation
    4. drops string/categorical columns with 1 unique value
    
    '''
    
    from Utils.create_eda import drop_duplicate_records as dd
    from Utils.create_eda import drop_missing as dm
    from Utils.create_eda import drop_constant_num as dcn
    from Utils.create_eda import drop_constant_str as dcs
    from Utils.create_eda import drop_columns as dc


    # drop unwanted columns
    if drop_varlist is not None:
        df = dc.drop_columns(df,drop_varlist)  
        print("--------------------------------------------------")    
        print('Columns droped : ',len(drop_varlist))
        print(drop_varlist)
        print("--------------------------------------------------")    


    # drop duplicates
    nrows = df.shape[0]
    df= dd.dup_records(df)
    print("--------------------------------------------------")    
    print('Duplicate rows droped : ', nrows- df.shape[0])
    print("--------------------------------------------------")    
    

    # drop columns with 100% missing values
    ncols = df.shape[1]
    df = dm.drop_missing(df)
    print("--------------------------------------------------")    
    print('Columns with 100% missing values droped : ', ncols- df.shape[1])
    print("--------------------------------------------------")    


    # drop features with constant numeric values
    if dat_float_lst is not None:
        ncols = df.shape[1]
        df = dcn.drop_constant_num(df,dat_float_lst)
        print("--------------------------------------------------")    
        print('Columns with constant numeric values droped : ', ncols- df.shape[1])
        print("--------------------------------------------------")    

        
    # drop features with constant int values
    if dat_int_lst is not None:
        ncols = df.shape[1]
        df = dcn.drop_constant_num(df,dat_int_lst)
        print("--------------------------------------------------")    
        print('Columns with constant integer values droped : ', ncols- df.shape[1])
        print("--------------------------------------------------")    



    # drop features with constant string values
    if dat_obj_lst is not None:
        ncols = df.shape[1]
        df = dcs.drop_constant_str(df,dat_obj_lst)
        print("--------------------------------------------------")    
        print('Columns with constant string values droped : ', ncols- df.shape[1])
        print("--------------------------------------------------")    

    
    # drop features with constant category values
    if dat_cat_lst is not None:
        ncols = df.shape[1]
        df = dcs.drop_constant_str(df,dat_cat_lst)
        print("--------------------------------------------------")    
        print('Columns with constant categorical values droped : ', ncols- df.shape[1])
        print("--------------------------------------------------")    

    
    return df