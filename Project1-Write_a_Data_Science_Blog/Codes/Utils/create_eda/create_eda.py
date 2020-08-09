def create_eda(dat_df):
    
    '''
    ---------------------------------------------------------------------
	Creates a summary report of data
	---------------------------------------------------------------------

    Args-    
    	dat_df (dataframe) : data on which summary report is created
             
    Returns -     
    	eda_df (dataframe) : data summary
            
    '''
    
    summary_df = dat_df.describe(include="all").transpose()
    summary_df['unique'] = dat_df.nunique(axis=0)
    summary_df.insert(0,'col_type',dat_df.dtypes)
    summary_df.insert(1,'missing_vals',dat_df.isnull().sum())
    return(summary_df)


