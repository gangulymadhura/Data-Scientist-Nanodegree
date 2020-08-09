def missing_values_plots(df,var_list,miss_var):
    '''
    This function plots frequency distributions of missing values for feature "miss_var" over all
    features specified in var_list
    
    '''
    
    from Utils.create_plots import create_countplots as pl
    from Utils.create_plots import create_distplots as ch
    import numpy as np
     
    temp_df = df.loc[df[miss_var].isnull()]
    temp_list = var_list.copy()
    var_list = [x for x in temp_list if x!= miss_var ]

    ncol =2
    nrow = int(np.floor(len(var_list)/2) + np.ceil(len(var_list)%2))
    fig_size = (15,5*nrow)
    
    print("--------------------------------------------------")    
    print('Missing value distribution for ',miss_var)
    print("--------------------------------------------------")    

    pl.create_countplots(temp_df,var_list,nrow,ncol,fig_size)
    
def create_missing_val_plots(df , var_list):

    missing_varlist = df.columns[df.isnull().sum() > 0]
    #print(missing_varlist)
    
    if (len(var_list) > 0) and (len(missing_varlist) >0) :
        for var in missing_varlist:
            missing_values_plots(df,var_list,var)


