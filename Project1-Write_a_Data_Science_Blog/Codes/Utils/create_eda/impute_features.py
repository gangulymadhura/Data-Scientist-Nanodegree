# imputing continuous features
def impute_cont_features(df,cont_varlist,cat_varlist):

    '''
    For each continuous feature in "cont_varlist" this function finds the categorical feature in "cat_varlist" 
    that best explains the variation in this feature -> impute_cat_car

    The missing values in the continuous feature is imputed with the median value within corresponding 
    categories of impute_cat_car
 
    '''
    from scipy.stats import f_oneway
    from scipy import stats
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    high_missrate_varlist =[]
    impute_df = pd.DataFrame(columns =['imputed_feature','miss_rate','cat_feature','Oneway_F_stat_pval','KS_stat_pval'])
    
    for cont_var in cont_varlist:
        miss_rate = df[cont_var].isnull().sum()/len(df[cont_var])
        #print(cont_var,miss_rate)
        if miss_rate > 0 and miss_rate <= 0.9 :
            fstats_vals={}

            for cat_var in cat_varlist:
                if (df.loc[df[cont_var].isnull()].index).equals(df.loc[df[cat_var]=='missing'].index):  
                    #print(f'{cont_var} has {np.round(100*miss_rate,4)}% miss-rate, not able to impute as categorical variable is missing')             
                    continue
                elif df.loc[df[cont_var].isnull(),cat_var].isin(df.loc[~df[cont_var].isnull(),cat_var]).sum()==0:  
                    #print(f'{cont_var} has {np.round(100*miss_rate,4)}% miss-rate, not able to impute as category is unique to missing values')             
                    continue
                else:
                    temp_df = df.loc[~df[cont_var].isnull()]
                    fstats = f_oneway(*[temp_df[cont_var][temp_df[cat_var] == x] for x in temp_df[cat_var].unique().tolist()])
                    fstats_vals[cat_var] = fstats.pvalue

            #print(fstats_vals)
            if bool(fstats_vals) :
                fstats_vals = pd.DataFrame(pd.Series(fstats_vals),columns=['pval']).sort_values(by='pval',ascending=True)
                impute_cat_car = fstats_vals.index[0]
                impute_cat_pval = fstats_vals.pval[0]

                #print(fstats_vals)
                if impute_cat_pval >0:

                    # impute missing values in cont_var with median values within categories of impute_cat_car
                    cat_var_summary = temp_df.groupby(impute_cat_car)[cont_var].median()
                    missing_df = df.loc[df[cont_var].isnull(),:]
                    missing_df.loc[:,cont_var] = missing_df[impute_cat_car].map(cat_var_summary)

                    df = pd.concat([temp_df,missing_df],ignore_index=True)
                    #print(cont_var, impute_cat_car, df[cont_var].isnull().sum())
                    
                    # check for distribution changes due to imputation            
                    # if KS distnace between the probability distributions of the feature
                    # before and after is small then no change is expected
                    
                    ks = stats.ks_2samp(temp_df[cont_var].to_numpy(), df[cont_var].to_numpy())
                    # ks_list[cont_var] = ks.pvalue
                    
                    #print(cont_var, impute_cat_car, df[cont_var].isnull().sum(), fstats_vals.fstatistic[0], ks.pvalue)
                    t=pd.DataFrame({'imputed_feature':[cont_var],'miss_rate':[miss_rate],'cat_feature':[impute_cat_car],'Oneway_F_stat_pval': [impute_cat_pval] ,'KS_stat_pval': [ks.pvalue] })
                    impute_df = pd.concat([impute_df,t])
                    print(f'{cont_var} has {np.round(100*miss_rate,4)}% miss-rate, imputation done based on {impute_cat_car}')
            else:
                print(f'{cont_var} has {np.round(100*miss_rate,4)}% miss-rate, not able to impute')

        elif miss_rate == 0:
            print(f'{cont_var} has 0% miss-rate, no imputation needed')

        else:
            print(f'{cont_var} has more than 90% miss-rate, needs further investigation for imputation')
            high_missrate_varlist.append(cont_var)

    #print(impute_df.head(3))                             
    return df,impute_df,high_missrate_varlist
        
        
def impute_all_features(df,output_filepath,file_suffix,dat_cat_lst,dat_obj_lst,dat_float_lst,dat_int_lst):
    '''
    This function imputes missing values -
    
    for categorical/ string type features - replaces missing values with the text "missing"
    for float/ int type features - 
        finds the categorical feature that explains maximum variation (with highest one-way ANOVA F-statistic)
        replaces missing value with feature median by that categorical feature
        
    '''
        
    from Utils.create_eda import impute_cat as ic
    from Utils.create_eda import impute_obj as io
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    impute_df_float=pd.DataFrame()
    impute_df_int=pd.DataFrame()
    high_missrate_float =[]
    high_missrate_int=[]

    # impute missing values in categorical variables with "missing" text
    if dat_cat_lst is not None:
        
        print("--------------------------------------------------")    
        print('Running missign value imputation on categorical features')
        print("--------------------------------------------------")    

        varlist = dat_cat_lst[dat_cat_lst.isin(df.columns.values)]
        print("--------------------------------------------------")    
        print(varlist)
        print("--------------------------------------------------")    
        
        df[varlist] = df[varlist].apply(ic.impute_missing_cat,0)
        # check
        print("Missing value summary :\n",df[varlist].isnull().sum())


    if dat_obj_lst is not None:
        print("--------------------------------------------------")    
        print('Running missign value imputation on object features')
        print("--------------------------------------------------")    

        varlist = dat_obj_lst[dat_obj_lst.isin(df.columns.values)]
        print("--------------------------------------------------")    
        print(varlist)
        print("--------------------------------------------------")    

        df[varlist] = df[varlist].apply(io.impute_missing_object,0)
        # check
        print("Missing value summary :\n",df[varlist].isnull().sum())

        
    if dat_float_lst is not None:
        print("--------------------------------------------------")    
        print('Running missign value imputation on numeric features')
        print("--------------------------------------------------")    

        cont_varlist = dat_float_lst[dat_float_lst.isin(df.columns.values)].reset_index(drop=True)
        print(cont_varlist)
        print("--------------------------------------------------")    
        #print(cont_varlist)
        print("--------------------------------------------------")    

        #print("***")
        cat_varlist = dat_cat_lst[dat_cat_lst.isin(df.columns.values)].reset_index(drop=True)
        #print(cat_varlist)
        df,impute_df_float,high_missrate_float =  impute_cont_features(df,cont_varlist,cat_varlist)
        #print("***")

        # check
        print("Missing value summary :\n",df[cont_varlist].isnull().sum())

        
    if dat_int_lst is not None:
        print("--------------------------------------------------")    
        print('Running missign value imputation on integer features')
        print("--------------------------------------------------")    

        cont_varlist = dat_int_lst[dat_int_lst.isin(df.columns.values)].reset_index(drop=True)
        print("--------------------------------------------------")    
        print(cont_varlist)
        print("--------------------------------------------------")    

        cat_varlist = dat_cat_lst[dat_cat_lst.isin(df.columns.values)].reset_index(drop=True)
        df,impute_df_int,high_missrate_int =  impute_cont_features(df,cont_varlist,cat_varlist)
        # check
        print("---------------------------------------------------")
        print("Missing value summary :\n",df[cont_varlist].isnull().sum())
        print("---------------------------------------------------")


    # review f stats for float features   
    if len(impute_df_float)>0:
        plt.figure(figsize=(20,5))
        sns.barplot(x=impute_df_float.cat_feature,y=impute_df_float.Oneway_F_stat_pval,ci=None)
        plt.title("FStatistic p-values")
        plt.show()
        impute_df_float.to_csv(output_filepath+file_suffix+"_impute_df_float.csv")
            
    # review KS stats for float features
    if len(impute_df_float)>0:
        plt.figure(figsize=(20,5))
        sns.barplot(x=impute_df_float.cat_feature,y=impute_df_float.KS_stat_pval,ci=None)
        plt.title("KS p-values")
        plt.show()

        #print(impute_df_float)

    # review f stats for int features
    if len(impute_df_int)>0:
        plt.figure(figsize=(20,5))
        sns.barplot(x=impute_df_int.cat_feature,y=impute_df_int.Oneway_F_stat_pval,ci=None)
        plt.title("FStatistic p-values")
        plt.show()
        impute_df_int.to_csv(output_filepath+file_suffix+"_impute_df_int.csv")
        
    # review KS stats for int features
    if len(impute_df_int)>0:
        plt.figure(figsize=(20,5))
        sns.barplot(x=impute_df_int.cat_feature,y=impute_df_int.KS_stat_pval,ci=None)
        plt.title("KS p-values")
        plt.show()

       
    return df,impute_df_float,impute_df_int,high_missrate_float,high_missrate_int

