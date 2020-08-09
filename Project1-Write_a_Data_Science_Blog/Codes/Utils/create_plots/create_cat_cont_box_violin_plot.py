def cat_cont_bivariate_analysis(temp_df,cat_var,cont_var):  

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import f_oneway
    from scipy import stats
    
    temp_df[cat_var] = temp_df[cat_var].astype('category')
    temp_df[cat_var] = temp_df[cat_var].cat.set_categories(np.sort(temp_df[cat_var].unique()))

    # fig, (ax1,ax2) = plt.subplots(2,1,figsize=(20,12))
    
   # # Vizualize with a violin plot
   #  sns.violinplot(data=temp_df,x=cat_var, y=cont_var, inner=None,hue=cat_var,ax=ax1)    
   #  sns.despine(left=True, bottom=True)
   #  plt.xlabel(cat_var)
   #  plt.ylabel(cont_var)
   #  plt.title(cont_var+' by '+cat_var)
    #plt.show()
    
   # Vizualize with a box plot
    plt.figure(figsize=(15,6))
    g=sns.boxplot(data=temp_df,x=cat_var,y=cont_var,hue=cat_var, showfliers = False,palette='coolwarm') 
    plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    plt.title(cont_var+' by '+cat_var)        
    plt.show()
    
    # Anova to check equality of means
    #print(rp.summary_cont(temp_df[cont_var].groupby(temp_df[cat_var])))
    print(stats.f_oneway(*[temp_df[cont_var][temp_df[cat_var] == x] for x in temp_df[cat_var].unique().tolist()]))
    

