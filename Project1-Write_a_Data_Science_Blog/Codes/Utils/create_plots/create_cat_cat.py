def cat_cat_bivariate_analysis(temp_df,cat_var_1,cat_var_2,fliers=True):

        import numpy as np
        import pandas as pd   
        from scipy.stats import chi2_contingency
    
        temp_df[cat_var_1] =  temp_df[cat_var_1].astype('category')
        temp_df[cat_var_1] = temp_df[cat_var_1].cat.set_categories(np.sort(temp_df[cat_var_1].unique()))

        temp_df[cat_var_2] =  temp_df[cat_var_2].astype('category')
        temp_df[cat_var_2] = temp_df[cat_var_2].cat.set_categories(np.sort(temp_df[cat_var_2].unique()))
        
        print("-------------------------------------------------------------------------------")
        print("Checking dependence between ", cat_var_1," and ",cat_var_2)

        cr_tab= pd.DataFrame(pd.crosstab(temp_df[cat_var_1],temp_df[cat_var_2],margins = True,normalize='index'))
        #cr_tab=cr_tab.sort_values('All',ascending=False)
        #cr_tab.drop('All',axis=0,inplace=True)
        #cr_tab.drop('All',axis=1,inplace=True)
        display(cr_tab.style.background_gradient(cmap='Blues',axis=None))

        cr_tab2= pd.DataFrame(pd.crosstab(temp_df[cat_var_1],temp_df[cat_var_2],margins = True))
        #cr_tab=cr_tab.sort_values('All',ascending=False)
        # cr_tab.drop('All',axis=0,inplace=True)
        # cr_tab.drop('All',axis=1,inplace=True)
        display(cr_tab2.style.background_gradient(cmap='Blues',axis=None))

        # get 2 sample chisquare test p-value
        chi2, p, dof, ex = chi2_contingency(cr_tab, correction=False)
        print("Chi-square 2 sample test of independence p-value :",p)
        print("-------------------------------------------------------------------------------")
 
        #print(cat_var_1,cat_var_2,p)
