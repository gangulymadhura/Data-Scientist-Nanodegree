def cat_var_univariate_analysis(temp_df,var):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    
    temp_df[var] = temp_df[var].astype('category')
    temp_df[var] = temp_df[var].cat.set_categories(np.sort(temp_df[var].unique()))
    #print(temp_df.head())
    
    temp_df = temp_df[var].value_counts().to_frame().reset_index()
    #print(temp_df.head())
    temp_df.columns = [var,'probability']
    temp_df['probability']=100*temp_df['probability']/temp_df['probability'].sum()
        
    #print(temp_df)
    # Set default Seaborn style
    sns.set()    
    fig_dims = (20, 5)
    fig, ax = plt.subplots(figsize=fig_dims)

    # Vizualize with histogram
    g=sns.barplot(temp_df[var],temp_df['probability'],ax=ax, palette="coolwarm")    
    g.set_xticklabels(g.get_xticklabels(),rotation = 90,fontsize=15)

    ax.set(xlabel=var)
    ax.set(ylabel='Probability')
    ax.set(title='Probability distribution of '+var)
    plt.show()
