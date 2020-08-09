def cont_var_univariate_analysis(temp_df,var,logscale=False):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    # Set default Seaborn style
    sns.set()
    
    fig_dims = (20, 5)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=fig_dims)

    # Vizualize with histogram
    if logscale==True:
        sns.distplot(np.log(temp_df[var]+0.1),ax=ax1,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"},\
            kde_kws={"color": "grey", "lw": 1, "label": "KDE"},\
            kde=True)    
        ax1.set(xlabel="log("+var+")")
        ax1.set(ylabel='Probability')
        x_low = np.min(np.log(temp_df[var]+0.1))
        x_high = np.max(np.log(temp_df[var]+0.1))
        ax1.set_xlim([x_low,x_high])

    else:
        sns.distplot(temp_df[var],ax=ax1,hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"},\
            kde_kws={"color": "grey", "lw": 1, "label": "KDE"},\
            kde=True)    
        ax1.set(xlabel=var)
        ax1.set(ylabel='Probability')
        x_low = np.min(temp_df[var])
        x_high = np.max(temp_df[var])
        ax1.set_xlim([x_low,x_high])

        #print(x_low,x_high)

    # Vizualize the ECDF
    from empiricaldist import Cdf
    if logscale==True:
        cdf_df = Cdf.from_seq(np.log(temp_df[var]+0.1)).to_frame().reset_index()
        # Plot it
        ax2.plot(cdf_df.iloc[:,[0]].values,cdf_df.iloc[:,[1]].values)
        ax2.set(xlabel="log("+var+")")
        ax2.set(ylabel='CDF')

        x_low = np.min(cdf_df.iloc[:,[0]].values)
        x_high = np.max(cdf_df.iloc[:,[0]].values)
        ax2.set_xlim([x_low,x_high])
        plt.show()
    else:
        cdf_df = Cdf.from_seq(temp_df[var]).to_frame().reset_index()
        # Plot it
        ax2.plot(cdf_df.iloc[:,[0]].values,cdf_df.iloc[:,[1]].values)
        ax2.set(xlabel=var)
        ax2.set(ylabel='CDF')

        x_low = np.min(cdf_df.iloc[:,[0]].values)
        x_high = np.max(cdf_df.iloc[:,[0]].values)
        ax2.set(xlim=[x_low,x_high])

        #print(x_low,x_high)

        plt.show()

