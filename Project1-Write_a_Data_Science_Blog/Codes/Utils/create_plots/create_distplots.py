def create_hist(dat_df,chart_varlist,nrow,ncol,fig_size):
    '''
    INPUT -
    dat_df : dataframe containing the data to plot
    chart_varlist : list of column names for which hitogram is created
    nrow  : number of rows in the plotting grid
    ncol  : number of columns in the plotting grid
    fig_size : size of figures

    OUTPUT -
    plots the histogram for the column names passed in chart_varlist

    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools


    if ncol == 1 :
        fig, axes = plt.subplots(nrows = nrow,figsize = fig_size)
        palette = itertools.cycle(sns.color_palette("Paired"))
        k=0
        for i in range(nrow):

            g=sns.distplot(dat_df[chart_varlist[k]], ax = axes[i], color=next(palette))
            plt.tight_layout()
            g.set_xlabel(chart_varlist[k],fontsize=25)
            g.set_ylabel('Distribution',fontsize=25)
            if k == len(chart_varlist)-1 :
                break
            else:
                k+=1

    if nrow == 1 :
        fig, axes = plt.subplots(ncols = ncol,figsize = fig_size)
        palette = itertools.cycle(sns.color_palette("Paired"))
        k=0
        for i in range(ncol):

            g=sns.countplot(dat_df[chart_varlist[k]], ax = axes[i], color=next(palette), order = dat_df[chart_varlist[k]].value_counts().index)
            plt.tight_layout()
            g.set_xlabel(chart_varlist[k],fontsize=25)
            g.set_ylabel('Distribution',fontsize=25)

            if k == len(chart_varlist)-1 :
                break
            else:
                k+=1

    if nrow != 1 and ncol != 1 :
        fig, axes = plt.subplots(nrows=nrow, ncols = ncol,figsize = fig_size)
        palette = itertools.cycle(sns.color_palette("Paired"))
        k=0
        for i in range(nrow):
            for j in range(ncol):

                g=sns.countplot(dat_df[chart_varlist[k]], ax = axes[i,j], color=next(palette), order = dat_df[chart_varlist[k]].value_counts().index)
                plt.tight_layout()
                g.set_xlabel(chart_varlist[k],fontsize=25)
                g.set_ylabel('Distribution',fontsize=25)


                if k == len(chart_varlist)-1 :
                    break
                else:
                    k+=1
