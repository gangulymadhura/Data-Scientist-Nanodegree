def corr(x, y, **kwargs):
    '''
    computes correlation between x and y

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    # Calculate the value
    coef = np.corrcoef(x, y)[0][1]
    # Make the label
    label = r'$\rho$ = ' + str(round(coef, 2))
    #print(label)
    
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy = (0.2, 0.95), size = 20, xycoords = ax.transAxes)


    
def cont_cont_bivariate_analysis(temp_df,cont_varlist):
  '''
  computes pair plot of features in "cont_varlist"

  '''
  import seaborn as sns
  import matplotlib.pyplot as plt

  g=sns.pairplot(temp_df, 
               vars=cont_varlist,
               kind="scatter",
               diag_kind='hist'
               )
  g=g.map_lower(corr)
  plt.title("Scatter plot with correlation")
  plt.show()


    
# Overlay pairplot with a categorical variable
def cont_cont_bivariate_analysis_catoverlay(temp_df,cont_varlist,cat_var=None):
  '''
  computes pair plot of features in "cont_varlist" with hue as "cat_var"

  '''
  import seaborn as sns
  import matplotlib.pyplot as plt

  g=sns.pairplot(temp_df, 
                 vars=cont_varlist,
                 kind="scatter", hue=cat_var, 
                 diag_kind='hist',
                 plot_kws = {'alpha': 0.2, 's': 50, 'edgecolor': 'k'})
  g=g.map_lower(corr)
  g.fig.suptitle("Scatter plot")
  plt.show()
