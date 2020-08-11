def run_kmeans_exp(df,to_scale=True):
	    '''
    Runs KMEANS algorithm on data for different number of clusters and plots the within 
    sum of squares and silhouette scores.

    Args:
    df (dataframe) - data on whcih KMEANS is run
    to_scale (boolean) - True by default
                         If False data is not scaled before running KMEANS

    '''

    # Import KMeans
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


    if to_scale:
        scaler =  MinMaxScaler() 
        scaler =  scaler.fit(np.array(df))
        df_scaled=       pd.DataFrame(scaler.transform(df))
    else:
        df_scaled=df

    ks = range(2, 10)
    inertias = []
    silhouette_avg = []

    for k in ks:

        np.random.seed(1234)
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(df_scaled)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

        # Append the silhouette score to the list of silhouette score  
        cluster_labels = model.fit_predict(df_scaled)
        silhouette_avg.append(silhouette_score(df_scaled, cluster_labels))


    fig, axes = plt.subplots(1,2)    
    # Plot ks vs inertias
    axes[0].plot(ks, inertias, '-o')
    axes[0].set_xlabel('number of clusters, k')
    axes[0].set_ylabel('inertia')
    axes[0].set_xticks(ks)
    if df.shape[1]==1:
        axes[0].set_title(df.columns[0])

    # Plot ks vs silhouette score
    axes[1].plot(ks, silhouette_avg, '-o')
    axes[1].set_xlabel('number of clusters, k')
    axes[1].set_ylabel('silhouette score')
    axes[1].set_xticks(ks)
    if df.shape[1]==1:
        axes[1].set_title(df.columns[0])
    plt.tight_layout()
    plt.show()


def bin_with_kmeans(df,k,to_scale=True):
    '''
    Fit KMEANs to the feature to be binned
    Plot distribution of feature for each cluster
    Bin feature based on cluster min and max values
    
    '''
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    
    # select cluster number based on inertia and silhoette distance plots
    if to_scale:
        scaler =  MinMaxScaler() 
        scaler =  scaler.fit(np.array(df))
        df_scaled=       pd.DataFrame(scaler.transform(df))
    else:
        df_scaled=df

    # set seed
    np.random.seed(1234)
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to samples
    model.fit(df_scaled)
    # get the column names
    col = df.columns.values
    # assign observations to clusters
    df['cluster'] =  model.fit_predict(df_scaled)
    # get cluster count
    n_cluster = df.cluster.nunique()

    # plot distribution for each cluster
    fig, axes = plt.subplots(n_cluster,1,figsize=(15,8))
    cluster_df = pd.DataFrame(columns=['cluster','min','max','n_obs'])
    for i,ax in enumerate(axes) :

        temp_df =df.loc[df.cluster == i]
        sns.distplot( temp_df[col[0]], ax = ax, kde=False)

        #print(df.loc[df.cluster == i,col[0]].min(),df.loc[df.cluster == i,col[0]].max())
        x= pd.DataFrame({'cluster':[i],'min':[temp_df[col[0]].min()],'max':[temp_df[col[0]].max()],'n_obs':[temp_df.shape[0]]})
        cluster_df = pd.concat([cluster_df,x])
        diff = df[col[0]].max()/20
        #print(diff)
        ax.set_xticks(np.linspace(0, df[col[0]].max()+diff , num= 20))
        #print(np.linspace(0, df[col[0]].max()+diff , num= 20))
        ax.set_title(col[0] + " Cluster "+str(i))
        plt.tight_layout()

    plt.show()
    cluster_df = cluster_df.sort_values(by='min').reset_index(drop=True)
    #print(cluster_df)

    for i in range(len(cluster_df)):    
        if i>0:
            cluster_df.loc[i,'min'] = cluster_df.loc[i-1,'max']+0.001

        df.loc[ (df[col[0]] >= cluster_df.loc[i,"min"]) & (df[col[0]] <= cluster_df.loc[i,"max"]), col[0]+"_bin"] = \
        "["+str(np.round(cluster_df.loc[i,"min"],2))+"-"+str(np.round(cluster_df.loc[i,"max"],2))+"]"
    
    df[col[0]+"_bin"] = df[col[0]+"_bin"].astype('category')
    print("feature cut-offs:\n",cluster_df)
    print(df.head())

    return df[col[0]+"_bin"]




def bin_with_quantiles(df,var,q):
    '''
    Plot distribution of feature for each cluster
    Bin feature based on cluster min and max values
    
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    if df.columns.isin([var+'_qbin']):
        df.drop([var+'_qbin'],axis=1,inplace=True)
    df[var+'_qbin']= pd.cut(df[var],bins=q,duplicates='drop').astype('category')
    q_list = df[var+'_qbin'].unique()

    # plot distribution for each cluster
    # fig, axes = plt.subplots(q,1,figsize=(15,8))
    # q_df = pd.DataFrame(columns=['quantile','min','max','n_obs'])
    # for i,ax in enumerate(axes) :

    #     temp_df =df.loc[df[var+'_qbin'] == q_list[i]]
    #     sns.distplot( temp_df[var], ax = ax)

    #     #print(df.loc[df.cluster == i,col[0]].min(),df.loc[df.cluster == i,col[0]].max())
    #     x= pd.DataFrame({'quartile':q_list[i],'min':[temp_df[var].min()],'max':[temp_df[var].max()],'n_obs':[temp_df.shape[0]]})
    #     q_df = pd.concat([q_df,x])
    #     diff = df[var].max()/20
    #     #print(diff)
    #     ax.set_xticks(np.linspace(0, df[var].max()+diff , num= 20))
    #     #print(np.linspace(0, df[col[0]].max()+diff , num= 20))
    #     ax.set_title(var + " quantile "+str(q_list[i]))
    #     plt.tight_layout()

    # plt.show()
 
    return df[var+"_qbin"]

