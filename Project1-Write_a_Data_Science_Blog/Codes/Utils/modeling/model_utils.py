
def run_decision_tree_reg(X_train,y_train,X_test,y_test,dt,is_log=False,is_scale=False,scaler_y=None):

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error  as MSE
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Fit dt to the training set
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    if is_scale:
        print("Scaling back y ")
        print("....")

        y_pred_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_train)),index=y_train.index).squeeze()
        y_pred_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_test)),index=y_test.index).squeeze()
        
        y_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_train)),index=y_train.index).squeeze()
        y_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_test)),index=y_test.index).squeeze()

    if is_log :     
        print("Taking inverse log transform ")
        print("....")    

        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)

        y_train = np.exp(y_train)
        y_test = np.exp(y_test)

    # Compute mse_dt
    mse_dt = MSE(y_train, y_pred_train)
    # Compute rmse_dt
    rmse_dt = mse_dt**(1/2)
    # Print rmse_dt
    print("Train set RMSE of dt: {:.2f}".format(rmse_dt))


    # Compute mse_dt
    mse_dt = MSE(y_test, y_pred_test)
    # Compute rmse_dt
    rmse_dt = mse_dt**(1/2)
    # Print rmse_dt
    print("Test set RMSE of dt: {:.2f}".format(rmse_dt))


    # create feature importance dataframe
    feat_importance_df = pd.DataFrame({'feature':X_train.columns,\
                                       'decision_tree_reg_importance':dt.feature_importances_}).\
                                        sort_values(by='decision_tree_reg_importance',ascending=False).\
                                        reset_index(drop=True)    
    #print("Top 10 predictors :\n")
    #print(feat_importance_df.head(5))
    # Draw a horizontal barplot of importances_sorted
    feat_importance_df.iloc[0:10].plot(y='decision_tree_reg_importance',x='feature',kind='barh', color='blue')
    plt.title('Top 10 Features Importances')
    plt.show()
        
    print(type(y_pred_train))
    y_train_df = pd.DataFrame({'y_true':y_train,'y_pred':y_pred_train,'type':'train','idx':y_train.index})
    y_train_df['err'] = y_train_df['y_true'] - y_train_df['y_pred']
    
    y_test_df = pd.DataFrame({'y_true':y_test,'y_pred':y_pred_test,'type':'test','idx':y_test.index})
    y_test_df['err'] = y_test_df['y_true'] - y_test_df['y_pred']
        
    return y_train_df,y_test_df,feat_importance_df,dt


def run_cv_scoring(X,y,model,scoring,cv=10):
    from sklearn.model_selection import cross_validate
    import numpy as np
    
    # Compute the array containing the 10-folds CV RMSEs
    np.random.seed(536)
    cv_scores = cross_validate(model, X, y, cv=cv,
                                    scoring=scoring,
                                    return_train_score=True)
    #print(cv_scores)  
    if type(scoring) == list:
        for i,scr_method in enumerate(scoring):
            
            # Print Test Scores
            cv_score_mean = np.round(cv_scores['test_'+scr_method].mean(),2)
            cv_score_std = np.round(cv_scores['test_'+scr_method].std(),2)
            
            print(f'Average test {scr_method} : {cv_score_mean}' )
            print(f'Test {scr_method} CI : {cv_score_mean} (+/- {cv_score_std}) ')
            print(f'Test {scr_method} for {cv} folds :')
            print(cv_scores['test_'+scr_method])
            print('\n')
            

            # Print Train Scores
            cv_score_mean = np.round(cv_scores['train_'+scr_method].mean(),2)
            cv_score_std = np.round(cv_scores['train_'+scr_method].std(),2)

            print(f'Average train {scr_method} : {cv_score_mean}' )
            print(f'Train {scr_method} CI : {cv_score_mean} (+/- {cv_score_std}) ')
            print(f'Train {scr_method} for {cv} folds :')
            print(cv_scores['train_'+scr_method])
            print('\n')
            
    else:
            # Print Test Scores
            cv_score_mean = np.round(cv_scores['test_score'].mean(),2)
            cv_score_std = np.round(cv_scores['test_score'].std(),2)

            print(f'Average test {scoring} : {cv_score_mean}' )
            print(f'Test {scoring} CI : {cv_score_mean} (+/- {cv_score_std}) ')
            print(f'Test {scoring} for {cv} folds :')
            print(cv_scores['test_score'])
            print('\n')

            # Print Train Scores
            cv_score_mean = np.round(cv_scores['train_score'].mean(),2)
            cv_score_std = np.round(cv_scores['train_score'].std(),2)

            print(f'Average train {scoring} : {cv_score_mean}' )
            print(f'Train {scoring} CI : {cv_score_mean} (+/- {cv_score_std}) ')
            print(f'Train {scoring} for {cv} folds :')
            print(cv_scores['train_score'])
            print('\n')

def score(y_true,y_pred,score_method):
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_log_error
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import f1_score
    import numpy as np
    
    if score_method == 'mse':
        score_val = mean_squared_error(y_true, y_pred)
    if score_method == 'rmse':
        score_val = np.sqrt(mean_squared_error(y_true, y_pred))
    if score_method == 'mae':
        score_val = mean_absolute_error(y_true, y_pred)
    if score_method == 'r2':
        score_val = r2_score(y_true, y_pred)
    if score_method == 'rmsle':
        score_val = mean_squared_log_error(y_true, y_pred)
    if score_method == 'auc':
        score_val = roc_auc_score(y_true, y_pred)        
    if score_method == 'f1':
        score_val = f1_score(y_true, y_pred)
       
    return score_val

def run_cv_scoring_2(X,y,model,scoring,cv_folds,stratify_by=None,stratify=False,\
    transform=None, is_scale=False,scaler_y=None,coef=False):

    from sklearn.model_selection import KFold
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from collections import defaultdict

    print(len(X),len(y))
    if stratify:
        cv_stratified = StratifiedShuffleSplit(n_splits=cv_folds,random_state=42)
        splitter= cv_stratified.split(X,stratify_by)
    else:
        cv = KFold(n_splits=cv_folds, random_state=42, shuffle=True)
        splitter = cv.split(X)
    
    cv_train = pd.DataFrame()
    cv_test = pd.DataFrame()

    cv_score_train= defaultdict(list)
    cv_score_test= defaultdict(list)

    i=1
    # fig,axes = plt.subplots(cv_folds, figsize=(20,10))
    # fig2,axes2 = plt.subplots(cv_folds,2, figsize=(15,15))
    # fig3,axes3 = plt.subplots(cv_folds, figsize=(20,10))

    for train_index, test_index in splitter:
        print("Fold ",i)
        
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

        # Fit dt to the training set
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        if is_scale:
            print("Scaling back y ")
            print("....")

            y_pred_train = scaler_y.inverse_transform(pd.DataFrame(y_pred_train))
            y_pred_train = pd.Series(y_pred_train.squeeze())
            y_pred_test = scaler_y.inverse_transform(pd.DataFrame(y_pred_test))
            y_pred_test = pd.Series(y_pred_test.squeeze())

            y_train = scaler_y.inverse_transform(pd.DataFrame(y_train))
            y_train = pd.Series(y_train.squeeze())
            y_test = scaler_y.inverse_transform(pd.DataFrame(y_test))
            y_test = pd.Series(y_test.squeeze())


        if transform == "log" :     
            print("Taking inverse transform ")
            print("....")    

            y_pred_train = pd.Series(np.exp(y_pred_train))
            y_pred_test = pd.Series(np.exp(y_pred_test))

            y_train = pd.Series(np.exp(y_train))
            y_test = pd.Series(np.exp(y_test))

        elif transform == "square" :     
            print("Taking inverse sqaure transform ")
            print("....")    

            y_pred_train = pd.Series(np.power(y_pred_train,(1/2)))
            y_pred_test = pd.Series(np.power(y_pred_test,(1/2)))

            y_train = pd.Series(np.power(y_train,(1/2)))
            y_test = pd.Series(np.power(y_test,(1/2)))

        elif transform == "cube" :     
            print("Taking inverse cube transform ")
            print("....")    

            y_pred_train = pd.Series(np.power(y_pred_train,(1/3)))
            y_pred_test = pd.Series(np.power(y_pred_test,(1/3)))

            y_train = pd.Series(np.power(y_train,(1/3)))
            y_test = pd.Series(np.power(y_test,(1/3)))


        elif transform == "seven" :     
            print("Taking inverse seven transform ")
            print("....")    

            y_pred_train = pd.Series(np.power(y_pred_train,(1/7)))
            y_pred_test = pd.Series(np.power(y_pred_test,(1/7)))

            y_train = pd.Series(np.power(y_train,(1/7)))
            y_test = pd.Series(np.power(y_test,(1/7)))

        elif transform == "exp" :     
            print("Taking inverse seven transform ")
            print("....")    

            y_pred_train = pd.Series(np.log(y_pred_train))
            y_pred_test = pd.Series(np.log(y_pred_test))

            y_train = pd.Series(np.log(y_train))
            y_test = pd.Series(np.log(y_test))

        #print(type(y_test))
        #print(type(y_train))
        
        # plot scatter plot of actuals versus predicted scores
        x_low = np.min([np.min(y_train),np.min(y_test)])
        x_high = np.max([np.max(y_train),np.max(y_test)])

        y_low = np.min([np.min(y_pred_train),np.min(y_pred_test)])
        y_high = np.max([np.max(y_pred_train),np.max(y_pred_test)])

        buffer =2

        # sns.scatterplot(y_train,y_pred_train,label="Train",ax=axes2[i-1,0], color="green")
        # axes2[i-1,0].set_xlim([x_low-buffer,x_high+buffer]) 
        # axes2[i-1,0].set_ylim([y_low-buffer,y_high+buffer]) 

        # sns.scatterplot(y_test,y_pred_test,label="Test",ax=axes2[i-1,1],color="blue")  
        # axes2[i-1,1].set_xlim([x_low-buffer,x_high+buffer]) 
        # axes2[i-1,1].set_ylim([y_low-buffer,y_high+buffer]) 

        #plt.legend()
        if coef:
            #plot feature importance for each fold
            feat_importance_df = pd.DataFrame({'feature':X_train.columns,\
                                           'feat_importance':model.coef_}).\
                                            sort_values(by='feat_importance',ascending=False).\
                                            reset_index(drop=True)    

        else:
            #plot feature importance for each fold
            feat_importance_df = pd.DataFrame({'feature':X_train.columns,\
                                           'feat_importance':model.feature_importances_}).\
                                            sort_values(by='feat_importance',ascending=False).\
                                            reset_index(drop=True)    

        # feat_importance_df.iloc[0:10].plot(y='feat_importance',x='feature',kind='barh',ax=axes3[i-1])
        # axes3[i-1].set_title('Fold'+str(i)+'- Top 10 Features Importances')


        for scr in scoring:
     
            if scr == 'mse':
                cv_score_train[scr].append(score(y_train, y_pred_train,'mse'))
                cv_score_test[scr].append(score(y_test, y_pred_test,'mse'))
            if scr == 'rmse':
                cv_score_train[scr].append(score(y_train, y_pred_train,'rmse'))
                cv_score_test[scr].append(score(y_test, y_pred_test,'rmse'))
            if scr == 'mae':
                cv_score_train[scr].append(score(y_train, y_pred_train,'mae'))
                cv_score_test[scr].append(score(y_test, y_pred_test,'mae'))
            if scr == 'r2':
                cv_score_train[scr].append(score(y_train, y_pred_train,'r2'))
                cv_score_test[scr].append(score(y_test, y_pred_test,'r2'))
            if scr == 'rmsle':
                cv_score_train[scr].append(score(y_train, y_pred_train,'rmsle'))
                cv_score_test[scr].append(score(y_test, y_pred_test,'rmsle'))


        cv_test_y = pd.DataFrame({'fold':[i],'test_mean':[y_test.mean()],'test_median':[y_test.median()],'test_std':[y_test.std()]})
        cv_train_y = pd.DataFrame({'fold':[i],'train_mean':[y_train.mean()],'train_median':[y_train.median()],'train_std':[y_train.std()]})

        for scr in scoring:
            cv_train_y.loc[0,scr] =cv_score_train[scr][i-1]
            cv_test_y.loc[0,scr] =cv_score_test[scr][i-1]

        cv_test = pd.concat([cv_test,cv_test_y])
        cv_train = pd.concat([cv_train,cv_train_y])

        # sns.distplot(y_train,label="Train",ax=axes[i-1],bins=20)
        # sns.distplot(y_test,label="Test",ax=axes[i-1],bins=20)   
        # axes[i-1].legend()     

        i=i+1

    cv_all ={}
    for scr in scoring:
        cv_score_mean_tr = np.round(np.mean(cv_score_train[scr]),2)
        cv_score_std_tr = np.round(np.std(cv_score_train[scr]),2)

        print(f'Average train {scr} : {cv_score_mean_tr}' )
        print(f'Train {scr} CI : {cv_score_mean_tr} (+/- {cv_score_std_tr}) ')
        print(f'Train {scr} for {cv_folds} folds :')
        print(cv_score_train[scr])
        print('\n')

        # Print Train Scores
        cv_score_mean_ts = np.round(np.mean(cv_score_test[scr]),2)
        cv_score_std_ts = np.round(np.std(cv_score_test[scr]),2)

        print(f'Average test {scr} : {cv_score_mean_ts}' )
        print(f'Test {scr} CI : {cv_score_mean_ts} (+/- {cv_score_std_ts}) ')
        print(f'Test {scr} for {cv_folds} folds :')
        print(cv_score_test[scr])
        print('\n')

        cv_all["train_"+scr] = [cv_score_mean_tr]
        cv_all["test_"+scr] = [cv_score_mean_ts]

    print("Train data summary by folds")
    print(cv_train)
    print("\n")
    print("Test data summary by folds")
    print(cv_test)

    print("Train and Test y distribution")
    #plt.show()

    plt.show()
    return cv_train,cv_test,cv_all


def run_grid_search_2(X_train,y_train,params_set,scoring_set,cv_folds,stratify_by=None,\
    stratify=False,transform=None,is_scale=False,scaler_y=None):
    
    from sklearn.model_selection import ParameterGrid
    import pandas as pd
    import xgboost as xgb

    param_list = list(ParameterGrid(params_set)) 
    
    # performance df
    perf_df = pd.DataFrame()

    for par in param_list:       
        print(par)
        model = xgb.XGBRegressor(**par)
        
        print(model)
        cv_train,cv_test,cv_all = run_cv_scoring_2(X_train,y_train.squeeze(),\
                                                               model=model,scoring=scoring_set,\
                                                               cv_folds=cv_folds,stratify_by=stratify_by,\
                                                               stratify= stratify,transform=transform,
                                                               is_scale = is_scale,scaler_y= scaler_y)
                                                               
        
        par = {x:[par[x]] for x in par.keys()}
        temp_df = pd.concat([pd.DataFrame(par),pd.DataFrame(cv_all)],axis=1)
        #print("temp :",temp_df)
        perf_df = pd.concat([perf_df,temp_df])
        #print("Perf: ",perf_df.shape)
        
    return perf_df

def run_grid_search(X_train,y_train,model,params_dt,cv_folds,scoring):

    # Import GridSearchCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

    # Instantiate grid_dt
    grid_dt = GridSearchCV(estimator=model,
                           param_grid=params_dt,
                           scoring= scoring,
                           cv=cv_folds,
                           n_jobs=-1)
    # fir the grid
    g = grid_dt.fit(X_train, y_train)

    # get the best model
    best_model = g.best_estimator_
    print("The best model is :")
    print(best_model)
    
    # get performance for each fold
    params = g.cv_results_['params']
    cv_score_mean = g.cv_results_['mean_test_score']
    cv_score_std = g.cv_results_['std_test_score']
    
    return best_model

def error_plot(X_train,y_train,err_train,X_test,y_test,err_test):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    # get x and y limits
    x_low = np.min([np.min(y_train),np.min(y_test)])
    x_high = np.max([np.max(y_train),np.max(y_test)])

    y_low = np.min([np.min(err_train),np.min(err_test)])
    y_high = np.max([np.max(err_train),np.max(err_test)])

    buffer = 2

    fig,axes = plt.subplots(1,2,figsize=(15,4))
    sns.scatterplot(x=y_train,y=err_train,ax=axes[0])
    axes[0].set_title('train target vs train error')
    axes[0].set_xlim([x_low-buffer,x_high+buffer])   
    axes[0].set_ylim([y_low-buffer,y_high+buffer])
 
    sns.scatterplot(x=y_test,y=err_test,ax=axes[1])
    axes[1].set_title('test target vs test error')
    axes[1].set_xlim([x_low-buffer,x_high+buffer])   
    axes[1].set_ylim([y_low-buffer,y_high+buffer]) 
    plt.show()
        
    for col in X_train.columns:
        x_low = np.min([np.min(X_train[col]),np.min(X_test[col])])
        x_high = np.max([np.max(X_train[col]),np.max(X_test[col])])

        fig,axes = plt.subplots(1,2,figsize=(15,4))
        sns.scatterplot(x=X_train[col],y=err_train,ax=axes[0])
        axes[0].set_title(f'{col} vs train error')
        axes[0].set_xlim([x_low-buffer,x_high+buffer])   
        axes[0].set_ylim([y_low-buffer,y_high+buffer])

        sns.scatterplot(x=X_test[col],y=err_test,ax=axes[1])
        axes[1].set_title(f'{col} vs test error')
        axes[1].set_xlim([x_low-buffer,x_high+buffer])   
        axes[1].set_ylim([y_low-buffer,y_high+buffer])

        plt.show()

def error_barplot(X_train,y_train,err_train,X_test,y_test,err_test):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
 
    # fig,axes = plt.subplots(1,2,figsize=(15,4))
    # sns.barplot(x=y_train,y=err_train,ax=axes[0],estimator=np.sum)
    # axes[0].set_title('train target vs train error')
    # sns.barplot(x=y_test,y=err_test,ax=axes[1],estimator=np.sum)
    # axes[1].set_title('test target vs test error')
    # plt.show()

    for col in X_train.columns:
        fig,axes = plt.subplots(1,2,figsize=(15,4))
        sns.barplot(x=X_train[col],y=err_train,ax=axes[0],estimator=np.sum)
        axes[0].set_title(f'{col} vs test error')
        sns.barplot(x=X_test[col],y=err_test,ax=axes[1],estimator=np.sum)
        axes[1].set_title(f'{col} vs test error')
        plt.show()

def error_distplot(X_train,err_train,X_test,err_test):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
 
    # get x and y limits
    x_low = np.min([np.min(err_train),np.min(err_test)])
    x_high = np.max([np.max(err_train),np.max(err_test)])
    buffer=2

    fig,axes = plt.subplots(1,2,figsize=(15,4))
    sns.distplot(err_train,ax=axes[0])
    axes[0].set_title('train error distribution')
    axes[0].set_xlim([x_low-buffer,x_high+buffer])   
 
    sns.distplot(err_test,ax=axes[1])
    axes[1].set_title('test error distribution')
    axes[1].set_xlim([x_low-buffer,x_high+buffer])   
 
    plt.show()

    cols = X_train.columns.values
    X_train['type']="train"; X_train['err']=err_train
    X_test['type']="test"; X_test['err']=err_test
    X = pd.concat([X_train,X_test])
    
    for col in cols:
        g = sns.FacetGrid(X, hue=col, col='type', height =4, col_wrap=2)
        g = g.map(sns.distplot, "err",  hist=False, rug=False).add_legend()
        #plt.title(f'{col} vs error')   
        #plt.legend()
        plt.show()

    
def run_lasso_reg(X_train,y_train,X_test,y_test,dt,is_log=False,is_scale=False,scaler_y=None):
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error  as MSE
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Fit dt to the training set
    dt.fit(X_train, y_train)

    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    if is_scale:
        print("Scaling back y ")
        print("....")

        y_pred_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_train)),index=y_train.index).squeeze()
        y_pred_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_test)),index=y_test.index).squeeze()
        
        y_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_train)),index=y_train.index).squeeze()
        y_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_test)),index=y_test.index).squeeze()

    if is_log :     
        print("Taking inverse log transform ")
        print("....")    

        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)

        y_train = np.exp(y_train)
        y_test = np.exp(y_test)

    # # Compute mse_dt
    # mse_dt = MSE(y_train, y_pred_train)
    # # Compute rmse_dt
    # rmse_dt = mse_dt**(1/2)
    rmse_dt=score(y_train, y_pred_train,'rmse')
    # Print rmse_dt
    print("Train set RMSE of dt: {:.2f}".format(rmse_dt))


    # # Compute mse_dt
    # mse_dt = MSE(y_test, y_pred_test)
    # # Compute rmse_dt
    # rmse_dt = mse_dt**(1/2)
    rmse_dt=score(y_test, y_pred_test,'rmse')
    # Print rmse_dt
    print("Test set RMSE of dt: {:.2f}".format(rmse_dt))

    # create feature importance dataframe
    feat_importance_df = pd.DataFrame({'feature':X_train.columns,\
                                       'lass_coef':dt.coef_}).\
                                        sort_values(by='lass_coef',ascending=False).\
                                        reset_index(drop=True)    
    #print("Top 10 predictors :\n")
    #print(feat_importance_df.head(5))
    # Draw a horizontal barplot of importances_sorted
    feat_importance_df.iloc[0:10].plot(y='lass_coef',x='feature',kind='barh', color='blue')
    plt.title('Top 10 Features Importances')
    plt.show()
    
    print(type(y_train))
        
    y_train_df = pd.DataFrame({'y_true':y_train,'y_pred':y_pred_train,'type':'train','idx':y_train.index})
    y_train_df['err'] = y_train_df['y_true'] - y_train_df['y_pred']
    
    y_test_df = pd.DataFrame({'y_true':y_test,'y_pred':y_pred_test,'type':'test','idx':y_test.index})
    y_test_df['err'] = y_test_df['y_true'] - y_test_df['y_pred']
        
    return y_train_df,y_test_df,feat_importance_df,dt


def run_boostingtree_model(X_train,y_train,X_test,y_test,dt,scr,is_log=False,is_scale=False,scaler_y=None,type="Regression",model_name="xgb"):

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error  as MSE
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Fit dt to the training set
    dt.fit(X_train, y_train)

    if type == "Classification":
        y_pred_train = dt.predict_proba(X_train)[:,1]
        y_pred_test = dt.predict_proba(X_test)[:,1]
    else:
        y_pred_train = dt.predict(X_train)
        y_pred_test = dt.predict(X_test)
        
    if is_scale:
        print("Scaling back y ")
        print("....")

        y_pred_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_train)),index=y_train.index).squeeze()
        y_pred_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_pred_test)),index=y_test.index).squeeze()
        
        y_train = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_train)),index=y_train.index).squeeze()
        y_test = pd.DataFrame(scaler_y.inverse_transform(pd.DataFrame(y_test)),index=y_test.index).squeeze()

    if is_log :     
        print("Taking inverse log transform ")
        print("....")    

        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)

        y_train = np.exp(y_train)
        y_test = np.exp(y_test)

    # # Compute evaluation metric on train set
    eval_dt = score(y_train, y_pred_train,scr)
    print(f'Train set {scr} of dt: {eval_dt:.2f}')

    # # Compute evaluation metric on test set
    eval_dt = score(y_test, y_pred_test,scr)
    print(f'Test set {scr} of dt: {eval_dt:.2f}')


    # create feature importance dataframe
    feat_importance_df = pd.DataFrame({'feature':X_train.columns,\
                                       'feature_importance':dt.feature_importances_}).\
                                        sort_values(by='feature_importance',ascending=False).\
                                        reset_index(drop=True)    

    # Draw a horizontal barplot of importances_sorted
    fig, axes = plt.subplots(figsize=(10,10))
    sns.barplot(data=feat_importance_df.iloc[0:20], x='feature_importance',y='feature',\
        palette=sns.color_palette("winter",20),ax=axes)

    plt.title('Top 20 Features Importances')
    plt.show()
        
    #print(y_pred_train)
    y_train_df = pd.DataFrame({'y_true':y_train,'y_pred':y_pred_train,'type':'train','idx':y_train.index})
    if type == "Regression":
        y_train_df['err'] = y_train_df['y_true'] - y_train_df['y_pred']
    
    y_test_df = pd.DataFrame({'y_true':y_test,'y_pred':y_pred_test,'type':'test','idx':y_test.index})
    if type == "Regression":
        y_test_df['err'] = y_test_df['y_true'] - y_test_df['y_pred']
         
    return y_train_df,y_test_df,feat_importance_df,dt
