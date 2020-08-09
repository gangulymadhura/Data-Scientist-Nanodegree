def convert_to_int(col):    
    '''
    INPUT -
    col - type series
          data column that needs to be converted to datetime type
    
    OUTPUT -
    col - dataframe column converted to datetime type   
    
    '''      
    import pandas as pd
    import numpy as np

    print(col.name)
    print("Before conversion :\n", col.dtype )
    
    # replace missing if any with -999
    if col.isnull().sum()>0:
        try:
            col.fillna("-999",inplace=True)
        except:
            col.fillna(-999,inplace=True)


    # convert to int
    col = col.astype(int)
    col = col.replace(-999, np.nan)
    
    print("After conversion :\n", col.dtype )
    print("\n")
    return col
