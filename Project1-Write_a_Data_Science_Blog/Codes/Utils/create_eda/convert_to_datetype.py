def convert_to_date(col):    
    '''
    INPUT -
    col - type series
          data column that needs to be converted to datetime type
    
    OUTPUT -
    col - dataframe column converted to datetime type   
    
    '''      
    import pandas as pd

    print(col.name)
    print("Before conversion :\n", col.dtype )
    
    # remove dollar and comma and convert to float
    col = pd.to_datetime(col)
    
    print("After conversion :\n", col.dtype )
    print("\n")
    return col
