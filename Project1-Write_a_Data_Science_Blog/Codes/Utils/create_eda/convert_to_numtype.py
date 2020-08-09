def convert_to_float(col):    
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
    
    # convert to float
    col = pd.to_numeric(col)
    
    print("After conversion :\n", col.dtype )
    print("\n")
    return col
