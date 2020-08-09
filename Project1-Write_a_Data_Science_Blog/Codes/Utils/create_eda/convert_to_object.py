def convert_to_object(col):    
    '''
    INPUT -
    col - type series
          data column that needs to be converted to category type
    
    OUTPUT -
    col - dataframe column converted to category type   
    
    '''      
    import pandas as pd

    print(col.name)
    print("Before conversion :\n", col.dtype )
    
    # convert to category type
    col = col.astype('object')
    
    print("After conversion :\n", col.dtype )
    print("\n")
    return col
