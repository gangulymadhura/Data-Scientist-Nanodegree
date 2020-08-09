def percent_to_numeric(col):
    '''
    INPUT -
    col - type series
          data column that needs to be converted from dollar to numeric
    
    OUTPUT -
    col - dataframe column converted to float type    
    
    '''        
    print(col.name)
    print("Before conversion :\n", col[~col.isnull()].head() )
    
    # remove dollar and comma and convert to float
    col = col.str.replace('%',"").astype(float)
    
    print("After conversion :\n", col[~col.isnull()].head())
    print("\n")
    return col