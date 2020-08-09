import pandas as pd
# function for reading in data
def func_read_data(input_data_filename,input_data_filepath,file_type,sheet_name=0,index_col=None,parse_dates=False):
    '''
    INPUT -
    input_data_filename : type string 
                          name of file to be read
    input_data_filepath : type string
                          location of file to be read
    file_type : type string 
                "csv" or "xlsx"
    sheet_name : type string, default None
                 if file_type = "xlsx" then name of the sheet to be read
    index_col : type string
                column that will be the index of the dataframe
    parse_dates : type boolean or list of strings
                  default False, if True then pandas will try to infer whicg data columns are date/ datetime,
                  if list of strings will parse the column that match the strings as dates
                    
    '''
    if file_type == 'csv':
        dat_df = pd.read_csv(input_data_filepath + input_data_filename,index_col=index_col,parse_dates=parse_dates)
    if file_type == 'xlsx':
        dat_df = pd.read_excel(input_data_filepath + input_data_filename,sheet_name=sheet_name,index_col=index_col,parse_dates=parse_dates)
        
    print("--------------------------------------------------")
    # no NaN values in sample data shown
    print(" File read successfully")
    #print('Top Records \n',dat_df.dropna(how="all").head(3))
    print('Top Records \n',dat_df.head(3))
    print("--------------------------------------------------")    
    print("--------------------------------------------------")    
    print('Shape of file \n',dat_df.shape)
    print("--------------------------------------------------")    
    print("--------------------------------------------------")   
    #print("\n")    
    
    return(dat_df)

