def change_datatype(df,input_filename,input_filepath,input_filetype,convert_to):  
    '''
    This function converts columns specified in "input_filename" to data type specified in parameter "convert_to"

    Returns dataframe with columns specified in "input_filename" converted to data type specified in parameter "convert_to"
    
    '''
    
    # import functions
    from Utils.read_data import read_csv_xlsx_data as rd
    from Utils.create_eda import convert_to_datetype as cd
    from Utils.create_eda import convert_to_numtype as cn
    from Utils.create_eda import convert_to_int as ci
    from Utils.create_eda import convert_to_object as co
    from Utils.create_eda import convert_to_category as cc

    # convert to numeric
    if convert_to == "float" :
                        
        try:   
            print("--------------------------------------------------")
            print("Attempting to read column list for conversion to float")
            print("--------------------------------------------------")

            cnv_lst =  rd.func_read_data(input_filename,input_filepath,input_filetype)
            
        except:
            print("No float conversion file found")
            cnv_lst = None

        if cnv_lst is not None:
            try:
                print("--------------------------------------------------")
                print("Attempting to run conversion to float")
                print("--------------------------------------------------")
                
                # run fucntion "convert_to_float" from Utils.create_eda
                cnv_list = cnv_lst.column_name[cnv_lst.column_name.isin(df.columns.values)]
                df[cnv_list] = df[cnv_list].apply(cn.convert_to_float,0)
                
                print("--------------------------------------------------")
                print("Conversion to float successful")
                print("--------------------------------------------------")
            except:
                print("--------------------------------------------------")
                print("Conversion to float NOT successful !!! ")
                print("--------------------------------------------------")

    # convert to int
    if convert_to == "int" :
                        
        try:   
            print("--------------------------------------------------")
            print("Attempting to read column list for conversion to int")
            print("--------------------------------------------------")
            
            cnv_lst =  rd.func_read_data(input_filename,input_filepath,input_filetype)
            
        except:
            print("No int conversion file found")
            cnv_lst = None

        if cnv_lst is not None:
            try:
                print("--------------------------------------------------")
                print("Attempting to run conversion to int")
                print("--------------------------------------------------")
                
                # run fucntion "convert_to_int" from Utils.create_eda
                cnv_list = cnv_lst.column_name[cnv_lst.column_name.isin(df.columns.values)]
                df[cnv_list] = df[cnv_list].apply(ci.convert_to_int,0)

                print("--------------------------------------------------")
                print("Conversion to int successful")
                print("--------------------------------------------------")
            except:
                print("--------------------------------------------------")
                print("Conversion to int NOT successful !!! ")
                print("--------------------------------------------------")
                
    # convert to date
    if convert_to == "date" :
                        
        try:   
            print("--------------------------------------------------")
            print("Attempting to read column list for conversion to date")
            print("--------------------------------------------------")

            cnv_lst =  rd.func_read_data(input_filename,input_filepath,input_filetype)
            
        except:
            print("No date conversion file found")
            cnv_lst = None

        if cnv_lst is not None:
            try:
                print("--------------------------------------------------")
                print("Attempting to run conversion to date")
                print("--------------------------------------------------")
                
                # run fucntion "convert_to_date" from Utils.create_eda
                cnv_list = cnv_lst.column_name[cnv_lst.column_name.isin(df.columns.values)]
                df[cnv_list] = df[cnv_list].apply(cd.convert_to_date,0)
                
                print("--------------------------------------------------")
                print("Conversion to date successful")
                print("--------------------------------------------------")
            except:
                print("--------------------------------------------------")
                print("Conversion to date NOT successful !!! ")
                print("--------------------------------------------------")
                
    # convert to date
    if convert_to == "category" :
                        
        try:   
            print("--------------------------------------------------")
            print("Attempting to read column list for conversion to category")
            print("--------------------------------------------------")

            cnv_lst =  rd.func_read_data(input_filename,input_filepath,input_filetype)
            
        except:
            print("No category conversion file found")
            cnv_lst = None

        if cnv_lst is not None:
            try:
                print("--------------------------------------------------")
                print("Attempting to run conversion to category")
                print("--------------------------------------------------")
                
                # run fucntion "convert_to_category" from Utils.create_eda
                cnv_list = cnv_lst.column_name[cnv_lst.column_name.isin(df.columns.values)]
                df[cnv_list] = df[cnv_list].apply(cc.convert_to_category,0)
                
                print("--------------------------------------------------")
                print("Conversion to category successful")
                print("--------------------------------------------------")
            except:
                print("--------------------------------------------------")
                print("Conversion to category NOT successful !!! ")
                print("--------------------------------------------------")
                
    # convert to date
    if convert_to == "object" :
                        
        try:   
            print("--------------------------------------------------")
            print("Attempting to read column list for conversion to object")
            print("--------------------------------------------------")

            cnv_lst =  rd.func_read_data(input_filename,input_filepath,input_filetype)
            
        except:
            print("No object conversion file found")
            cnv_lst = None

        if cnv_lst is not None:
            try:
                print("--------------------------------------------------")
                print("Attempting to run conversion to object")
                print("--------------------------------------------------")
                
                # run fucntion "convert_to_object" from Utils.create_eda
                cnv_list = cnv_lst.column_name[cnv_lst.column_name.isin(df.columns.values)]
                df[cnv_list] = df[cnv_list].apply(co.convert_to_object,0)
                
                print("--------------------------------------------------")
                print("Conversion to object successful")
                print("--------------------------------------------------")
            except:
                print("--------------------------------------------------")
                print("Conversion to object NOT successful !!! ")
                print("--------------------------------------------------")
            
    return (cnv_lst,df)


def run_all_type_conversion(df,input_filepath,input_filetype,file_suffix):
    '''
    This function calls function "change_datatype" for each of the following data types -
    1. int
    2. float
    3. date
    4. category
    5. object
    
    Returns the transformed dataframe and all the type conversion dataframes containing column names
    for respective conversions
    '''
    # int
    input_filename= file_suffix+"_df_int."+input_filetype
    dat_int_lst,df = change_datatype(df,input_filename,input_filepath,input_filetype,convert_to="int")
     
    # float
    input_filename= file_suffix+"_df_float."+input_filetype
    dat_float_lst,df = change_datatype(df,input_filename,input_filepath,input_filetype,convert_to="float")

    # date
    input_filename= file_suffix+"_df_date."+input_filetype
    dat_date_lst,df = change_datatype(df,input_filename,input_filepath,input_filetype,convert_to="date")

    # category
    input_filename= file_suffix+"_df_category."+input_filetype
    dat_cat_lst,df = change_datatype(df,input_filename,input_filepath,input_filetype,convert_to="category")

    # object
    input_filename= file_suffix+"_df_object."+input_filetype
    dat_obj_lst,df = change_datatype(df,input_filename,input_filepath,input_filetype,convert_to="object")

    print("--------------------------------------------------")
    print("Final dtypes ")
    print("--------------------------------------------------")
    
    print(df.dtypes)

    return dat_float_lst,dat_int_lst,dat_date_lst,dat_cat_lst,dat_obj_lst,df
