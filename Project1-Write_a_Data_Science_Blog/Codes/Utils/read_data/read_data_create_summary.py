def read_data_create_summary(input_filename,input_filepath,output_filepath,file_type,file_suffix):
    '''
    This function reads in the data and writes out a summary report

    '''   
    # import custome defined functions
    from Utils.read_data import read_csv_xlsx_data as rd
    from Utils.create_eda import create_eda as eda

    # Step 1 : read data
    print("--------------------------------------------------")    
    print("Reading :", input_filename)
    print("--------------------------------------------------")    
    dat_df = rd.func_read_data(input_filename,input_filepath,file_type)
    
    
    # Step 2 : create summary report
    print("--------------------------------------------------")    
    print("Running summary report")
    print("--------------------------------------------------")    
    eda.create_eda(dat_df).to_csv(output_filepath+file_suffix+"_df_eda.csv")
    print("Created summary report : ",(file_suffix+"_df_eda.csv"))
    
    return dat_df
