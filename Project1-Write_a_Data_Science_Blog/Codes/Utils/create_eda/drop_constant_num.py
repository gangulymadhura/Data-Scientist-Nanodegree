def drop_constant_num(dat_df,varlist):
 
	varlist = dat_df.columns[dat_df.columns.isin(varlist)].values
	std_df = dat_df[varlist].std()
	#print(std_df)
	constant_var = std_df.index[std_df<=0.0]
	dat_df.drop(constant_var,axis=1,inplace=True)
	print('Columns dropped due to constant values : \n')
	print(constant_var)
	return dat_df