def drop_constant_str(dat_df,varlist):

	varlist = dat_df.columns[dat_df.columns.isin(varlist)].values
	uniq_df = dat_df[varlist].nunique()
	constant_var = uniq_df.index[uniq_df<=1]
	dat_df.drop(constant_var,axis=1,inplace=True)
	print('Columns dropped due to constant values : \n')
	print(constant_var)

	return dat_df