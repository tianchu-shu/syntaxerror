
import requests

#Andrew Deng & Jessica Song



#for example,
#table = pd.read_csv("person.csv")
#zipcodes = pd.read_csv("zipcodes.csv")


def alter_zip(x):
'''
Padding with zeros

Input - (int) x
Ouput - (str) x
'''

    x = str(x)
    if len(x)==3:
        x = "00" + x
    return x


def to_lat_lon(table, ziptable):
'''
Padding with zeros if the length of zipcode is 3
Dropping irrelevant columns and merge given tables

Input - (DataFrame) table, ziptable
Output - (DataFrame) table
	: merged table on zipcode

'''
	
	ziptable['zip'] = ziptable['zip'].apply(lambda x: alter_zip(x))   
	ziptable = ziptable.drop(['city', 'state', 'timezone','dst'], axis=1) 

	table = pd.merge(table, ziptable, on ='zip')

	return table



def get_fips(table):
'''
Getting fips from geo API

Input - (DataFrame) table
Output - (DataFrame) table
	: table with additional 'fips code' column
	
'''

	fips = []
	for i, row in table.iterrows():
	    lat = row['latitude']
	    lon = row['longitude']
	    url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&format=json'.format(lat,lon)
	    r= requests.get(url)
	    file = r.json()
	    if r:
	        code = file['Block']["FIPS"]
	        fips.append(code)
	    else:
	        fips.append(None)
	table['fips'] = fips



def break_down(table):
'''
Breaking down fips into state, county, tract, and blockgroup codes

Input - (DataFrame) table
Output - (DataFrame) table
	: table with additional state, county, tract and blockgroup columns
	
'''

	temp = np.array(table['fips'])
	fips = [x for x in temp if x!=None]
	state = [x[:2] for x in temp if x!=None]
	county = [x[2:5] for x in temp if x!=None]
	tract = [x[5:11] for x in temp if x!=None]
	blockgroup = [x[11:12] for x in temp if x!=None]
	fips_df = pd.DataFrame({'fips' : fips, 'state' : state, 'county' : county, 'tract' : tract, 'blockgroup' : blockgroup })
	unique_fips = fips_df.drop_duplicates()
	table = pd.merge(table, fips_df, left_on = 'fips', right_on = 'fips')

	return table


	
def info_retrieve(table):
'''
Getting demographic data from census API

Input - (DataFrame) table
Output - (DataFrame) total_df
	: table with demographic data
	
'''

	asc = []
	for i, row in table.iterrows():
	    fips = row['fips']
	    blkgrp = row['blockgroup']
	    state = row['state_y']
	    county = row['county']
	    tract = row['tract']


        search_term = 'B19301_001E,B17021_001E,B19001_001E,B25087_001E,B14005_001E,B09002_001E, B19056_001E, B99104_001E, B21002_001E, B15002_001E, B25075_001E, B19059_001E, B25070_001E'
		key = <Insert Key Here>
		address = 'https://api.census.gov/data/2010/acs5?get={}&for=block+group:{}&in=state:{}+county:{}+tract:{}&key={}'.format(search_term, blkgrp, state, county, tract, key)

		json_results = requests.get(address)
		json_dict = json_results.json()


		if json_results:
			info = [fips] + json_dict[1]
			asc.append(info)
		else:
			asc.append([None]*len(json_dict[1]))

			
	# return api dataframe
    colnames = ['fips','per_capita_income', 'poverty_stat', 'household_income', 'mortgage_stat', 'school_enrollment_16-19', 'own_children_under18', 'SSI_income', 'grandparent_care', 'military_service', 'education_25', 'property_value_occupied', 'retirement_income', 'rent_over_income', 'state', 'county', 'tract', 'blockgroup']
	asc_df = pd.DataFrame(asc, columns = colnames)
	unique_asc = asc_df.drop_duplicates()

	# merge with the person table
	# total_df = pd.merge(table, unique_asc, left_on = 'fips', right_on = 'fips')

	
	return unique_asc



def merge(table1, table2):

	total_df = pd.merge(table1, table2, left_one ='fips', right_on = 'fips')

	return total_df



#if __name__=="__main__":








