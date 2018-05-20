
import requests

#Andrew Deng & Jessica Song



#for example,
#table = pd.read_csv("person.csv")
#zipcodes = pd.read_csv("zipcodes.csv")


def alter_zip(x):

    x = str(x)
    if len(x)==3:
        x = "00" + x
    return x


def to_lat_lon(table, ziptable):
	
	ziptable['zip'] = ziptable['zip'].apply(lambda x: alter_zip(x))   
	ziptable = ziptable.drop(['city', 'state', 'timezone','dst'], axis=1) 
	table = table.drop(['tract2010id', 'blockgroup2010id', 'block2010id'], axis=1)

	table = pd.merge(table, ziptable, on ='zip')

	return table



def get_fips(table):

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

	temp = np.array(table['fips'])
	fips = [x for x in temp if x!=None]
	state = [x[:2] for x in temp if x!=None]
	county = [x[2:5] for x in temp if x!=None]
	tract = [x[5:11] for x in temp if x!=None]
	blockgroup = [x[11:12] for x in temp if x!=None]
	fips = pd.DataFrame({'fips' : fips, 'state' : state, 'county' : county, 'tract' : tract, 'blockgroup' : blockgroup })
	table = pd.merge(table, fips, left_on = 'fips', right_on = 'fips')

	return table


	
def info_retrieve(table):

	asc = []
	for i, row in table.iterrows():
	    fips = row['fips']
	    blkgrp = row['blockgroup']
	    state = row['state']
	    county = row['county']
	    tract = row['tract']


		search_term = 'Name, B19058_001E, B9008_002E, B24124_001E, B19301_001E, B23018_001E, B19055_001E, B17001_001E, B20004_002E'
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
	colnames = ['food_stamps', 'unmarried_partner', 'occupation', 'capita_income', 'mean_work_hr', 'ss_inc', 'pov_ind', 'edu25']
	asc_df = pd.DataFrame(asc, columns = colnames)

	# merge with the person table
	total_df = pd.merge(table, asc_df, left_on = 'fips', right_on = 'fips')

		# food_stamps = json_dict[1][1]
		# unmarried_partner = json_dict[1][2]
		# occupation = json_dict[1][3]
		# capita_income = json_dict[1][4]
		# mean_work_hr = json_dict[1][5]
		# ss_inc = json_dict[1][6]
		# pov_ind = json_dict[1][7]
		# edu25 = json_dict[1][8]
	
	return asc_df, total_df









