
import requests

#Andrew Dend

def info_retrieve(blkgrp, county, tract, gender):
	
	if gender == 'male':
		search_term = 'Name, B22007, B09008, B24125, B19301, B23020, B19055, B17001, B15003'
	elif gender == 'female' 
		search_term = 'Name, B22007, B09008, B24126, B19301, B23020, B19055, B17001, B15003'
	key = <Insert Key Here>
	address = 	address = 'https://api.census.gov/data/2015/acs5?get={}&for=block+group:{}&in=state:17+county:{}+tract:{}&key={}'.format(search_term, blkgrp, county, tract, key)

	json_results = request.get(address)
	json_dict = json_results.json()

	food_stamps = json_dict[1][1]
	unmarried_partner = json_dict[1][2]
	occupation = json_dict[1][3]
	capita_income = json_dict[1][4]
	mean_work_hr = json_dict[1][5]
	ss_inc = json_dict[1][6]
	pov_ind = json_dict[1][7]
	edu25 = json_dict[1][8]
	
	return (food_stamps, unmarried_partner, occupation, capita_income, mean_work_hr, ss_inc, pov_ind, edu25)


