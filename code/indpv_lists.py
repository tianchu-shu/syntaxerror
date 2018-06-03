
gender_var = ['sexFEMALE', 'sexMALE', 'sexmissing']
mh_var = ['mh_treatment']
race_var = ['raceAMERICAN INDIAN OR ALASKA NATIVE', 'raceASIAN', 'raceBLACK OR AFRICAN AMERICAN', 'raceWHITE']
marital_var = ['mar_statA', 'mar_statD', 'mar_statM', 'mar_statN', 'mar_statS', 'mar_statU', 'mar_statW', 'mar_statY', 'mar_statmissing']
crime_var = ['case_typeCR', 'case_typeDV', 'case_typeJV',
 'arresting_agencyFAIRWAY P.D.', 'arresting_agencyGARDNER P.D.', 'arresting_agencyJOHNSON COUNTY PARK PATROL', "arresting_agencyJOHNSON COUNTY SHERIFF'S DEPARTMENT", 'arresting_agencyKANSAS HIGHWAY PATROL', 'arresting_agencyLAKE QUIVIRA P.D.', 'arresting_agencyLEAWOOD P.D.',
 'arresting_agencyLENEXA P.D.', 'arresting_agencyMERRIAM P.D.', 'arresting_agencyMISSION P.D.', 'arresting_agencyOLATHE P.D.', 'arresting_agencyOTHER AGENCY',
 'arresting_agencyOVERLAND PARK P.D.', 'arresting_agencyPRAIRIE VILLAGE P.D.', 'arresting_agencyROELAND PARK P.D.', 'arresting_agencySHAWNEE MISSION SCHOOL SECURITY',
 'arresting_agencySHAWNEE P.D.', 'arresting_agencySPRING HILL P.D.', 'arresting_agencyWESTWOOD P.D.', 'arresting_agency_typeCITY',
 'arresting_agency_typeCOUNTY PARK DISTRICT', 'arresting_agency_typeCOUNTY SHERIFF', 'arresting_agency_typeOTHER AGENCY', 'arresting_agency_typeSCHOOL', 'arresting_agency_typeSTATE']
individual_var = ['age_bin1', 'age_bin2', 'age_bin3', 'age_bin4', 'own_children_under18_bin1', 'own_children_under18_bin2', 'own_children_under18_bin3', 'own_children_under18_bin4',  'grandparent_care_bin1', 'grandparent_care_bin2', 'grandparent_care_bin3', 'grandparent_care_bin4','military_service_bin'1, 'military_service_bin'2, 'military_service_bin3', 'military_service_bin4']
bail_var = ['bailed_out', 'bail_typeCA', 'bail_typeGPS', 'bail_typeORCD', 'bail_typePR', 'bail_typeSUR', 'bail_amt_bin']
econ_var = ['per_capita_income_bin1', 'per_capita_income_bin2', 'per_capita_income_bin3', 'per_capita_income_bin4', 'poverty_stat_bin1', 'poverty_stat_bin2', 'poverty_stat_bin3', 'poverty_stat_bin4', 'household_income_bin1', 'household_income_bin2', 'household_income_bin3', 'household_income_bin4',
            'mortgage_stat_bin1', 'mortgage_stat_bin2', 'mortgage_stat_bin3', 'mortgage_stat_bin4', 'SSI_income_bin1', 'SSI_income_bin2', 'SSI_income_bin3', 'SSI_income_bin4', 'property_value_occupied_bin1', 'property_value_occupied_bin2', 'property_value_occupied_bin3', 'property_value_occupied_bin4', 'retirement_income_bin1', 'retirement_income_bin2', 'retirement_income_bin3', 'retirement_income_bin4', 'rent_over_income_bin1', 'rent_over_income_bin2', 'rent_over_income_bin3', 'rent_over_income_bin4']
educ_var = ['school_enrollment_16-19_bin1', 'school_enrollment_16-19_bin2', 'school_enrollment_16-19_bin3', 'school_enrollment_16-19_bin4', 'education_25_bin1', 'education_25_bin2', 'education_25_bin3', 'education_25_bin4']

all_var = gender_var + mh_var + race_var + marital_var + crime_var + individual_var + bail_var + econ_var + educ_var
mh_bail_var = mh_var + bail_var
census_vars = individual_var + econ_var + educ_var

trial_var = crime_var + bail_var
personal_var = marital_var + individual_var + race_var + gender_var
societal_var = econ_var + educ_var
person_societal_var = personal_var + societal_var
