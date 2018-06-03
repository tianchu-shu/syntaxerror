
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
individual_var = ['age', 'own_children_under18_bin', 'grandparent_care_bin', 'military_service_bin']
bail_var = ['bailed_out', 'bail_typeCA', 'bail_typeGPS', 'bail_typeORCD', 'bail_typePR', 'bail_typeSUR', 'bail_amt_bin']
econ_var = ['per_capita_income_bin', 'poverty_stat_bin', 'household_income_bin', 'mortgage_stat_bin', 'SSI_income_bin', 'property_value_occupied_bin', 'retirement_income_bin', 'rent_over_income_bin']
educ_var = ['school_enrollment_16-19_bin', 'education_25_bin']

all_var = gender_var + mh_var + race_var + marital_var + crime_var + individual_var + bail_var + econ_var + educ_var
mh_bail_var = mh_var + bail_var
census_vars = individual_var + econ_var + educ_var

trial_var = crime_var + bail_var
personal_var = marital_var + individual_var + race_var + gender_var
societal_var = econ_var + educ_var
person_societal_var = personal_var + societal_var
