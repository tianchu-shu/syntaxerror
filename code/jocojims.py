from indpv_lists import *

Y = ['within_one', 'within_two]
     
DATE_COLS = ['booking_date', 'release_date']

ACS_DROP = ['latitude', 'longitude', 'fips']
DROP_COLS = ['re_entry', 'mni_no', 'dedupe_id', 'dob', 'state', 'city',  'zip', 'release_date', 'case_no', 'booking_no', 'after_released']

DUMMIES = ['sex', 'race','mar_stat', 'case_type', 'arresting_agency', 'arresting_agency_type', 'bail_type' ,'pri_dx_value', 'refferal_source']


CATS = ['age', 'per_capita_income', 'poverty_stat', 'household_income',
       'mortgage_stat', 'school_enrollment_16-19', 'own_children_under18',
       'SSI_income', 'grandparent_care', 'military_service', 'education_25',
       'property_value_occupied', 'retirement_income', 'rent_over_income','bail_amt']
     
     
MODELS_TO_RUN = ['RF', 'Boost', 'Bag', 'Logit', 'Tree', 'ET', 'NB']
     
FEATURE_LISTS = [mh_info, bail_info, person_societal_var, all_var]
     
FEATURES_TO_SEE = ['bail_amt', 'bail_type', 'bailed_out','case_type', 'city', 
       'education_25', 'grandparent_care', 'household_income', 'mar_stat',
       'mh_treatment', 'military_service', 'mortgage_stat',
       'own_children_under18', 'per_capita_income', 'poverty_stat',
       'property_value_occupied', 'race', 're_entry', 'rent_over_income']

