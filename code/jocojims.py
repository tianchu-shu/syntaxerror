from indpv_lists import *

DATE_COLS = ['booking_date', 'release_date']

ACS_DROP = ['latitude', 'longitude', 'fips']
DROP_COLS = ['re_entry', 'mni_no', 'dedupe_id', 'dob', 'state', 'city',  'zip', 'release_date', 'case_no', 'booking_no', 'pri_dx_value',
       'refferal_source', 'after_released']

DUMMIES = ['sex', 'race','mar_stat', 'case_type', 'arresting_agency', 'arresting_agency_type', 'bail_type' ]


CATS = ['age', 'per_capita_income', 'poverty_stat', 'household_income',
       'mortgage_stat', 'school_enrollment_16-19', 'own_children_under18',
       'SSI_income', 'grandparent_care', 'military_service', 'education_25',
       'property_value_occupied', 'retirement_income', 'rent_over_income','bail_amt']

#FEATURES = all_var

