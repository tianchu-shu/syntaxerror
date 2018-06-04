from final_pipeline import *
from final_plot import *
from jocojims import *
from indepv import *
from final_temporal import *
from final_classifier_final import *



def run():


	# READ DATA
	df = load_from_file('final_data.csv')
	acs = load_from_file('fips_acs.csv')

	# CLEAN & MERGE DATA
	acs = acs.drop(columns=ACS_DROP, axis=1)
	df = df.dropna(subset=['zip'])

	df = acs.merge(df, how="inner")
	df = df.drop_duplicates()

	df = df.drop(DROP_COLS, axis=1)

	# PRE-PROCESS DATA
	df, binned = bin_gen(df, CATS, label='binned', fix_value='prefix')
	df = dummy_variable(df, DUMMIES+binned)

	remove = DATE_COLS
	features = binned+DUMMIES
	features = list(set(features).difference(set(remove)))
	
		
	# MODELS & EVALUATION
	#result = temporal_eval(features, df)
	
	# COPYTING THE DATASET TO BE USED FOR SPLITTING 
	viz_df = df.copy()
	
	# SETTING DATES FOR SEPARATING TRAIN & TESTING DATA
	END = df['booking_date'].max()[:10]
	START = df['booking_date'].min()[:10]
	MID = '2015-07-01'
	
	
	# SPLITTING DATASET INTO TRAINING AND TESTING
	train, test = temporal_split(df, 'booking_date', start, middle_date, end)
	trainv,testv = temporal_split(viz_df, 'booking_date', start, middle_date, end)
	
	x_train, x_test, y_train, y_test = split_data(train, test, Y)
	
	a, viz_x, b, c = split_data(trainv,testv, Y)
	
	
	while len(features_list)>0:
		num = len(features_list)
		# Running on All the var including mental health and bail var
		x_train = x_train[features_list[num]]
		x_test = x_test[features_list[num]]
	
	
		for i in range(len(Y)):

			results = clf_loop(MODELS_TO_RUN, x_train, x_test, y_train[Y[i]], y_test[Y[i]])
			best = best_grid(results)

			#Use the best performed Random Forest model to see the top 10% at risk people's data
			df_sorting = finding_risk_scores(x_train, x_test, y_train[Y[i]], y_test[Y[i]], best, viz_x)
			RF_df = df_sorting.sort_values(by=['RF'], ascending=False)

			#PLOTTING THE SELECTED FEATURES
			plot_df(RF_df[:50], features_to_see, save=False)

			#PRECISON_RECALL GRAPHS OF THE BEST MODEL
			plot_best(MODELS_TO_RUN,  x_train, x_test, y_train[Y[i]], y_test[Y[i]], best)
			
		del (features_list[-1])
	
	

	

										 
	
		

	
	



if __name__=="__main__":
	run()
