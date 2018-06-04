# syntaxerror-mlpp2018

Contributors: [Andrew Deng](https://github.com/CAPPAndrew), [Amir Kazi](https://github.com/amirkazi), [Tianchu Shu](https://github.com/tianchu-shu) and [Jessica Song](https://github.com/belovedsong)


## Project goal
__Syntax Error__ aims to identify which inmates are most at risk of recidivism following their release within one year or two year, using the resources of the Johnson county jail system.


## Content
- Data exploration
- Data cleaning and preprocessing
- Feature generation
- Machine learning pipeline
- Model evaluation and bias analysis


## Package to install
1. pandas
2. psycopg2
3. numpy
4. matplotlib
5. seaborn
6. psycopg2
7. sklearn
8. datetime
9. requests
10. graphviz

# syntaxerror-mlpp2018

Contributors: [Andrew Deng](https://github.com/CAPPAndrew), [Amir Kazi](https://github.com/amirkazi), [Tianchu Shu](https://github.com/tianchu-shu) and [Jessica Song](https://github.com/belovedsong)


__Syntax Error__ aims to identify which inmates are most at risk of recidivism following their release within one year or two year, using the resources of the Johnson county jail system.

## Project Goal
Identify individuals at risk going back to jail using personal data, mental health records, and demographic data of 
neighborhood areas


## Data Sources
Johnson County Jail, Census Bureau



## What We've Done
We have worked on 
1) Data Exploration
2) Data Cleaning, Data Integration, and Pre-Processing
3) Feature Generation
4) Machine learning Classifiers & Evaluations (Pipeline)


## Package to install

1. pandas
2. psycopg2
3. numpy
4. matplotlib
5. seaborn
6. psycopg2
7. sklearn
8. datetime
9. requests
10. graphviz


## Features from Census
In general, all code for the pipeline can be found in code with the prefix 'final' (eg. final_plot) and all work for preliminary steps of the project is located in raw. 


## Codes
1) Jupyter notebook

Final modeling results
- All-var.ipynb
- Person_societal_var.ipynb
- Bail_info.ipynb
- Comiling_everything.ipynb
- mh_var.ipynb


2) Python code ("/code")

< File Name>
- census.py
- run_api.py

< Description>
  : Codes to retrieve census data to be used as demographic data


< File Name >
- final_default_grids.py
- jocojims.py
- indpv_lists.py

< Description >
  : Set up files for the pipeline

< File Name >

- final_connection.py
- final_load_dfs.py
- final_explore_and_viz.py
- final_preprocessing.py
- final_temporal.py
- final_classifier_final.py
- final_plot.py

< File Name >
- final_pipeline.py
- final_run.py

< Description >
  : Gathered functions for each part of pipeline and put together in final_pipeline except data exploration


## Running the code




## Brief
In general, all work for preliminary steps of the project is located in raw， all code for the pipeline can be found in code with the prefix 'final' (eg. final_plot).

## Results
We compared the effectiveness of different models trained on “biased” and “unbiased” feature sets: personal_societal, personal_societal + mental_health, personal_societal + bail_info, and all varaiables.
