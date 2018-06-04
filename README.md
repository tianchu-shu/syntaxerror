# syntaxerror-mlpp2018

Contributors: [Andrew Deng](https://github.com/CAPPAndrew), [Amir Kazi](https://github.com/amirkazi), [Tianchu Shu](https://github.com/tianchu-shu) and [Jessica Song](https://github.com/belovedsong)


## Project goal
__Syntax Error__ aims to identify which inmates are most at risk of recidivism following their release within one year or two year, using personal data, mental health records from Johnson county jail system and demographic data of 
neighborhood areas


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


## Data Sources
Johnson County Jail, Census Bureau


## What We've Done
We have worked on 
1) Data Exploration
2) Data Cleaning, Data Integration, and Pre-Processing
3) Feature Generation
4) Machine learning Classifiers & Evaluations (Pipeline)


## Codes
1) Jupyter notebook

Final modeling results
- All-var.ipynb
- Person_societal_var.ipynb
- Bail_info.ipynb
- Comiling_everything.ipynb
- mh_var.ipynb


2) Python code ("/code")

- census.py
- run_api.py

Codes to retrieve census data to be used as demographic data


- final_default_grids.py
- jocojims.py
- indpv_lists.py

Settings for the pipeline

- final_connection.py
- final_load_dfs.py
- final_explore_and_viz.py
- final_preprocessing.py
- final_temporal.py
- final_classifier_final.py
- final_plot.py

Codes for the final pipeline

- final_pipeline.py
- final_run.py

Gathered functions for each part of pipeline and put together in final_pipeline except data exploration


## Running the code

- Via Jupyter Notebook

- Via Python file & Terminal
  ```
  run finaLrun.py 
  ```


## Brief
In general, all work for preliminary steps of the project is located in raw， all code for the pipeline can be found in code with the prefix 'final' (eg. final_plot).

## Results
We compared the effectiveness of different models trained on “biased” and “unbiased” feature sets: personal_societal, personal_societal + mental_health, personal_societal + bail_info, and all varaiables.
