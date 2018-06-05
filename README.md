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
6. graphviz
7. sklearn
8. datetime
9. requests


## Data Sources
Johnson County Jail, Census Bureau


## What We've Done
We have worked on 
1) Data Exploration
2) Data Cleaning, Data Integration, and Pre-Processing
3) Feature Generation
4) Machine learning Classifiers & Evaluations (Pipeline)


## Code
1) Jupyter notebook

#### Final modeling results
- Person_societal_var.ipynb
- Bail_info.ipynb
- mh_info.ipynb
- All-var.ipynb


2) Python code ("/code")

#### Code to retrieve census data to be used as demographic data

- census.py
- run_api.py

#### Settings for the pipeline

- final_default_grids.py
- jocojims.py
- indpv_lists.py

#### Code for the final pipeline

- final_connection.py
- final_load_dfs.py
- final_explore_and_viz.py
- final_preprocessing.py
- final_temporal.py
- final_classifier_final.py
- final_plot.py
- final_pipeline.py

#### Gathered functions for each part of pipeline and put together except data exploration

- final_run.py


## Running the code

- Via Jupyter Notebook

- Via Python file & Terminal
  ```
  python final_run.py 
  ```


## Summary
In general, all work for preliminary steps of the project is located in raw， all code for the pipeline can be found in code with the prefix 'final' (eg. final_plot).

## Results
We compared the effectiveness of different models trained on “biased” and “unbiased” feature sets: 
1. All variables (Using all our variables for training data)
2. Personal & Mental Health related Variables 11
3. Personal & Bail related Variables
4. Personal related Variables.
