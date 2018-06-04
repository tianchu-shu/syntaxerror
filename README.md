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


## Breif
In general, all work for preliminary steps of the project is located in raw， all code for the pipeline can be found in code with the prefix 'final' (eg. final_plot).

## Results
We compared the effectiveness of different models trained on “biased” and “unbiased” feature sets: personal-societal, personal-societal + mental-health, personal-societal + bail-info, and all varaiables.
