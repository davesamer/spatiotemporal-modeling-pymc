# spatiotemporal-modeling-pymc

Using the probabilistic programming framework pymc to create spatiotemporal models for diverse topics

Task 1: Create predictive bayesian models for californian house price prediction

- Methodology: Bayesian Linear Regression.

  - Full pooling (coefficients are same for all counties)
  - No pooling (coefficients vary from county to county. Coefficients independently estimated for each county)
  - Partial pooling (coefficients vary from county to county. Each county intercept is informed by state-leve intercept)
  - Gaussian process priors (coefficients vary from county to county. Coefficients of counties near to each other are assumed to be more similar, than counties far apart)

# Install requirements

```
conda env create --name pymc_st_modeling --file environment.yml
conda activate pymc_st_modeling
```

## Data

Download the shp-file of the Californian countries from
https://catalog.data.gov/dataset/tiger-line-shapefile-2019-state-california-current-county-subdivision-state-based

The californian housing dataset can be loaded from sklearn.datasets

## TODOs

- Californian Housing data

  - Implement Gaussian Priors
  - Train models on decreasing data sizes. Evaluate and compare all models
  - Can we restructure code (generalize)
  - Evaluate & compare models (BIC, AIC)
  - Create nice plots & maps

- Hierachical BNN

## Overall notes:

- What if for hierachical models the train data does not contain all spatial groups?
  - What if the test data contains group that are not in the train data?
  - I think I have to specify all spatial groups either way for the model. Does not matter if the groups exist in the training data or not
