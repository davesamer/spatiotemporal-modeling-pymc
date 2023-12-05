import pandas as pd
import pymc as pm

"""
models
 1) BLR - full pooling
 2) BLR - no pooling
 3) BLR - partial pooling
 4) BLR - gaussian process priors

 as soon as overall process stands (model training, prediction, evaluation), further models can be implemented
 Hierachical BNN's
 Besag York Mollie Model
 Conditional Autoregressive (CAR) 
"""


def create_blr_full_pooling(X: pd.DataFrame,
                            y: pd.Series) -> pm.Model:
    """Normal Bayesian Linear Regression Model"""

    with pm.Model() as blr_full_pooling:

        # create data containers for all the features
        medinc = pm.MutableData('medinc', X.MedInc)
        house_age = pm.MutableData('house_age', X.HouseAge)
        ave_rooms = pm.MutableData('ave_rooms', X.AveRooms)
        ave_bedrms = pm.MutableData('ave_bedrms', X.AveBedrms)
        population = pm.MutableData('population', X.Population)
        ave_occup = pm.MutableData('ave_occup', X.AveOccup)
        median_house_value = pm.MutableData(
            'median_house_value', y)

        # specify priors for the features
        intercept = pm.Normal('intercept', 0, 10)
        beta_medinc = pm.Normal('beta_medinc', 0, 10)
        beta_house_age = pm.Normal('beta_house_age', 0, 10)
        beta_ave_rooms = pm.Normal('beta_ave_rooms', 0, 10)
        beta_ave_bedrms = pm.Normal('beta_ave_bedrms', 0, 10)
        beta_population = pm.Normal('beta_population', 0, 10)
        beta_ave_occup = pm.Normal('beta_ave_occup', 0, 10)
        sigma = pm.HalfCauchy("sigma", beta=10)

        # Logistic Regression
        mean = intercept + \
            beta_medinc * medinc + \
            beta_house_age * house_age + \
            beta_ave_rooms * ave_rooms + \
            beta_ave_bedrms * ave_bedrms + \
            beta_population * population + \
            beta_ave_occup * ave_occup

        likelihood = pm.Normal("y", mu=mean, sigma=sigma,
                               observed=median_house_value)

        return blr_full_pooling


def create_blr_no_pooling(X: pd.DataFrame,
                          y: pd.Series,
                          coords: dict,
                          spatial_grouping_var: str) -> pm.Model:
    """Hierachical Bayesian Linear Regression Model with no pooling"""

    with pm.Model(coords=coords) as blr_full_pooling:

        # create data containers for all the features
        medinc = pm.MutableData('medinc', X.MedInc)
        house_age = pm.MutableData('house_age', X.HouseAge)
        ave_rooms = pm.MutableData('ave_rooms', X.AveRooms)
        ave_bedrms = pm.MutableData('ave_bedrms', X.AveBedrms)
        population = pm.MutableData('population', X.Population)
        ave_occup = pm.MutableData('ave_occup', X.AveOccup)
        median_house_value = pm.MutableData(
            'median_house_value', y)
        spatial_group_idx = pm.MutableData(
            'spatial_group_idx', X[spatial_grouping_var])

        # specify priors for the features
        intercept = pm.Normal('intercept', 0, 10, dims=("spatial_groups"))
        beta_medinc = pm.Normal('beta_medinc', 0, 10, dims=("spatial_groups"))
        beta_house_age = pm.Normal(
            'beta_house_age', 0, 10, dims=("spatial_groups"))
        beta_ave_rooms = pm.Normal(
            'beta_ave_rooms', 0, 10, dims=("spatial_groups"))
        beta_ave_bedrms = pm.Normal(
            'beta_ave_bedrms', 0, 10, dims=("spatial_groups"))
        beta_population = pm.Normal(
            'beta_population', 0, 10, dims=("spatial_groups"))
        beta_ave_occup = pm.Normal(
            'beta_ave_occup', 0, 10, dims=("spatial_groups"))
        sigma = pm.HalfCauchy("sigma", beta=10)

        # Logistic Regression
        mean = intercept[spatial_group_idx] + \
            beta_medinc[spatial_group_idx] * medinc + \
            beta_house_age[spatial_group_idx] * house_age + \
            beta_ave_rooms[spatial_group_idx] * ave_rooms + \
            beta_ave_bedrms[spatial_group_idx] * ave_bedrms + \
            beta_population[spatial_group_idx] * population + \
            beta_ave_occup[spatial_group_idx] * ave_occup

        likelihood = pm.Normal("y", mu=mean, sigma=sigma,
                               observed=median_house_value)

        return blr_full_pooling
