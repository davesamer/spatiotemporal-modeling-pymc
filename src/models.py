import pandas as pd
import pymc as pm
import numpy as np

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

    with pm.Model() as model:

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
        intercept = pm.Normal('intercept', 0, 1)
        beta_medinc = pm.Normal('beta_medinc', 0, 1)
        beta_house_age = pm.Normal('beta_house_age', 0, 1)
        beta_ave_rooms = pm.Normal('beta_ave_rooms', 0, 1)
        beta_ave_bedrms = pm.Normal('beta_ave_bedrms', 0, 1)
        beta_population = pm.Normal('beta_population', 0, 1)
        beta_ave_occup = pm.Normal('beta_ave_occup', 0, 1)
        sigma = pm.HalfCauchy("sigma", beta=1)

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

        return model


def create_blr_no_pooling(X: pd.DataFrame,
                          y: pd.Series,
                          coords: dict,
                          spatial_grouping_var: str) -> pm.Model:
    """Hierachical Bayesian Linear Regression Model with no pooling"""

    with pm.Model(coords=coords) as model:

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
        intercept = pm.Normal('intercept', 0, 1, dims=("spatial_groups"))
        beta_medinc = pm.Normal('beta_medinc', 0, 1)
        beta_house_age = pm.Normal('beta_house_age', 0, 1)
        beta_ave_rooms = pm.Normal('beta_ave_rooms', 0, 1)
        beta_ave_bedrms = pm.Normal('beta_ave_bedrms', 0, 1)
        beta_population = pm.Normal('beta_population', 0, 1)
        beta_ave_occup = pm.Normal('beta_ave_occup', 0, 1)
        sigma = pm.HalfCauchy("sigma", beta=1)

        # Logistic Regression
        mean = intercept[spatial_group_idx] + \
            beta_medinc * medinc + \
            beta_house_age * house_age + \
            beta_ave_rooms * ave_rooms + \
            beta_ave_bedrms * ave_bedrms + \
            beta_population * population + \
            beta_ave_occup * ave_occup

        likelihood = pm.Normal("y", mu=mean, sigma=sigma,
                               observed=median_house_value)

        return model


def create_blr_partial_pooling(X: pd.DataFrame,
                               y: pd.Series,
                               coords: dict,
                               spatial_grouping_var: str) -> pm.Model:
    """Hierachical Bayesian Linear Regression Model with no pooling"""

    with pm.Model(coords=coords) as model:

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
        intercept_mu = pm.Normal('intercept_mu', 0, 1)
        intercept_sigma = pm.HalfCauchy('intercept_sigma', beta=1)
        intercept = pm.Normal('intercept', intercept_mu,
                              intercept_sigma, dims=("spatial_groups"))

        beta_medinc = pm.Normal('beta_medinc', 0, 1)
        beta_house_age = pm.Normal('beta_house_age', 0, 1)
        beta_ave_rooms = pm.Normal('beta_ave_rooms', 0, 1)
        beta_ave_bedrms = pm.Normal('beta_ave_bedrms', 0, 1)
        beta_population = pm.Normal('beta_population', 0, 1)
        beta_ave_occup = pm.Normal('beta_ave_occup', 0, 1)
        sigma = pm.HalfCauchy("sigma", beta=1)

        # Logistic Regression
        mean = intercept[spatial_group_idx] + \
            beta_medinc * medinc + \
            beta_house_age * house_age + \
            beta_ave_rooms * ave_rooms + \
            beta_ave_bedrms * ave_bedrms + \
            beta_population * population + \
            beta_ave_occup * ave_occup

        likelihood = pm.Normal("y", mu=mean, sigma=sigma,
                               observed=median_house_value)

        return model


def create_blr_partial_pooling_gp(X: pd.DataFrame,
                                  y: pd.Series,
                                  coords: dict,
                                  spatial_grouping_var: str,
                                  spatial_groups_coordinates: np.array) -> pm.Model:

    with pm.Model(coords=coords) as model:

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
        spatial_groups_coords = pm.MutableData(
            "spatial_groups_coords", spatial_groups_coordinates)

        # Create features priors;
        # They are composed of state level means and the variation of the spatial groups represented by the gp priors
        intercept_mu = pm.Normal("intercept_mu", mu=0.0, sigma=1.0)
        intercept_sigma = pm.Exponential("intecept_sigma", 1)

        ls = pm.Gamma(name=f"intercept_ls", alpha=0, beta=1)
        amplitude = pm.Gamma(name=f"intercept_amplitude", alpha=0, beta=1)
        cov = amplitude ** 2 * pm.gp.cov.ExpQuad(input_dim=2, ls=ls)
        latent = pm.gp.Latent(cov_func=cov)
        gp = latent.prior(
            f"intercept_gp", X=spatial_groups_coords, jitter=1e-7)

        intercept = pm.Normal('intercept', intercept_mu,
                              intercept_sigma * gp, dims=("spatial_groups"))

        beta_medinc = pm.Normal('beta_medinc', 0, 1)
        beta_house_age = pm.Normal('beta_house_age', 0, 1)
        beta_ave_rooms = pm.Normal('beta_ave_rooms', 0, 1)
        beta_ave_bedrms = pm.Normal('beta_ave_bedrms', 0, 1)
        beta_population = pm.Normal('beta_population', 0, 1)
        beta_ave_occup = pm.Normal('beta_ave_occup', 0, 1)
        sigma = pm.HalfCauchy("sigma", beta=1)

        # Logistic Regression
        mean = intercept[spatial_group_idx] + \
            beta_medinc * medinc + \
            beta_house_age * house_age + \
            beta_ave_rooms * ave_rooms + \
            beta_ave_bedrms * ave_bedrms + \
            beta_population * population + \
            beta_ave_occup * ave_occup

        likelihood = pm.Normal("y", mu=mean, sigma=sigma,
                               observed=median_house_value)

    return model
