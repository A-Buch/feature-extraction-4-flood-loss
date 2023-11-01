#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pipelines for feature selection"""

import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet
from xgboost import XGBRegressor


from sklearn.pipeline import Pipeline

import utils.settings as s
s.init()
seed = s.seed


def main():

    ############ Logistic Regression ##########

    pipe_logreg = Pipeline( steps = [
        ('scaler', MinMaxScaler()), 
        ('model', LogisticRegression(random_state=seed)) 
    ] )

    # Logistic Regression with Bagging
    ensemble_model = {
        'model': BaggingClassifier,   # default bootstrap=True
        'kwargs': {'estimator': LogisticRegression(random_state=seed),
                  }
    }
    pipe_logreg_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])


    ############  Elastic Net  ##################

    pipe_en = Pipeline( steps = [
        ('scaler', MinMaxScaler()), 
        #('model', SelectFromModel(
        #    ElasticNet(random_state=seed),
        #    max_features=10)
        ('model', ElasticNet(random_state=seed)),
    ])

    # Elastic Net with Bagging
    ensemble_model = {
        'model': BaggingRegressor,   # default bootstrap=True
        'kwargs': {'estimator': ElasticNet(random_state=seed),
                   'bootstrap': True,
                  }
    }
    pipe_en_bag = Pipeline([
        ('scaler', MinMaxScaler()),
        ('bagging', ensemble_model['model'] (**ensemble_model['kwargs']) )
    ])


    ############  XGBoost Regressor  ##################

    pipe_xgb = Pipeline(steps = [
        ('scaler', MinMaxScaler()), 
        ('model', XGBRegressor(random_state=seed)),
    ])

   ############  Random Forest Regressor  ##################

    pipe_rf = Pipeline(steps = [
        ('scaler', MinMaxScaler()), 
        ('model', RandomForestRegressor(random_state=seed)),
    ])



    ###########  Conditional Random Forest ##############

    pipe_crf = "cforest"
    ## --> it seems to be possible to incoporate R models into sklearn pipeline but for this usecase this implementation is out of scope
    ## R model is called directly in python scripts



    joblib.dump(pipe_logreg, './pipelines/pipe_logreg.pkl')
    joblib.dump(pipe_logreg_bag, './pipelines/pipe_logreg_bag.pkl')

    joblib.dump(pipe_en, './pipelines/pipe_en.pkl')
    joblib.dump(pipe_en_bag, './pipelines/pipe_en_bag.pkl')

    joblib.dump(pipe_xgb, './pipelines/pipe_xgb.pkl')

    joblib.dump(pipe_crf, './pipelines/pipe_crf.pkl')

    joblib.dump(pipe_rf, './pipelines/pipe_rf.pkl')


if __name__ == "__main__":
    main()
