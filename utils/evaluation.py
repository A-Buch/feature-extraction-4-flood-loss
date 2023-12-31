#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for model evaluation"""

import sys
import numpy as np
import pandas as pd
import functools

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from scipy import stats

import utils.feature_selection as fs
import utils.training as t
import utils.evaluation_metrics as em

#import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.packages import importr


# get R packages
#base = importr("base")
#nestedcv = importr("nestedcv")
#permimp = importr("permimp")  # conditional permutation feature importance
#pdp = importr("pdp")
#caret = importr('caret') # package version needs to be higher than  >=  6.0-90

# pandas.DataFrames to R dataframes 
#pandas2ri.activate()

mt = t.ModelFitting  # call Class for model training


class ModelEvaluation(object):
    """
    Model evaluation by nested cross-validation
    """        
    def __init__(self, models_trained_ncv, Xy, target_name, cv, kfolds, score_metrics, seed):
        #super(model_fitting, self).__init__()
        self.models_trained_ncv = models_trained_ncv
        self.X = pd.DataFrame(Xy.drop(target_name, axis=1))
        self.X = pd.DataFrame(
                MinMaxScaler().fit_transform(self.X),   
                columns=self.X.columns) 
        self.y: pd.DataFrame = Xy[target_name]
        self.outer_cv = cv
        self.k_folds:int = kfolds
        self.score_metrics = score_metrics
        self.seed: int = seed
        self.y_pred = None
        self.y_proba = None
        self.residuals = None
        self.p_values = None
        #self.final_model = mt.ModelFitting().final_model_fit() or via arg in run script   
 
        # self.metrics = {  # TODO impl as eval_set
        #     "name": self.name,
        #     "train": len(self.X_train),
        #     "test": len(self.X_test),
        # }


    def model_evaluate_ncv(self, prediction_method="predict"):
        """  
        Run and Evaluate sklearn model by nested cross-validation [outer folds]
        prediction_method (str): "predict" or "predict_proba"
        return: predict y and return model generalization perfromance
        """    
        ## predict y on outer folds of nested cv
        self.y_pred = cross_val_predict(
            self.models_trained_ncv,  # estimators from inner cv
            self.X, self.y,
            cv=self.k_folds, # KFold without repeats to have for each sample one predicted value 
            method=prediction_method
        )

        ## Probability predictions (self.y_pred is 2-dimensional: predicted probabilities and respective predictions)
        if prediction_method == "predict_proba":

            y_pred_proba = self.y_pred       # rename it to avoid name confusion

            ## store highest predicted probabilities and respective predictions
            self.y_pred = np.argmax(y_pred_proba, axis=1)
            self.y_proba = np.take_along_axis(
                y_pred_proba, 
                np.expand_dims(self.y_pred, axis=1), 
                axis=1
            )
            self.y_proba = self.y_proba.flatten()


        ## get generalization performance on outer folds of nested cv
        model_performance_ncv = cross_validate(
            self.models_trained_ncv, 
            self.X, self.y, 
            scoring=self.score_metrics,  # Strategies to evaluate the performance of the cross-validated model on the test set.
            cv=self.outer_cv, 
            return_estimator=True,
        )         
        try:
            print(
                "model performance measured in MAE (std) on outer CV: %.3f (%.3f)"%(
                    model_performance_ncv["test_MAE"].mean(), np.std(model_performance_ncv["test_MAE"])
                ))
        except KeyError:
            print(
                "model performance measured in Accuracy (std) on outer CV: %.3f (%.3f)"%(
                    model_performance_ncv["test_accuracy"].mean(), np.std(model_performance_ncv["test_accuracy"])
                ))
        self.calc_residuals()    

        return model_performance_ncv
    

    # def r_models_cv_predictions(self, idx=0):
    #     """
    #     Get y_pred and y_true for a certain model during CV in R
    #     model : fitted R model from nestedcv.train()
    #     idx (int): index position of trained model from inner cv
    #     return: pandas Dataframe with y_pred and y_test values 
    #     """
    #     robjects.r('''
    #         r_models_cv_predictions <- function(m, idx, verbose=FALSE) {
    #             m$outer_result[[idx]]$preds
    #         }
    #     ''') 
    #     r_model_prediction = robjects.globalenv['r_models_cv_predictions']

    #     return fs.r_dataframe_to_pandas(r_model_prediction(self.models_trained_ncv, idx))

    
    # def r_model_evaluate_ncv(self):
    #     """ 
    #     Evaluate R model by nested cross-validation [outer folds]
    #     return: predict y and return model generalization perfromance
    #     """
    #     ## get y_pred of all k-folds into one pd.DataFrame
    #     robjects.r('''
    #         r_get_y_pred <- function(model, verbose=FALSE) {
    #             model$outer_result
    #         }
    #     ''')
    #     r_get_y_pred = robjects.globalenv['r_get_y_pred'] 
    #     r_y_pred = r_get_y_pred(self.models_trained_ncv)

    #     df_y_pred = pd.DataFrame()
    #     for i in range(self.k_folds):
    #         df_y_pred = pd.concat(
    #             [df_y_pred, fs.r_dataframe_to_pandas(r_y_pred[i][0])], # r_y_pred[i][0] == y_pred of one outer fold
    #             axis=0)
    #     df_y_pred.reset_index(drop=True)

    #     y_true = np.array(df_y_pred["testy"])
    #     self.y_pred = np.array(df_y_pred["predy"])

    #     self.calc_residuals()

    #     ## TODO scores slightly differe from the average number of performance scores for eahc estimator
    #     # get generalization performance on outer folds of nested cv
    #     model_performance_ncv = {
    #         #"test_MAE_old" : base.summary(self.models_trained_ncv)[3][2],  # higher than from test_MAE
    #         "test_MAE" : mean_absolute_error(y_true, self.y_pred),
    #         "test_RMSE" : em.root_mean_squared_error(y_true, self.y_pred),
    #         "test_MBE" : em.mean_bias_error(y_true, self.y_pred),
    #         "test_R2": r2_score(y_true, self.y_pred),
    #         "test_SMAPE" : em.symmetric_mean_absolute_percentage_error(y_true, self.y_pred),
    #     }
    #     return model_performance_ncv


    def calc_residuals(self):
        """
        Get and store residuals
        return: residuals
        """
        self.residuals = pd.DataFrame(
            {
                "y_true": self.y,
                "y_pred": np.array(self.y_pred),
                "residuals": self.y - np.array(self.y_pred),
            },
            index=self.y.index,
        )
        return self.residuals
        

    ## FIXME errorneous calculation of p-values 
    # def calc_standard_error(self, y, y_pred, newX):  # TODO move them outside class or to utils.py
    #     """
    #     y (np.array): observed target values
    #     y_pred (np.array): predicted target values, in same length as y
    #     newX (np.array): contains values of X plus one column for the later intercept values
    #     return (np.array): standard error
    #     """
    #     MSE = (sum((y - y_pred)**2))/(len(newX)-len(newX[0]))
    #     ## MSE = (sum((y-y_pred)**2))/(len(newX)-len(X.columns))
    #     var_b = MSE*(np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    #     return np.sqrt(var_b)


    # def calc_p_values(self, ts_b, newX):
    #     """
    #     ts_b (np.array): t values derived by : coefficent values / standard errors
    #     newX (np.array): contains values of X plus one column for the later intercept values
    #     return (np.array): significance of coefficients (p-values)
    #     """
    #     p_values =  [2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]
    #     return p_values

    

    def calc_regression_coefficients(self, model):
        """
        Calculate regression coefficients and signficance from sklearn linear model
        final_model: fitted model from sklearn
        model_name (str): name of model
        return: pd.DataFrame with coefficents and their significance 
        """

        # ## make sanity check
        # if sanity_test:
        #     X_exog = MinMaxScaler().fit_transform(self.X)#, 
        #     y = self.y

        #     ## reference: p-values from statsmodels
        #     m = sm.OLS(y, sm.add_constant(X_exog))
        #     m_res = m.fit()
        #     #print(m_res.summary())
        #     p_values_reference = m_res.summary2().tables[1]['P>|t|']

        #     ## self calculated p-values
        #     reg = LinearRegression().fit(X_exog, y)
        #     y_pred_test = reg.predict(X_exog)
        #     coefs_intercept = np.append(reg.intercept_, list(reg.coef_))

        #     ## calc p-values
        #     newX = np.append(np.ones((len(X_exog),1)), X_exog, axis=1)
        #     sd_b = self.calc_standard_error(self.y, y_pred_test, newX)  # standard error calculated based on MSE of newX
        #     ts_b = coefs_intercept / sd_b        # t values
        #     p_values = self.calc_p_values(ts_b, newX)   # significance

        #     assert (list(np.round(p_values_reference, 3)) == np.round(p_values, 3)).all(), sys.exit("different calculation of p values")


        ## get coefficients and intercept
        model_coefs = model.named_steps['model'].coef_
        model_intercept = model.named_steps['model'].intercept_
        coefs_intercept = np.append(model_intercept, list(model_coefs))
        
        ## calc significance of coefficient,  modified based on : https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
        
        ## calc p-values
        ## FIXME errorneous calculation of p-values when moved inside the class
        # newX = np.append(np.ones((len(self.X),1)), self.X, axis=1)
        # sd_b = self.calc_standard_error(self.y, self.y_pred, newX)  # standard error calculated based on MSE of newX
        # ts_b = coefs_intercept / sd_b        # t values
        # p_values = self.calc_p_values(ts_b, newX)   # significance

        model_coef = pd.DataFrame(
            {
                "features": ["intercept"] + self.X.columns.to_list(),
                "coefficients": np.round(coefs_intercept, 4),
                # "standard errors": np.round(sd_b, 3),
                # "t values": np.round(ts_b, 3),
                # "probabilities": np.round(p_values, 5),
            }, index=range(len(coefs_intercept))
        )
        return model_coef




    def permutation_feature_importance(self, final_model, repeats=10):
    #def permutation_feature_importance(model, X_test, y_test, y_pred, criterion= r2_score):
        """
        Calculate permutation based feature importance , the importance scores represents the increase in model error
        final_model : final sklearn model       
        return: pd DataFrame with averaged importance scores
        """
        permutation_fi = permutation_importance(
            final_model, 
            self.X, self.y, 
            n_repeats=repeats, random_state=self.seed
        )

        return permutation_fi.importances_mean, permutation_fi.importances_std, permutation_fi.importances


    # def r_permutation_feature_importance(self, final_model):
    #     """  
    #     final_model: final R model
    #     return importance scores and standard deviations in R dataframe format
    #     """ 
    #     importances = permimp.permimp(
    #         final_model, 
    #         threshold=0.95, conditional=True, 
    #         progressbar=False
    #     )
    #     return importances



    ## @decorator(model=final_models_trained["crf"], Xy=eval_set_list["crf"]["crf"], target_name=target, feature_name="flowvelocity", scale=True) 
    ## not using decorator @
    def get_partial_dependence(self, **kwargs):
        """
        Derive partial dependences
        feature_name (str): 
        return: pd.DataFrame with 1 column named by feature_name contain gridvaleus and 1 column with partial dependences
        """
        model= kwargs["model"]  # TODO get from class properties
        Xy = kwargs["Xy"]
        y_name = kwargs["y_name"]
        feature_name = kwargs["feature_name"]
        scale = kwargs["scale"]

        X = Xy.dropna().drop(y_name, axis=1)

        # scale feature distributions in pd plots across models
        if scale:
            X =  pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

        partial_dep = partial_dependence(   
            model, X=X, features=feature_name, 
            grid_resolution=X.shape[0], kind="average", #**further_params,
        )
        partial_dep = pd.DataFrame(
            {
                feature_name: partial_dep.grid_values[0],
                "yhat": partial_dep.average[0],
            }
        )
        return partial_dep


    # ## decorator for R model
    # def decorator_func(self, model , Xy, y_name, feature_name, scale=True):
    #     """
    #     Decorator to get partial dependence instead of python-sklearn-model from R-party-model
    #     """
    #     def r_get_partial_dependence(func):
    #         @functools.wraps(func)  # preserve original func information
    #         def wrapper(*args, **kwargs):
    
    #             X = Xy.dropna().drop(y_name, axis=1)
    #             Xy = pd.concat([Xy[y_name], X], axis=1)

    #             robjects.r('''
    #                 r_partial_dependence <- function(model, df, predictor_name, verbose=FALSE) {
    #                     pdp::partial(model, train=df, pred.var=predictor_name, type="regression", plot=FALSE )  
    #                 }
    #             ''') #  , plot=FALSE --> to get pdp values
    #             r_partial_dependence = robjects.globalenv['r_partial_dependence'] 
                
    #             partial_dep = r_partial_dependence(model, Xy, feature_name)
                
    #             return fs.r_dataframe_to_pandas(partial_dep)
            
    #         return wrapper
    #     return r_get_partial_dependence
    




