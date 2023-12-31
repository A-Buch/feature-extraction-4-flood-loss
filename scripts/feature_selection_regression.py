
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data preprocessing for HCMC survey dataset"""

__author__ = "Anna Buch, Heidelberg University"
__email__ = "a.buch@stud.uni-heidelberg.de"

# ## Feature selection 
# Enitre workflow with all models for the target variables relative content loss and business reduction (degree of loss) as well for the binary version of relative content loss (chance of loss)
# 
# Due to the samll sample size a nested CV is used to have the possibility to even get generalization error, in the inner CV the best hyperaparamters based on k-fold are selected; in the outer cv the generalization error across all tested models is evaluated. A seprate unseen validation set as done by train-test split would have an insufficent small sample size.
# Nested CV is computationally intensive but with the samll sample size and a well chosen set of only most important hyperparameters this can be overcome.
# 
# - Logistic Regression (binary rcloss)
# - Elastic Net
# - eXtreme Gradient Boosting
# - Random Forest
# 

import sys
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
import itertools

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, "../")
import utils.feature_selection as fs
import utils.training as t
import utils.evaluation as e
import utils.evaluation_metrics as em
import utils.figures as f
import utils.settings as s
import utils.pipelines as p
import utils.preprocessing as pp

p.main()  # create/update model settings
#s.init()
seed = s.seed

pd.set_option('display.max_columns', None)
plt.figure(figsize=(20, 10))

import contextlib
import warnings
warnings.filterwarnings('ignore')


## user-input -- settings for entire script
parser = argparse.ArgumentParser()
parser.add_argument("aoi_and_floodtype") # eg. "german_flash", "german_fluvial" 
parser.add_argument("year")  # string e.g "2002", "2021", "combi"
args = parser.parse_args()
aoi_and_floodtype = args.aoi_and_floodtype
years = [args.year]

targets = ["rloss_b", "rloss_e", "rloss_gs"]
## settings for cv
kfolds_and_repeats = 2, 2  # <k-folds, repeats> for nested cv
cv = RepeatedKFold(n_splits=kfolds_and_repeats[0], n_repeats=kfolds_and_repeats[1], random_state=seed)


## save models and their evaluation in following folders:
Path(f"../models_trained/commercial/nested_cv_models/{aoi_and_floodtype}").mkdir(parents=True, exist_ok=True)
Path(f"../models_trained/commercial/final_models/{aoi_and_floodtype}").mkdir(parents=True, exist_ok=True)
Path(f"../models_evaluation/commercial/{aoi_and_floodtype}").mkdir(parents=True, exist_ok=True)
Path(f"../selected_features/commercial/{aoi_and_floodtype}").mkdir(parents=True, exist_ok=True)



## Fit model 
score_metrics = {
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "RMSE": make_scorer(em.root_mean_squared_error, greater_is_better=False),
    "MBE": make_scorer(em.mean_bias_error, greater_is_better=False),
    "R2": "r2",
    "SMAPE": make_scorer(em.symmetric_mean_absolute_percentage_error, greater_is_better=False)
}


## iterate over piplines. Each pipline contains a scaler and regressor (and optionally a bagging method) 
pipelines = ["pipe_en", "pipe_rf", "pipe_xgb"]  


for year, target in itertools.product(years, targets): # iterate over years/combined and targets

    print("\n ##########  Starting model processing for ", year, target, "##############")
    
    df_candidates = pd.read_csv(f"../input/{aoi_and_floodtype}/df_{year}_{target}_commercial_{aoi_and_floodtype.split('_')[-1]}.csv")
    print(df_candidates.shape)

    eval_sets = {}
    models_trained = {}
    final_models_trained = {}
    models_coef = {}
    predicted_values = {}
    df_feature_importances = pd.DataFrame(index=df_candidates.drop(target, axis=1).columns.to_list())
    models_scores = {}

    ## Load set of hyperparamters
    hyperparams_set = pp.load_config("../utils/hyperparameter_sets.json")


    for pipe_name in pipelines:

        model_name = pipe_name.split('_')[1]
        print( f"\nApplying {model_name} on {target} and {year}")

        df_Xy = df_candidates
        X_names = df_Xy.drop(target, axis=1).columns.to_list()

        # ## remove zero-loss records only for combined dataset
        # if target == "combi":
        #     print(f"Removing {df_Xy.loc[df_Xy[target]==0.0,:].shape[0]} zero loss records")
        #     df_Xy = df_Xy.loc[df_Xy[target]!=0.0,:]
        #     print(f"Keeping {df_Xy.shape} damage cases for model training and evaluation")


        ## drop samples where target is nan
        print(f"Dropping {df_Xy[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
        df_Xy = df_Xy[ ~df_Xy[f"{target}"].isna()]

        ## Elastic Net and Random Forest: drop samples where any value is nan
        if (model_name == "en") | (model_name == "rf"):
            df_Xy.dropna(inplace=True)

        print(
            "Using ",
            df_Xy.shape[0],
            " records, from those are ",
            {(df_Xy[target][df_Xy[target] == 0.0]).count()},
            " cases with zero-loss or zero-reduction",
        )

        X = df_Xy[X_names]
        y = df_Xy[target]

        ## load model pipelines and hyperparameter space
        pipe = joblib.load(f'./pipelines/{pipe_name}.pkl')
        param_space = hyperparams_set[f"{model_name}_hyperparameters"]

        ## if bagging is used
        if "bag" in pipe_name.split("_"):
            print(f"Testing {model_name} with bagging")
            param_space = { k.replace('model', 'bagging__estimator') : v for (k, v) in param_space.items()}

        ## fit model for unbiased model evaluation and for final model used for Feature importance, Partial Dependence etc.
        mf = t.ModelFitting(
            model=pipe, 
            Xy=df_Xy,
            target_name=target,
            param_space=hyperparams_set[f"{model_name}_hyperparameters"],
            tuning_score="neg_mean_absolute_error",
            cv=cv,
            kfolds_and_repeats=kfolds_and_repeats,
            seed=seed,
        )
        models_trained_ncv = mf.model_fit_ncv()

        # save models from nested cv and final model on entire ds
        joblib.dump(models_trained_ncv, f"../models_trained/commercial/nested_cv_models/{aoi_and_floodtype}/{model_name}_{target}_{year}_{aoi_and_floodtype}.joblib")
            
        ## evaluate model    
        me = e.ModelEvaluation(
            models_trained_ncv=models_trained_ncv, 
            Xy=df_Xy,
            target_name=target,
            score_metrics=score_metrics,
            cv=cv,
            kfolds=kfolds_and_repeats[0],
            seed=seed,
        )
        model_evaluation_results = me.model_evaluate_ncv()

        
         ## visual check if hyperparameter ranges are good or need to be adapted
        for i in range(len(model_evaluation_results["estimator"])):
            print(f"{model_name}: ", model_evaluation_results["estimator"][i].best_params_)


        ## store fitted models and their evaluation results for later 
        eval_sets[model_name] = df_Xy
        models_scores[model_name] =  {
            k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
        } # get evaluation scores, metric names start with "test_<metricname>"
        models_trained[f"{model_name}"] = models_trained_ncv
        predicted_values[model_name] = me.residuals


        ## Final model

        ## get final model based on best MAE score during outer cv
        best_idx = list(models_scores[model_name]["test_MAE"]).index(max(models_scores[model_name]["test_MAE"]))
        final_model = model_evaluation_results["estimator"][best_idx]
        print("used params for best model:", final_model.best_params_)  # use last model as the best one
        final_model = final_model.best_estimator_

        ## predict on entire dataset and save final model
        y_pred_final = final_model.predict(X) 
        final_models_trained[model_name] = final_model 
        joblib.dump(final_model, f"../models_trained/commercial/final_models/{aoi_and_floodtype}/{model_name}_{target}_{year}_{aoi_and_floodtype}.joblib")



        ## Feature importance of best model

        importances = me.permutation_feature_importance(final_model, repeats=5)

        print("\nSelect features based on permutation feature importance")
        df_importance = pd.DataFrame(
            {
                f"{model_name}_importances" : importances[0],   # averaged importnace scores across repeats
                f"{model_name}_importances_std" : importances[1]
            },
            index=X_names,
        )
        df_feature_importances = df_feature_importances.merge(
            df_importance[f"{model_name}_importances"],   # only use mean FI, drop std of FI
            left_index=True, right_index=True, how="outer")
        print("5 most important features:", df_importance.iloc[:5].index.to_list())
            

        ## regression coefficients and significance of linear models 
        with contextlib.suppress(Exception): 
            models_coef[model_name] = me.calc_regression_coefficients(final_model)
            outfile = f"../models_evaluation/commercial/{aoi_and_floodtype}/regression_coefficients_{model_name}_{target}_{year}_{aoi_and_floodtype}.xlsx"
            models_coef[model_name].round(3).to_excel(outfile, index=True)
            print("Regression Coefficients:\n", models_coef[model_name].sort_values("probabilities", ascending=False), f"\n.. saved to {outfile}")
    

        ## store fitted models and their evaluation results for later 
        eval_sets[model_name] = df_Xy
        models_scores[model_name] =  {
            k: model_evaluation_results[k] for k in tuple("test_" + s for s in list(score_metrics.keys()))
        } # get evaluation scores, metric names start with "test_<metricname>"
        models_trained[f"{model_name}"] = models_trained_ncv
        predicted_values[model_name] = me.residuals
        final_models_trained[model_name] = final_model



    # ## Evaluation

    ## Evaluate models based on performance on outer cross-validation 
    ## TODO remove overhead
    xgb_model_evaluation = pd.DataFrame(models_scores["xgb"]).mean(axis=0)  # get mean of outer cv metrics (negative MAE and neg RMSE, pos. R2, pos MBE, posSMAPE)
    xgb_model_evaluation_std = pd.DataFrame(models_scores["xgb"]).std(axis=0)   # get respective standard deviations
    rf_model_evaluation = pd.DataFrame(models_scores["rf"]).mean(axis=0)
    rf_model_evaluation_std = pd.DataFrame(models_scores["rf"]).std(axis=0)
    en_model_evaluation = pd.DataFrame(models_scores["en"]).mean(axis=0)
    en_model_evaluation_std = pd.DataFrame(models_scores["en"]).std(axis=0)

    model_evaluation = pd.concat([en_model_evaluation, en_model_evaluation_std, xgb_model_evaluation, xgb_model_evaluation_std, rf_model_evaluation, rf_model_evaluation_std], axis=1)
    model_evaluation.columns = ["en_score", "en_score_std", "xgb_score", "xgb_score_std", "rf_score", "rf_score_std"]

    model_evaluation.index = model_evaluation.index.str.replace("test_", "")
    model_evaluation.loc["MAE"] = model_evaluation.loc["MAE"].abs()
    model_evaluation.loc["RMSE"] = model_evaluation.loc["RMSE"].abs()

    outfile = f"../models_evaluation/commercial/{aoi_and_floodtype}/performance_{target}_{year}_{aoi_and_floodtype}.xlsx"
    model_evaluation.round(3).to_excel(outfile, index=True)
    print("Outer evaluation scores:\n", model_evaluation.round(3), f"\n.. saved to {outfile}")



    ## Feature Importances 

    #### prepare Feature Importances 
    ## Have the same feature importance method across all applied ML models
    ## Weight Importances by model performance on outer loop (mean MAE)
    ## **Overall FI ranking (procedure similar to Rözer et al 2019; Brill 2022)**

    ## weight FI scores based on performance ; weigth importances from better performed models stronger
    model_weights =  {
    "xgb_importances" : np.abs(models_scores["xgb"]["test_MAE"].mean()),
    "en_importances" : np.abs(models_scores["en"]["test_MAE"].mean()),
    "rf_importances" : np.abs(models_scores["rf"]["test_MAE"].mean()),
    }

    df_feature_importances_w = fs.calc_weighted_sum_feature_importances(df_feature_importances, model_weights)
    df_feature_importances_w.head(5)


    ####  Plot Feature importances

    ## the best model has the highest weighted feature importance value
    # df_feature_importances_w.describe()

    df_feature_importances_plot = df_feature_importances_w

    ## drop features which dont reduce the loss
    #df_feature_importances_plot = df_feature_importances_plot.loc[df_feature_importances_plot.weighted_sum_importances > 2, : ] 

    f.plot_stacked_feature_importances(
        df_feature_importances_plot[["rf_importances_weighted", "en_importances_weighted", "xgb_importances_weighted",]],
        target_name=target,
        model_names_plot = ("Random Forest", "Elastic Net", "XGBoost"),
        outfile=f"../models_evaluation/commercial/{aoi_and_floodtype}/feature_importances_{target}_{year}_{aoi_and_floodtype}.jpg"
    )


    ### Save final feature space 
    ## The final selection of features is used later for the non-parametric Bayesian Network

    ## drop records with missing target values
    print(f"Dropping {df_candidates[f'{target}'].isna().sum()} records from entire dataset due that these values are nan in target variable")
    df_candidates = df_candidates[ ~df_candidates[target].isna()]
    print(f"Keeping {df_candidates.shape[0]} records and {df_candidates.shape[1]} features")


    ## sort features by their overall importance (weighted sum across across all features) 
    final_feature_names = df_feature_importances_w["weighted_sum_importances"].sort_values(ascending=False).index##[:10]
    print(final_feature_names)

    ## save importnat features, first column contains target variable
    fs.save_selected_features(
        df_candidates, 
        pd.DataFrame(df_candidates, columns=[target]), 
        final_feature_names,
        filename=f"../selected_features/commercial/{aoi_and_floodtype}/final_predictors_{target}_{year}_{aoi_and_floodtype}.xlsx"
    )


    ### Partial dependence
    ## PDP shows the marginal effect that one or two features have on the predicted outcome.


    ## store partial dependences for each model
    pdp_features = {a : {} for a in ["en", "xgb", "rf"]}


    ## get partial dependences
    for model_name in ["xgb", "en", "rf"]:

        Xy_pdp = eval_sets[model_name].dropna() #  solve bug on sklearn.partial_dependece() which can not deal with NAN values
        X_pdp, y_pdp = Xy_pdp[Xy_pdp.columns.drop(target)], Xy_pdp[target]
        X_pdp = pd.DataFrame(
            MinMaxScaler().fit_transform(X_pdp), # for same scaled pd plots across models
            columns=X.columns
            )
        Xy_pdp = pd.concat([y_pdp, X_pdp], axis=1)

        for predictor_name in X.columns.to_list(): 
            features_info =  {
                #"percentiles" : (0.05, .95) # causes NAN for some variables for XGB if (0, 1)
                "model" : final_models_trained[model_name], 
                "Xy" : Xy_pdp, 
                "y_name" : target, 
                "feature_name" : predictor_name, 
                "scale"  : True
            }         
            if model_name != "crf":   
                # print(predictor_name)
                partial_dep = me.get_partial_dependence(**features_info)

            pdp_features[model_name][predictor_name] = partial_dep



    plt.figure(figsize=(10,25))
    # plt.suptitle(f"Partial Dependences for {target}", fontsize=18, y=0.95)


    most_important_features = df_feature_importances_plot.sort_values("weighted_sum_importances", ascending=False).index

    categorical = [] # ["flowvelocity", "further_variables .."]
    ncols = 3
    nrows = len(most_important_features[:10])
    idx = 0

    ## create PDP for all three models
    for feature in most_important_features[:10]:
        for model_name, color, idx_col in zip(["rf", "en", "xgb"], ["darkblue", "steelblue","grey"], [0, 1, 2]):
            
            # idx position of subplot
            ax = plt.subplot(nrows, ncols, idx + 1 + idx_col)
            feature_info = {"color" : color, "ax" : ax} 

            # plot
            df_pd_feature = pdp_features[model_name][feature]  
            f.plot_partial_dependence(
                df_pd_feature, feature_name=feature, partial_dependence_name="yhat", 
                categorical=[],
                outfile=f"../models_evaluation/commercial/{aoi_and_floodtype}/pdp_{target}_{year}_{aoi_and_floodtype}.jpg",
                **feature_info
                )

        idx = idx + 3



    # ### Empirical median ~ predicted median
    # Compare median and mean of predicted  vs observed target values
    for k,v in predicted_values.items():
        print(f"\n{k}")
        print(em.empirical_vs_predicted(predicted_values[k]["y_true"], predicted_values[k]["y_pred"]))


    # ### Plot prediction error 
    f.plot_residuals(
        residuals=predicted_values, 
        model_names_abbreviation=["rf", "en", "xgb"],  
        model_names_plot=["Random Forest", "Elastic Net", "XGBoost"],
        outfile=f"../models_evaluation/commercial/{aoi_and_floodtype}/residuals_{target}_{year}_{aoi_and_floodtype}.jpg"
    )


    print(f"Finished processing for target {target}")  # TODO add time measure at least for nested cv

