## TODO needs to be completed

import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

import utils.evaluation as e


### Test p-value calculation
# Test if p-value are derived in the correct way, this means that also t vlaues and standard errors have to be correct.
# Assert that self-calculated p-value is the same as the p-values derived by stats-package for a simple linear regression

def test_calc_p_values(df_Xy, target_name):
    """
    test self calcualted p-values of regression coefficients with the owns derived by statsmodels (reference)
    df_Xy (pd.DataFrame): with target and predictors
    target_name (str): name of target clomun
    """
    X_exog = MinMaxScaler().fit_transform(df_Xy.drop(target_name, axis=1))
    y = df_Xy[target_name]

    ## reference: p-values from statsmodels
    #X_exog = sm.add_constant(X_exog)
    m = sm.OLS(y, sm.add_constant(X_exog))
    m_res = m.fit()
    # print(m_res.summary())
    p_values_reference = m_res.summary2().tables[1]['P>|t|']

    ## self calculated p-values
    reg = LinearRegression().fit(X_exog, y)
    y_pred = reg.predict(X_exog)

    coefs_intercept = np.append(reg.intercept_, list(reg.coef_))

    ## calc p-values
    newX = np.append(np.ones((len(X_exog),1)), X_exog, axis=1)
    sd_b = e.calc_standard_error(y, y_pred, newX)  # with MSE of newX
    ts_b = coefs_intercept / sd_b        
    p_values = e.calc_p_values(ts_b, newX)   

    assert (list(np.round(p_values_reference, 3)) == np.round(p_values, 3)).all(), "different calcuation of p values"
