from scipy import stats
import pandas as pd
import numpy as np


def ttest_mean_price_difference_between_groups_after_filter(
        data, filtering_column, filtering_value, grouping_column, group1, group2, price_column='price'):
    data_filtered = data[data[filtering_column] == filtering_value]
    group1_data = data_filtered[data_filtered[grouping_column] == group1]
    group2_data = data_filtered[data_filtered[grouping_column] == group2]
    t_stat, p_val = stats.ttest_ind(group1_data[price_column], group2_data[price_column])
    return t_stat, p_val

def make_ttest_results_df(data, features, grouping_column, group_1, group_2):
    feature_list = []
    feature_value_list = []
    t_stat_list = []
    p_val_list = []
    for feature in features:
        for feature_value in data[feature].unique():
            t_stat, p_val = ttest_mean_price_difference_between_groups_after_filter(
                data, feature, feature_value, grouping_column, group_1, group_2)
            feature_list.append(feature)
            feature_value_list.append(feature_value)
            t_stat_list.append(t_stat)
            p_val_list.append(p_val)

    ttest_df = pd.DataFrame(
        {'feature': feature_list, 'feature_value': feature_value_list, 't_stat': t_stat_list, 'p_val': p_val_list}
        )
    
    return ttest_df

def get_linear_regression_p_values(X, y, model):
    # Extract coefficients
    coefficients = model.coef_
    # Compute p-values
    n = len(y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    var_X = np.var(X, axis=0)
    t_values = coefficients / np.sqrt(mse / var_X)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n - X.shape[1]))

    # Create a pandas DataFrame
    result_df = pd.DataFrame({
        'Attribute': X.columns,
        'Coefficient': coefficients,
        'P-value': p_values
    })

    return result_df.sort_values(by='P-value')

def get_linear_regression_f_test_results(X, y, model):
    n = len(y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.mean(residuals**2)
    y_mean = np.mean(y)
    ssr = np.sum((y_pred - y_mean)**2)
    f_stat = (ssr / X.shape[1]) / (mse / (n - X.shape[1] - 1))
    p_val = 1 - stats.f.cdf(f_stat, X.shape[1], n - X.shape[1] - 1)
    return f_stat, p_val

