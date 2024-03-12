import plotly.express as px 
import shap

# print the JS visualization code to the notebook
shap.initjs()


def plot_avg_target_time_series_by_feature(data, feature, target='price'):
    data_sorted = data.sort_values('sold_at', ascending=True, inplace=False)
    data_sorted_grouped = data_sorted.groupby(
        [feature, 'sold_at'], as_index=False, observed=True).agg({target: 'mean'}).rename(columns={target: 'avg_' + target})
    fig = px.line(data_sorted_grouped, x='sold_at', y='avg_' + target, color=feature)
    fig.update_layout(title=f'Average {target} time series by {feature}')
    fig.show()

def plot_avg_target_time_series_by_features(data, features, target='price'):
    for feature in features:
        plot_avg_target_time_series_by_feature(data, feature, target='price')

def plot_distribution_of_feature(data, feature):
    fig = px.histogram(data, x=feature)
    fig.update_layout(title=f'Distribution of {feature}')
    fig.show()

def plot_distribution_of_features(data, features):
    for feature in features:
        plot_distribution_of_feature(data, feature)

def plot_distribution_of_target_by_feature(data, feature, target='price'):
    fig = px.box(data, x=feature, y=target)
    fig.update_layout(title=f'Distribution of price by {feature}')
    fig.show()

def plot_scatter_of_target_vs_feature(data, feature, target='price'):
    fig = px.scatter(data, x=feature, y=target)
    fig.update_layout(title=f'{target} vs {feature}')
    fig.show()

def plot_scatter_of_target_vs_features(data, features, target='price'):
    for feature in features:
        plot_scatter_of_target_vs_feature(data, feature, target='price')

def plot_shap_summary(metadata, model, X):
    if 'XGBoost linear' in metadata["name"]:
        print('Linear model detected. Shap explanation not supported.')
        explainer = None
    elif 'Linear regression model' in metadata["name"]:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=metadata['features'], plot_type='bar')
    elif 'XGBoost with trees' in metadata["name"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=metadata['features'], plot_type='bar')
    else:
        raise ValueError('Model type not supported')
    
    return explainer

def plot_shap_waterfall_for_car_sample(metadata, explainer, X_for_shap, car_index):
    if 'XGBoost linear' in metadata["name"]:
        print('Linear model detected. Shap explanation not supported.')
        plot = None
    elif 'Linear regression model' in metadata["name"] or 'XGBoost with trees' in metadata["name"]:
        shap_values = explainer(X_for_shap)
        plot = shap.plots.waterfall(shap_values[car_index])
    else:
        raise ValueError('Model type not supported')
    
