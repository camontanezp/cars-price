import plotly.express as px 


def plot_avg_target_time_series_by_feature(data, feature, target='price'):
    data_sorted = data.sort_values('sold_at', ascending=True, inplace=False)
    data_sorted_grouped = data_sorted.groupby(
        [feature, 'sold_at'], as_index=False).agg({target: 'mean'}).rename(columns={target: 'avg_' + target})
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