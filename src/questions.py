from scipy import stats


def ttest_mean_price_difference_between_groups_after_filter(
        data, filtering_column, filtering_value, grouping_column, group1, group2, price_column='price'):
    data_filtered = data[data[filtering_column] == filtering_value]
    group1_data = data_filtered[data_filtered[grouping_column] == group1]
    group2_data = data_filtered[data_filtered[grouping_column] == group2]
    t_stat, p_val = stats.ttest_ind(group1_data[price_column], group2_data[price_column])
    return t_stat, p_val