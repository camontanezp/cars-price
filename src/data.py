from sklearn.model_selection import train_test_split


def get_season_us_by_month(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "autumn"
    else:
        return "error"
    
def convert_object_columns_to_category(data):
    object_columns = data.select_dtypes(include='object').columns
    for column in object_columns:
        data[column] = data[column].astype('category')
    return data

def get_train_test_data(data, features, target, test_size=0.2):
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X, y, X_train, X_test, y_train, y_test