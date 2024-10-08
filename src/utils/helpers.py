def split_data(data, target_column):
    from sklearn.model_selection import train_test_split
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2)
