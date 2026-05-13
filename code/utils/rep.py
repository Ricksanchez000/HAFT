
def feature_state_generation(X):
    return _feature_state_generation_des(X)


def _feature_state_generation_des(X):  #X, a pandas dataframe
    feature_matrix = []
    for i in range(8):
        feature_matrix = feature_matrix + list(X.astype(np.float64).
                                               describe().iloc[i, :].describe().fillna(0).values)
    return feature_matrix  # a flattened list, yeah, we don't want matrix anyway

