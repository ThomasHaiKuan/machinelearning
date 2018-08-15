###方差选择法###

import numpy as np
from sklearn import feature_selection

matrix = np.array(X)
temp = feature_selection.VarianceThreshold(threshold=t).fit(matrix)
scores = [np.var(el) for el in matrix.T]
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# t: float
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
