###基于递归特征消除的特征选择###
import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.RFE(estimator=LogisticRegression(), n_features_to_select=n_features).fit(matrix, target)
scores = temp.ranking_.tolist()
indx = temp.support_.tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# n-features: int
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html