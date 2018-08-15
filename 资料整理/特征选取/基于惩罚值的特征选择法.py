###基于惩罚值的特征选择###
import numpy as np
from sklearn import feature_selection
from sklearn.linear_model import LogisticRegression

matrix = np.array(arr0)
target = np.array(target)
temp = feature_selection.SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html