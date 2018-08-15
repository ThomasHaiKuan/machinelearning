##基于树模型的特征选择##
import numpy as np
from sklearn import feature_selection
from sklearn.ensemble import GradientBoostingClassifier

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectFromModel(GradientBoostingClassifier()).fit(matrix, target)
indx = temp._get_support_mask().tolist()
scores = get_importance(temp.estimator_).tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result