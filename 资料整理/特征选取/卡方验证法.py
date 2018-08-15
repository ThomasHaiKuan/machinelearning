###卡方验证法###
import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import chi2

matrix = np.array(X)
target = np.array(target)
temp = feature_selection.SelectKBest(chi2, k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# k: int
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

