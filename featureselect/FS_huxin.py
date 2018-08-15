###互信息选择###
from minepy import MINE
import numpy as np
from sklearn import feature_selection

matrix = np.array(X)
target = np.array(target)
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)
temp = feature_selection.SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=k).fit(matrix, target)
scores = temp.scores_.tolist()
indx = temp.get_support().tolist()
result = temp.transform(matrix).tolist()
return scores, indx, result

# X: array-like
# target: array-like
# k: int
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html