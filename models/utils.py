import pickle
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def rfe(features, labels_by_class):
    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(lr, n_features_to_select=1000)
    rfe.fit(features, labels_by_class)
    
    with open('models/rfe_ranking.pkl', 'wb') as f:
        pickle.dump(rfe, f)
        
def get_rfe():
    with open('models/rfe_ranking.pkl', 'rb') as f:
        rfe = pickle.load(f)
    
    return rfe.ranking_, rfe.support_