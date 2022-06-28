import numpy as np
from scipy.stats.mstats import mquantiles


def sample_scores(scores, nb_thresh=None):
    if nb_thresh is None:
        thres = np.array(scores)

    else:
        try:
            if len(nb_thresh) == 0:
                nb_thresh = []
                thres = np.array(scores)
        except:
            thres = mquantiles(scores, np.linspace(0, 1, num= nb_thresh))
            thres = np.append(thres, 1.1)
            thres = np.append(thres, -0.1)
            thres = np.array(thres)
            thres = np.sort(thres)

    return thres

# print(sample_scores( np.load('../Tables/all_six_models_predict_proba.npy', allow_pickle=True)[0][0], 10))