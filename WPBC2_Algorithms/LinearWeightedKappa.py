from WPBC2_Algorithms.WPBC2_Contngncy import contngncy
from WPIBC_Algorithm.PBC_wtkappa import wtkappa


def linear_weighted_kappa(base_soft_detector, soft_detectors_score, nb_thresh):
    weighted_mat = [[0]*nb_thresh for _ in range(nb_thresh)]
    contngncy_mat = contngncy(base_soft_detector, soft_detectors_score, nb_thresh)

    for i in range(len(weighted_mat)):
        for j in range(len(weighted_mat)):
            weighted_mat[i][j] = 1 - (abs(i-j) / (nb_thresh-1))





    return wtkappa(contngncy_mat, weighted_mat)
#
# S1 = [3.9, 4.3, 3.4, 3.2, 3.6, 3.1, 3.0, 3.0, 2.3, 2.6, 2.5, 2.2, 0.8, 1.4, 0.8, 1.9, 0.8, 1.1, 1.3, 0.8, 0.8, 0.8, 1.1, 10.5, 0.8 ]
# S2 = [3.2, 3.4, 3.1, 3.0, 3.7, 3.0, 3.0, 3.0, 2.3, 2.5, 2.5, 2.2, 0.9, 1.2, 0.9, 1.8, 0.9, 1.0, 1.2, 0.9, 0.9, 0.9, 1.3, 10.8, 0.9 ]
#
# print(contngncy(S1, S2, 9))
# print(linear_weighted_kappa(S1, S2, 12))