import numpy as np


def contngncy(S1, S2, lavls):
    K = len(lavls)
    lavls = np.array(lavls)
    lavls = np.insert(lavls, 0, 1000000)
    cntgncy = np.zeros((K, K))

    for i in range(1, K + 1):
        for j in range(1, K + 1):
            count = 0
            for l in range(len(S1)):
                if (S1[l] >= lavls[j]) and (S1[l] < lavls[j - 1]):
                    if (S2[l] >= lavls[i]) and (S2[l] < lavls[i - 1]):
                        count += 1

            cntgncy[i - 1, j - 1] = count

    return cntgncy
