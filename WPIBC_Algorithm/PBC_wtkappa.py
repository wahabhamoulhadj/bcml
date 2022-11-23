def wtkappa(ctgncy, wt):
    n = ctgncy.sum()
    ctgncy = ctgncy / n
    r = ctgncy.sum(axis=1, dtype='float')
    r = r.reshape((len(ctgncy), 1))
    s = ctgncy.sum(axis=0, dtype='float')
    Ex = r * s

    po = (ctgncy * wt).sum()
    pe = (Ex * wt).sum()
    kp_cof = (po - pe) / (1 - pe)

    return kp_cof
