def tpr_fpr(actual, predicted):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(len(actual)):
        if actual[i] == 1 and predicted[i] == 1:
            true_pos += 1
        elif actual[i] == 0 and predicted[i] == 0:
            true_neg += 1
        elif actual[i] == 1 and predicted[i] == 0:
            false_neg += 1
        elif actual[i] == 0 and predicted[i] == 1:
            false_pos += 1

    try:
        tpr = true_pos / (true_pos + false_neg)
    except:
        tpr = 0

    try:
        fpr = false_pos / (false_pos + true_neg)
    except:
        fpr = 0

    return [tpr, fpr]
