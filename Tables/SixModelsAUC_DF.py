from matplotlib import pyplot as plt

from Tables.generate_soft_scores import generate_soft_scores
from Tables.splitTable import split_table
import glob


def six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable):
    dataframe_list = []  # Data Frame containing the AUC scores of 6 models and 27 DataSet [27 x 6]

    all_six_models_predict_proba = []  # Row vector of the Data Frame representing all 6 auc scores of
    # 6 models concerning a single dataset
    all_six_models_fpr = []
    all_six_models_tpr = []
    for count, fname in enumerate(glob.glob(dir_path)):
        rows = []
        predict_proba_rows = []
        roc_curves_fpr_list = []
        roc_curves_tpr_list = []
        X_train, X_test, Y_train, Y_test = split_table(fname, random_variable)
        for model, parameter_grid in zip(models, parameter_grid_models):
            gss = generate_soft_scores(model, X_train, X_test, Y_train, Y_test,
                                       parameter_grid)
            rows.append(gss[1])
            predict_proba_rows.append(gss[0])
            roc_curves_fpr_list.append(gss[2])
            roc_curves_tpr_list.append(gss[3])
            print(gss[1])

        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(roc_curves_fpr_list[0], roc_curves_tpr_list[0], label="Naive Bayes")
        # plt.plot(roc_curves_fpr_list[1], roc_curves_tpr_list[1], label="RandomForest")
        # plt.plot(roc_curves_fpr_list[2], roc_curves_tpr_list[2], label="DecisionTree")
        # plt.plot(roc_curves_fpr_list[3], roc_curves_tpr_list[3], label="LogisticRegression")
        # plt.plot(roc_curves_fpr_list[4], roc_curves_tpr_list[4], label="KNN")
        # plt.plot(roc_curves_fpr_list[5], roc_curves_tpr_list[5], label="SVM")
        # plt.legend(["Naive Bayes AUC : {}".format(rows[0]),
        #             "Random Forest AUC : {}".format(rows[1]),
        #             "Decision Tree AUC : {}".format(rows[2]),
        #             "Logistic Regression AUC : {}".format(rows[3]),
        #             "KNN AUC{} : ".format(rows[4]),
        #             "SVC AUC : {} ".format(rows[5])], loc='lower right')
        # plt.xlabel("FPR")
        # plt.ylabel("TPR")
        # plt.title('Receiver Operating Characteristic')
        # plt.show()

        dataframe_list.append(rows)
        all_six_models_predict_proba.append(predict_proba_rows)
        all_six_models_fpr.append(roc_curves_fpr_list)
        all_six_models_tpr.append(roc_curves_tpr_list)

    return all_six_models_predict_proba, dataframe_list, all_six_models_fpr, all_six_models_tpr
