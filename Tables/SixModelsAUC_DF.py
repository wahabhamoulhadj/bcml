from Tables.generate_soft_scores import generate_soft_scores
from Tables.splitTable import split_table
import glob


def six_models_prediction_data_frame(models, parameter_grid_models, dir_path, random_variable):
    dataframe_list = []  # Data Frame containing the AUC scores of 6 models and 27 DataSet [27 x 6]

    all_six_models_predict_proba = []  # Row vector of the Data Frame representing all 6 auc scores of
    # 6 models concerning a single dataset

    for count, fname in enumerate(glob.glob(dir_path)):
        rows = []
        predict_proba_rows = []
        X_train, X_test, Y_train, Y_test = split_table(fname, random_variable)
        for model, parameter_grid in zip(models, parameter_grid_models):
            auc = generate_soft_scores(model, X_train, X_test, Y_train, Y_test,
                                       parameter_grid)[1]
            rows.append(auc)
            predict_proba_rows.append(generate_soft_scores(model, X_train, X_test, Y_train, Y_test,
                                                           parameter_grid)[0])
            print(auc)

        dataframe_list.append(rows)
        all_six_models_predict_proba.append(predict_proba_rows)

    return all_six_models_predict_proba, dataframe_list
