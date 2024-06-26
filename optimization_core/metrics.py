""" This module contains both standard metrics from libraries and custom ones for evaluation of our models.
    The idea is to gather all of them in one module for better imports and referencing.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import os


def prediction_refactoring(pred, class_mode="binary", n_classes=2, threshold=None, majority=False):
    """ Takes the predictions and translates it to a format for the evaluation metrics.
        Args:
            pred (list): contains all predictions of the dataset in a single array
            mode (string):
            threshold (None): depends on mode and defines when a value belongs to a certain class
        Case 1:
            For binary refactor shape [1, 2] into [1] with labels [0, 1] -> [0] and [1, 0] -> [1]
        Case 2:
            For multiclass
        NOTE: only works correctly if prediction per sample sums up to 1
    """
    pred_ref = []
    if class_mode == "binary" and n_classes == 2:
        # The binary case delivers single outputs of size [1, 2]
        threshold = 0.5 if threshold is None else threshold
        pred = np.asarray(pred).reshape([-1, 2])  # Removes intermediate dimension by batch sizes > 1 if existent
        [pred_ref.append(1.) if pred[i][0] >= threshold else pred_ref.append(0.) for i in range(0, len(pred))]

    elif class_mode == "multi":
        # pred is now a list of arrays with shape (1, n_classes)
        pred = np.asarray(pred).reshape([-1, n_classes])  # Removes intermediate dimension by batch sizes > 1
        pred_ref = np.zeros_like(pred)
        if majority:
            for i in range(0, len(pred)):
                if pred[i, np.argmax(pred[i])].item(0) == np.max(pred[i]):
                    pred_ref[i, np.argmax(pred[i])] = 1
        else:
            threshold = 0.5 if threshold is None else threshold

            for i in range(0, len(pred)):
                if pred[i, np.argmax(pred[i])].item(0) >= threshold:
                    pred_ref[i, np.argmax(pred[i])] = 1

    return pred_ref


pred_test = [[0.77, 0.23], [0.22, 0.78], [0.5, 0.5]]
pred_ref = prediction_refactoring(pred_test, "binary")
# print(pred_ref)


def label_refactoring(labels, species_labels, class_mode="binary", n_classes=2):
    """ Takes the labels and translates it to a format for the evaluation metrics.
        Args:
            labels (list): contains all labels of the dataset in a single array
            mode (string):
        Case 1:
            For binary, refactor shape [1, 2] into [1] with labels [0, 1] -> [0] and [1, 0] -> [1]
        Case 2:
            For multiclass
        NOTE: only works correctly if labels are one-hot-encoded
    """
    labels_ref = []
    if class_mode == "binary" and n_classes == 2:  # Note: can also be replaced by the structure in class_mode == "multi"
        # The binary case delivers single labels of size [1, 2]
        labels = np.asarray(labels).reshape([-1, 2])  # Removes intermediate dimension by batch sizes > 1 if existent
        [labels_ref.append(1.) if labels[i][0] == 1. else labels_ref.append(0.) for i in range(0, len(labels))]
    elif class_mode == "multi":
        species_labels = np.asarray(species_labels).reshape([-1, n_classes])
        labels_ref = species_labels
    return labels_ref


labels_test = [[0., 1.], [0., 1.], [1., 0.]]
labels_ref = label_refactoring(labels_test, "binary")
# print(labels_ref)


def standard_metrics(labels, pred, class_mode="binary"):
    """ Function takes the predictions and corresponding labels once and returns all standard metrics from it.
        class_mode == binary can be removed and both conditions can be treated equally.
    """
    if class_mode == "binary":
        accuracy = accuracy_score(labels, pred)
        precision = precision_score(labels, pred)
        recall = recall_score(labels, pred)
        f1 = f1_score(labels, pred)
        f1_score_class_based = f1_score(labels, pred, average=None)
        print("Metrics computed!")
        return accuracy, precision, recall, f1, f1_score_class_based

    else:
        accuracy = accuracy_score(labels, pred)
        precision = np.mean(precision_score(labels, pred, average=None))
        recall = np.mean(recall_score(labels, pred, average=None))
        f1 = np.mean(f1_score(labels, pred, average=None))
        precision_class_based = precision_score(labels, pred, average=None)
        recall_class_based = recall_score(labels, pred, average=None)
        f1_score_class_based = f1_score(labels, pred, average=None)
        print("Metrics computed!")
        return accuracy, precision, recall, f1, f1_score_class_based


def multi_class_label_translator(bat_class_list):
    """ Turns string into zero list with value 1 at the respective species position. """
    trans_dict = {}
    for index, bat_species in enumerate(bat_class_list):
        trans_array_dummy = np.zeros((len(bat_class_list)))
        trans_array_dummy[index] = 1
        trans_dict[bat_species] = trans_array_dummy
    return trans_dict


def compute_confusion_matrix(labels, preds, bat_class_list, n_classes=2, store="False", mode="&Test&", epoch=0,
                             output_path=""):
    """ Computes the confusion matrix between predictions and labels.
        Expected shape is the full list of evaluated samples per validation step.
        Ideally, the intermediate batch dimension was previously removed.
        Multi-label information must be one-hot-encoded to work correctly.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    # Remove intermediate batch dimension if existent
    if labels.ndim >= 3:
        labels = np.asarray(labels).reshape([-1, n_classes])
    if preds.ndim >= 3:
        preds = np.asarray(preds).reshape([-1, n_classes])
    # Check for multi-label format and refactor to single integer label format
    try:
        if len(labels[0]) > 1:
            labels = np.argmax(labels, axis=1)
    except TypeError:
        print("Entry of labels array has no len().")
    try:
        if len(preds[0]) > 1:
            preds = np.argmax(preds, axis=1)
    except TypeError:
        print("Entry of preds array has no len().")

    conf_mat = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=bat_class_list)
    disp.plot()
    if store == "True":
        plt.savefig(f"{output_path}/confusion_matrix_{mode}_Epoch_{epoch}.png", dpi=300)
    return conf_mat


def compute_avg_confusion_matrix(list_conf_mat_all_runs, bat_class_list, store="False", mode="&Test&",
                                 name_postfix="Undefined", output_path=""):
    """ Currently designed to take a list of the confusion matrix for all runs of the same config.
        Each confusion matrix must be a flattened list.
    """
    conf_mats = np.asarray(list_conf_mat_all_runs)
    print("all: ", conf_mats)
    avg_conf_mat = np.around(np.reshape(np.mean(conf_mats, axis=0), [len(bat_class_list), len(bat_class_list)]))
    print("AVG: ", avg_conf_mat)

    disp = ConfusionMatrixDisplay(avg_conf_mat, display_labels=bat_class_list)
    disp.plot(values_format=".5g")
    if store == "True":
        plt.savefig(f"{output_path}/avg_confusion_matrix_{mode}_{name_postfix}.png", dpi=300)
    return avg_conf_mat


def species_to_genus_translator(species_labels, species_to_combine, genus_name):
    """ Translates the species labels that are single labels into an encompassing genus """
    species_labels_new = []
    for species in species_labels:
        if species in species_to_combine:
            species_labels_new.append(genus_name)
        else:
            species_labels_new.append(species)  # usually another genus that does not require translation
    return species_labels_new


def hard_case_image_path_tracing(preds, labels, species_labels, species_t_labels, image_path_collector, class_mode,
                                 n_classes, score_threshold, score_by_majority):
    """ Compares refactored predictions with labels and
        tracks down the corresponding image path of all cases of mismatch """

    hard_case_collector = []
    # remove batch dimension in pred, label and image_path_collector (for later return)
    preds_unbatched = np.asarray(preds).reshape([-1, n_classes])
    labels_unbatched = np.asarray(labels).reshape([-1, n_classes]) if class_mode == "binary" else \
        np.asarray(species_labels).reshape([-1, n_classes])
    image_path_collector_unbatched = np.asarray(image_path_collector).reshape([-1, 1])
    t_labels_unbatched = np.asarray(species_t_labels).reshape([-1, 1])

    # create pred_ref and label_ref
    preds_ref = prediction_refactoring(preds, class_mode=class_mode, n_classes=n_classes,
                                       threshold=score_threshold, majority=score_by_majority)
    labels_ref = label_refactoring(labels, species_labels, class_mode=class_mode, n_classes=n_classes)
    # compare pred_ref - label_ref
    # track index, use index for image_path_collector
    counter = 0
    for index, (pred_r, label_r) in enumerate(zip(preds_ref, labels_ref)):
        pred_r = np.asarray(pred_r)
        label_r = np.asarray(label_r)
        if not (pred_r == label_r).all():  # ndarray of bools checked for total true value
            counter += 1
            hard_case_collector.append([[round(x, 3) for x in preds_unbatched[index]], labels_unbatched[index],
                                        t_labels_unbatched[index], image_path_collector_unbatched[index]])
            # hard_case_collector is now a list of lists with three entries per sublist.
            # first entry is sublist of the prediction, second is the same with labels and last entry is the filepath
    # return the image_path and corresponding prediction and maybe label for now
    return hard_case_collector, counter


def performance_overview_collector(dir_df_overview, epoch, f1_cb_train, f1_cb_val, f1_cb_test,
                                   conf_mat_train, conf_mat_val, conf_mat_test, class_list):
    """ Consider a separation by runs. Bat-Noise-Classification still tracks wrong f1-scores """
    if os.path.exists(dir_df_overview):
        df = pd.read_csv(dir_df_overview)
    else:
        df = pd.DataFrame()

    epoch_dict = {"epoch": epoch}
    f1_cb_train_dict = {f"f1_cb_train_{class_list[i]}": f1_cb_train[i] for i in range(len(class_list))}
    f1_cb_val_dict = {f"f1_cb_val_{class_list[i]}": f1_cb_val[i] for i in range(len(class_list))}
    f1_cb_test_dict = {f"f1_cb_test_{class_list[i]}": f1_cb_test[i] for i in range(len(class_list))}
    conf_mat_train_dict = {"conf_mat_train": conf_mat_train.flatten()}  # use .tolist() to store with commas in csv-file
    conf_mat_val_dict = {"conf_mat_val": conf_mat_val.flatten()}
    conf_mat_test_dict = {"conf_mat_test": conf_mat_test.flatten()}
    overview_dict = {**epoch_dict, **f1_cb_train_dict, **f1_cb_val_dict, **f1_cb_test_dict,
                     **conf_mat_train_dict, **conf_mat_val_dict, **conf_mat_test_dict}
    # overview_dict = {**epoch_dict, **f1_cb_train_dict, **f1_cb_val_dict, **f1_cb_test_dict}

    print(overview_dict)
    new_df = pd.DataFrame([overview_dict])
    combined_df = df.append(new_df, sort=False)
    combined_df.to_csv(dir_df_overview, index=False)


def load_eval_csv_create_avg(eval_csvs, output_main_path=None):
    """ Load the performance overview of all single runs collected in a csv-file and
        compute the average values over all runs and store it in one csv-file. """
    df_averaging_collector_f1 = []
    df_averaging_collector_conf_mat = []

    for eval_csv in eval_csvs:
        df_eval_f1 = pd.read_csv(eval_csv)

        # Variants on bat noise classification
        df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Bat", "f1_cb_val_Bat", "f1_cb_test_Bat",
                                 "f1_cb_train_Noise", "f1_cb_val_Noise", "f1_cb_test_Noise"]]
        # Variants on species classification
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Ppip", "f1_cb_val_Ppip", "f1_cb_test_Ppip",
        #                          "f1_cb_train_Pnat", "f1_cb_val_Pnat", "f1_cb_test_Pnat"]]
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Ppip", "f1_cb_val_Ppip", "f1_cb_test_Ppip",
        #                          "f1_cb_train_Pnat", "f1_cb_val_Pnat", "f1_cb_test_Pnat",
        #                          "f1_cb_train_Ppyg", "f1_cb_val_Ppyg", "f1_cb_test_Ppyg"]]
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Ppip", "f1_cb_val_Ppip", "f1_cb_test_Ppip",
        #                          "f1_cb_train_Pnat", "f1_cb_val_Pnat", "f1_cb_test_Pnat",
        #                          "f1_cb_train_Ppyg", "f1_cb_val_Ppyg", "f1_cb_test_Ppyg",
        #                          "f1_cb_train_Ptief", "f1_cb_val_Ptief", "f1_cb_test_Ptief"]]
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Ppip", "f1_cb_val_Ppip", "f1_cb_test_Ppip",
        #                          "f1_cb_train_Pnat", "f1_cb_val_Pnat", "f1_cb_test_Pnat",
        #                          "f1_cb_train_Ppyg", "f1_cb_val_Ppyg", "f1_cb_test_Ppyg",
        #                          "f1_cb_train_Ptief", "f1_cb_val_Ptief", "f1_cb_test_Ptief",
        #                          "f1_cb_train_Phoch", "f1_cb_val_Phoch", "f1_cb_test_Phoch"]]

        # Variants on genus classification
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Nyctaloid", "f1_cb_val_Nyctaloid", "f1_cb_test_Nyctaloid",
        #                          "f1_cb_train_Myotis", "f1_cb_val_Myotis", "f1_cb_test_Myotis",
        #                          "f1_cb_train_Psuper", "f1_cb_val_Psuper", "f1_cb_test_Psuper"]]
        # df_eval_f1 = df_eval_f1[["epoch", "f1_cb_train_Nyctaloid", "f1_cb_val_Nyctaloid", "f1_cb_test_Nyctaloid",
        #                          "f1_cb_train_Myotis", "f1_cb_val_Myotis", "f1_cb_test_Myotis",
        #                          "f1_cb_train_Plecotus", "f1_cb_val_Plecotus", "f1_cb_test_Plecotus",
        #                          "f1_cb_train_Psuper", "f1_cb_val_Psuper", "f1_cb_test_Psuper"]]

        df_averaging_collector_f1.append(df_eval_f1)

        df_eval_conf_mat = pd.read_csv(eval_csv)
        df_eval_conf_mat = df_eval_conf_mat[["epoch", "conf_mat_train", "conf_mat_val", "conf_mat_test"]]
        df_averaging_collector_conf_mat.append(df_eval_conf_mat)

    # Variants on bat noise classification
    df_mean_f1 = (pd.concat(df_averaging_collector_f1)
                  .groupby(["epoch"])
                  .agg(fscore_train_Bat=("f1_cb_train_Bat", "mean"),
                       fscore_train_Noise=("f1_cb_train_Noise", "mean"),
                       fscore_test_Bat=("f1_cb_test_Bat", "mean"),
                       fscore_test_Noise=("f1_cb_test_Noise", "mean"),
                       fscore_val_Bat=("f1_cb_val_Bat", "mean"),
                       fscore_val_Noise=("f1_cb_val_Noise", "mean")
                       ))

    # Variants on species classification
    # df_mean_f1 = (pd.concat(df_averaging_collector_f1)
    #               .groupby(["epoch"])
    #               .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "mean"),
    #                    fscore_train_Pnat=("f1_cb_train_Pnat", "mean"),
    #                    fscore_test_Ppip=("f1_cb_test_Ppip", "mean"),
    #                    fscore_test_Pnat=("f1_cb_test_Pnat", "mean"),
    #                    fscore_val_Ppip=("f1_cb_val_Ppip", "mean"),
    #                    fscore_val_Pnat=("f1_cb_val_Pnat", "mean")
    #                    ))
    # df_mean_f1 = (pd.concat(df_averaging_collector_f1)
    #               .groupby(["epoch"])
    #               .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "mean"),
    #                    fscore_train_Pnat=("f1_cb_train_Pnat", "mean"),
    #                    fscore_train_Ppyg=("f1_cb_train_Ppyg", "mean"),
    #                    fscore_test_Ppip=("f1_cb_test_Ppip", "mean"),
    #                    fscore_test_Pnat=("f1_cb_test_Pnat", "mean"),
    #                    fscore_test_Ppyg=("f1_cb_test_Ppyg", "mean"),
    #                    fscore_val_Ppip=("f1_cb_val_Ppip", "mean"),
    #                    fscore_val_Pnat=("f1_cb_val_Pnat", "mean"),
    #                    fscore_val_Ppyg=("f1_cb_val_Ppyg", "mean")
    #                    ))
    # df_mean_f1 = (pd.concat(df_averaging_collector_f1)
    #               .groupby(["epoch"])
    #               .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "mean"),
    #                    fscore_train_Pnat=("f1_cb_train_Pnat", "mean"),
    #                    fscore_train_Ppyg=("f1_cb_train_Ppyg", "mean"),
    #                    fscore_train_Ptief=("f1_cb_train_Ptief", "mean"),
    #                    fscore_test_Ppip=("f1_cb_test_Ppip", "mean"),
    #                    fscore_test_Pnat=("f1_cb_test_Pnat", "mean"),
    #                    fscore_test_Ppyg=("f1_cb_test_Ppyg", "mean"),
    #                    fscore_test_Ptief=("f1_cb_test_Ptief", "mean"),
    #                    fscore_val_Ppip=("f1_cb_val_Ppip", "mean"),
    #                    fscore_val_Pnat=("f1_cb_val_Pnat", "mean"),
    #                    fscore_val_Ppyg=("f1_cb_val_Ppyg", "mean"),
    #                    fscore_val_Ptief=("f1_cb_val_Ptief", "mean")
    #                    ))
    # df_mean_f1 = (pd.concat(df_averaging_collector_f1)
    #               .groupby(["epoch"])
    #               .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "mean"),
    #                    fscore_train_Pnat=("f1_cb_train_Pnat", "mean"),
    #                    fscore_train_Ppyg=("f1_cb_train_Ppyg", "mean"),
    #                    fscore_train_Ptief=("f1_cb_train_Ptief", "mean"),
    #                    fscore_train_Phoch=("f1_cb_train_Phoch", "mean"),
    #                    fscore_test_Ppip=("f1_cb_test_Ppip", "mean"),
    #                    fscore_test_Pnat=("f1_cb_test_Pnat", "mean"),
    #                    fscore_test_Ppyg=("f1_cb_test_Ppyg", "mean"),
    #                    fscore_test_Ptief=("f1_cb_test_Ptief", "mean"),
    #                    fscore_test_Phoch=("f1_cb_test_Phoch", "mean"),
    #                    fscore_val_Ppip=("f1_cb_val_Ppip", "mean"),
    #                    fscore_val_Pnat=("f1_cb_val_Pnat", "mean"),
    #                    fscore_val_Ppyg=("f1_cb_val_Ppyg", "mean"),
    #                    fscore_val_Ptief=("f1_cb_val_Ptief", "mean"),
    #                    fscore_val_Phoch=("f1_cb_val_Phoch", "mean")
    #                    ))

    # Variants on genus classification
    # df_mean_f1 = (pd.concat(df_averaging_collector_f1)
    #               .groupby(["epoch"])
    #               .agg(fscore_train_Nyctaloid=("f1_cb_train_Nyctaloid", "mean"),
    #                    fscore_train_Myotis=("f1_cb_train_Myotis", "mean"),
    #                    fscore_train_Plecotus=("f1_cb_train_Plecotus", "mean"),
    #                    fscore_train_Psuper=("f1_cb_train_Psuper", "mean"),
    #                    fscore_test_Nyctaloid=("f1_cb_test_Nyctaloid", "mean"),
    #                    fscore_test_Myotis=("f1_cb_test_Myotis", "mean"),
    #                    fscore_test_Plecotus=("f1_cb_test_Plecotus", "mean"),
    #                    fscore_test_Psuper=("f1_cb_test_Psuper", "mean"),
    #                    fscore_val_Nyctaloid=("f1_cb_val_Nyctaloid", "mean"),
    #                    fscore_val_Myotis=("f1_cb_val_Myotis", "mean"),
    #                    fscore_val_Plecotus=("f1_cb_val_Plecotus", "mean"),
    #                    fscore_val_Psuper=("f1_cb_val_Psuper", "mean")
    #                    ))

    df_mean_f1["epoch"] = df_averaging_collector_f1[0]["epoch"].tolist()
    df_mean_f1 = df_mean_f1.round(3)

    # Variants on bat noise classification
    df_std_f1 = (pd.concat(df_averaging_collector_f1)
                 .groupby(["epoch"])
                 .agg(fscore_train_Bat=("f1_cb_train_Bat", "std"),
                      fscore_train_Noise=("f1_cb_train_Noise", "std"),
                      fscore_test_Bat=("f1_cb_test_Bat", "std"),
                      fscore_test_Noise=("f1_cb_test_Noise", "std"),
                      fscore_val_Bat=("f1_cb_val_Bat", "std"),
                      fscore_val_Noise=("f1_cb_val_Noise", "std"),
                      ))

    # Variants on species classification
    # df_std_f1 = (pd.concat(df_averaging_collector_f1)
    #              .groupby(["epoch"])
    #              .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "std"),
    #                   fscore_train_Pnat=("f1_cb_train_Pnat", "std"),
    #                   fscore_test_Ppip=("f1_cb_test_Ppip", "std"),
    #                   fscore_test_Pnat=("f1_cb_test_Pnat", "std"),
    #                   fscore_val_Ppip=("f1_cb_val_Ppip", "std"),
    #                   fscore_val_Pnat=("f1_cb_val_Pnat", "std"),
    #                   ))
    # df_std_f1 = (pd.concat(df_averaging_collector_f1)
    #              .groupby(["epoch"])
    #              .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "std"),
    #                   fscore_train_Pnat=("f1_cb_train_Pnat", "std"),
    #                   fscore_train_Ppyg=("f1_cb_train_Ppyg", "std"),
    #                   fscore_test_Ppip=("f1_cb_test_Ppip", "std"),
    #                   fscore_test_Pnat=("f1_cb_test_Pnat", "std"),
    #                   fscore_test_Ppyg=("f1_cb_test_Ppyg", "std"),
    #                   fscore_val_Ppip=("f1_cb_val_Ppip", "std"),
    #                   fscore_val_Pnat=("f1_cb_val_Pnat", "std"),
    #                   fscore_val_Ppyg=("f1_cb_val_Ppyg", "std"),
    #                   ))
    # df_std_f1 = (pd.concat(df_averaging_collector_f1)
    #              .groupby(["epoch"])
    #              .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "std"),
    #                   fscore_train_Pnat=("f1_cb_train_Pnat", "std"),
    #                   fscore_train_Ppyg=("f1_cb_train_Ppyg", "std"),
    #                   fscore_train_Ptief=("f1_cb_train_Ptief", "std"),
    #                   fscore_test_Ppip=("f1_cb_test_Ppip", "std"),
    #                   fscore_test_Pnat=("f1_cb_test_Pnat", "std"),
    #                   fscore_test_Ppyg=("f1_cb_test_Ppyg", "std"),
    #                   fscore_test_Ptief=("f1_cb_test_Ptief", "std"),
    #                   fscore_val_Ppip=("f1_cb_val_Ppip", "std"),
    #                   fscore_val_Pnat=("f1_cb_val_Pnat", "std"),
    #                   fscore_val_Ppyg=("f1_cb_val_Ppyg", "std"),
    #                   fscore_val_Ptief=("f1_cb_val_Ptief", "std")
    #                   ))
    # df_std_f1 = (pd.concat(df_averaging_collector_f1)
    #              .groupby(["epoch"])
    #              .agg(fscore_train_Ppip=("f1_cb_train_Ppip", "std"),
    #                   fscore_train_Pnat=("f1_cb_train_Pnat", "std"),
    #                   fscore_train_Ppyg=("f1_cb_train_Ppyg", "std"),
    #                   fscore_train_Ptief=("f1_cb_train_Ptief", "std"),
    #                   fscore_train_Phoch=("f1_cb_train_Phoch", "std"),
    #                   fscore_test_Ppip=("f1_cb_test_Ppip", "std"),
    #                   fscore_test_Pnat=("f1_cb_test_Pnat", "std"),
    #                   fscore_test_Ppyg=("f1_cb_test_Ppyg", "std"),
    #                   fscore_test_Ptief=("f1_cb_test_Ptief", "std"),
    #                   fscore_test_Phoch=("f1_cb_test_Phoch", "std"),
    #                   fscore_val_Ppip=("f1_cb_val_Ppip", "std"),
    #                   fscore_val_Pnat=("f1_cb_val_Pnat", "std"),
    #                   fscore_val_Ppyg=("f1_cb_val_Ppyg", "std"),
    #                   fscore_val_Ptief=("f1_cb_val_Ptief", "std"),
    #                   fscore_val_Phoch=("f1_cb_val_Phoch", "std")
    #                   ))

    # Variants on genus classification
    # df_std_f1 = (pd.concat(df_averaging_collector_f1)
    #              .groupby(["epoch"])
    #              .agg(fscore_train_Nyctaloid=("f1_cb_train_Nyctaloid", "std"),
    #                   fscore_train_Myotis=("f1_cb_train_Myotis", "std"),
    #                   fscore_train_Plecotus=("f1_cb_train_Plecotus", "std"),
    #                   fscore_train_Psuper=("f1_cb_train_Psuper", "std"),
    #                   fscore_test_Nyctaloid=("f1_cb_test_Nyctaloid", "std"),
    #                   fscore_test_Myotis=("f1_cb_test_Myotis", "std"),
    #                   fscore_test_Plecotus=("f1_cb_test_Plecotus", "std"),
    #                   fscore_test_Psuper=("f1_cb_test_Psuper", "std"),
    #                   fscore_val_Nyctaloid=("f1_cb_val_Nyctaloid", "std"),
    #                   fscore_val_Myotis=("f1_cb_val_Myotis", "std"),
    #                   fscore_val_Plecotus=("f1_cb_val_Plecotus", "std"),
    #                   fscore_val_Psuper=("f1_cb_val_Psuper", "std")
    #                   ))

    df_std_f1["epoch"] = df_averaging_collector_f1[0]["epoch"].tolist()
    df_std_f1 = df_std_f1.round(3)

    if output_main_path is not None:
        output_main_path = output_main_path
    else:
        output_main_path = "AVG_performance_overviews/"
    try:
        os.makedirs(output_main_path)
    except OSError:
        print(f"File path {output_main_path} already exists!")

    output_csv_mean = output_main_path + f"mean_{eval_csvs[0].split('/')[-1][:-10]}.csv"
    output_csv_std = output_main_path + f"std_{eval_csvs[0].split('/')[-1][:-10]}.csv"

    df_mean_f1.to_csv(output_csv_mean, index=False)
    df_std_f1.to_csv(output_csv_std, index=False)

    return df_mean_f1, df_std_f1, output_csv_mean, output_csv_std


if __name__ == "__main__":
    """ (DEPRECATED) """

    accuracy_ref_val = accuracy_score(labels_ref, pred_ref)
    print(accuracy_ref_val)

    precision_ref_val = precision_score(labels_ref, pred_ref)
    print(precision_ref_val)

    recall_ref_val = recall_score(labels_ref, pred_ref)
    print(recall_ref_val)

    f1_ref_val = f1_score(labels_ref, pred_ref)
    print(f1_ref_val)
