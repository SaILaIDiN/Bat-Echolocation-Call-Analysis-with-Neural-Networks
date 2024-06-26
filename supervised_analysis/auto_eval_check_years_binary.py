""" OUTPUT-VECTOR-ANALYSIS:
    One can reveal more in-depth tendencies of a model trained at one location-time setting with an
     output-vector analysis for an application on test data from two different location-time settings.
    I.e., a model trained at Northeast 2019 is tested on a unseen test data of Northeast 2019 and 2020
     and then, the output-vectors of both test runs are statistically analysed.
    This can happen in two levels.
    First, one can make a statistic out of the output-vector tendencies to the binary case.
    This would reveal if a model is internally more prone to falsely classify a bat as noise and vice versa,
     even though the overall performance metrics seems unchanged. (since it is a threshold-based metric)
    Second, the statistic can reveal which classes of the bat species are more prone to fluctuate
     between different location-time settings.
    Since the model is trained to distinguish bats from noise, it does not give unbiased insights to the
     discriminative ability of the model to detect different species.
    It only reveals how different species are detected when trained against noise in conjunction.
    I.e., a model has to be separately trained for species distinction in order to reveal unbiased tendencies
     of the model on detecting individual classes in test data of different location-time settings.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from optimization_core.validate import eval_net
from optimization_core.metrics import prediction_refactoring, label_refactoring, standard_metrics


# # # Simple evaluation pipeline for a pretrained model
# Initialize a dataset
def initialize_pred_set(dataset, index):
    pred_set = dataset[index]
    pred_loader = DataLoader(pred_set, batch_size=1, shuffle=True, num_workers=0)
    return pred_loader


# Initialize pre-trained model
def initialize_model(model, n_classes, dir_model_weights):
    net = model(n_classes)
    net.load_state_dict(torch.load(dir_model_weights))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    return net, device


# Perform inference/prediction
def prediction_model(net, device, pred_loader, class_mode="binary", threshold=None):
    _, data_preds, data_labels, data_paths, data_sp_labels, data_sp_t_labels = eval_net(
        net, pred_loader, device, class_mode=class_mode, subset="delete", config_name="delete",
        optim_name="SGD", test_content="delete")

    data_preds_ref = prediction_refactoring(data_preds, class_mode=class_mode, threshold=threshold)
    data_labels_ref = label_refactoring(data_labels, data_sp_labels, class_mode=class_mode)

    return data_preds, data_labels, data_preds_ref, data_labels_ref, data_paths


# Compute performance metrics
def compute_performance_metrics(data_preds_ref, data_labels_ref, class_mode="binary"):
    data_accuracy, data_precision, data_recall, data_f1 = standard_metrics(data_labels_ref, data_preds_ref, class_mode)
    data_accuracy, data_precision, data_recall, data_f1 = round(data_accuracy, 3), round(data_precision, 3), \
                                                          round(data_recall, 3), round(data_f1, 3)
    return data_accuracy, data_precision, data_recall, data_f1


# Compute gradients to reveal inclinations/tendencies
def compute_gradients_from_prediction(data_preds, data_labels, data_paths, output_grads="data_grads.csv"):
    # turn list into numpy array
    data_preds, data_labels = np.asarray(data_preds), np.asarray(data_labels)

    # compute gradient of each sample, remove intermediate dimensions
    data_labels_minus_preds = np.subtract(data_labels, data_preds).reshape([-1, 2])
    print(np.shape(data_labels_minus_preds))
    df_grads = pd.DataFrame(data_labels_minus_preds, columns=["delta x_1", "delta x_2"])
    df_grads["image_path"] = data_paths

    df_grads.to_csv(output_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    return df_grads


def compute_hit_and_miss_gradients_class_independent(df_grads, output_hit_grads="data_grads_hit.csv",
                                                     output_miss_grads="data_grads_miss.csv"):
    # divide data_labels_minus_preds into two frames split by threshold 0.5
    df_grads_hit = pd.DataFrame(df_grads.loc[df_grads["delta x_1"] <= abs(0.5)])
    df_grads_hit.to_csv(output_hit_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    df_grads_miss = pd.DataFrame(df_grads.loc[df_grads["delta x_1"] >= abs(0.5)])
    df_grads_miss.to_csv(output_miss_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    return df_grads_hit, df_grads_miss


def compute_hit_and_miss_gradients_class_dependent(df_grads,
                                                   output_hit_bats_grads="data_grads_hit_bats.csv",
                                                   output_miss_bats_grads="data_grads_miss_bats.csv",
                                                   output_hit_noise_grads="data_grads_hit_noise.csv",
                                                   output_miss_noise_grads="data_grads_miss_noise.csv"):
    # divide data_labels_minus_preds into four frames split by threshold 0.5/-0.5
    df_grads_hit_noise = pd.DataFrame(df_grads.loc[(0 <= df_grads["delta x_1"]) & (df_grads["delta x_1"] <= 0.5)])
    df_grads_hit_noise.to_csv(output_hit_noise_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    df_grads_hit_bats = pd.DataFrame(df_grads.loc[(-0.5 <= df_grads["delta x_1"]) & (df_grads["delta x_1"] <= 0)])
    df_grads_hit_bats.to_csv(output_hit_bats_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    df_grads_miss_noise = pd.DataFrame(df_grads.loc[(1 >= df_grads["delta x_1"]) & (df_grads["delta x_1"] > 0.5)])
    df_grads_miss_noise.to_csv(output_miss_noise_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    df_grads_miss_bats = pd.DataFrame(df_grads.loc[(-1 <= df_grads["delta x_1"]) & (df_grads["delta x_1"] < -0.5)])
    df_grads_miss_bats.to_csv(output_miss_bats_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    return df_grads_hit_bats, df_grads_miss_bats, df_grads_hit_noise, df_grads_miss_noise


def compute_specific_output_gradients_class_dependent(df_grads, threshold_low, threshold_high,
                                                      output_grads="data_grads_specific.csv"):
    df_grads_specific = pd.DataFrame(
        df_grads.loc[(threshold_low <= df_grads["delta x_1"]) & (df_grads["delta x_1"] <= threshold_high)])
    df_grads_specific.to_csv(output_grads, header=["delta x_1", "delta x_2", "image_path"], index=False)
    return df_grads_specific


def compute_mean_of_gradients(df_grads_list, grads_names_list):
    mean_list = []
    for df_grads, grads_name in zip(df_grads_list, grads_names_list):
        # take the magnitude of at least one of the columns to have the positive gradient
        df_grads = df_grads.abs()
        # compute the average of both frames (mean magnitude of gradients for hit and miss)
        mean_grad = df_grads.mean(axis=0)
        mean_list.append(mean_grad)
    return mean_list


def compute_histogram(df_grads_hit, df_grads_miss, title_1="title 1", title_2="title 2", output_1=None, output_2=None):
    # compute a histogram of both frames to show the distribution of gradients

    ax1 = df_grads_hit.plot.hist(by="delta x_1", bins=20, rwidth=0.7)
    ax2 = df_grads_miss.plot.hist(by="delta x_1", bins=40, rwidth=0.7)
    ax1.set_title(title_1)
    ax2.set_title(title_2)
    if output_1 is not None:
        os.makedirs(output_1, exist_ok=True)
        ax1.figure.savefig(f"{os.path.join(output_1, title_1)}.pdf")
    else:
        print("Nothing plotted. (Hist hit)")
    if output_2 is not None:
        ax2.figure.savefig(f"{os.path.join(output_2, title_2)}.pdf")
    else:
        print("Nothing plotted. (Hist miss)")

    return ax1, ax2


def compute_histogram_avg(hist_hit_list, hist_miss_list, bins_interval_1, bins_interval_2,
                          title_1="title 1", title_2="title 2", output_1=None, output_2=None):
    if output_1 is not None:
        bins = [round(i, 2) for i in np.arange(bins_interval_1[0], bins_interval_1[1], bins_interval_1[2])]
        width = 0.7 * (bins[1] - bins[0])
        plt.clf()
        plt.bar(bins, np.around(np.mean(hist_hit_list, axis=0), decimals=0).astype(int), align='center', width=width)
        plt.title(title_1)
        plt.savefig(f"{os.path.join(output_1, title_1)}.pdf")
    else:
        print("Nothing plotted. (Hist hit avg.)")
    if output_2 is not None:
        bins = [round(i, 2) for i in np.arange(bins_interval_2[0], bins_interval_2[1], bins_interval_2[2])]
        width = 0.7 * (bins[1] - bins[0])
        plt.clf()
        plt.bar(bins, np.around(np.mean(hist_miss_list, axis=0), decimals=0).astype(int), align='center', width=width)
        plt.title(title_2)
        plt.savefig(f"{os.path.join(output_2, title_2)}.pdf")
    else:
        print("Nothing plotted. (Hist miss avg.)")


if __name__ == "__main__":
    import os
    import fnmatch
    from optimization_core.models import YPNet, YPNetNew
    from supervised_analysis.check_years_binary import dataset_01, dataset_02, dataset_03, dataset_04, \
        dataset_05, dataset_06, dataset_07, dataset_08
    from data.path_provider import dir_results, dir_results_remote

    # parse through the last folder name provided by test_content in test_content_list
    # look for a file with ending ".pth" and a string from list_remark
    list_remark = [f"_run_{i+1}" for i in range(0, 5)]
    test_content_list = ["01_Test_1_NE19_train_NW19_test", "02_Test_1_NW19_train_NE19_test",
                         "03_Test_2_NE20_train_NW20_test", "04_Test_2_NW20_train_NE20_test",
                         "05_Test_3_NE19_train_NE20_test", "06_Test_3_NE20_train_NE19_test",
                         "07_Test_4_NW19_train_NW20_test", "08_Test_4_NW20_train_NW19_test"]
    dataset_list = [dataset_01, dataset_02, dataset_03, dataset_04,
                    dataset_05, dataset_06, dataset_07, dataset_08]
    partition_name_list = ["val_own", "val_foreign"]
    partition_index_list = [1, 2]
    class_mode = "binary"
    dir_model_weights_main = r"{arg}\pth_files\final_model".format(arg=dir_results)

    for dataset, test_content in zip(dataset_list, test_content_list):
        dir_model_weights_test_content = dir_model_weights_main + "\\" + test_content
        for partition_name, partition_index in zip(partition_name_list, partition_index_list):
            bats_grads_collector, noise_grads_collector = [], []
            for remark in list_remark:
                for pth_file in os.listdir(dir_model_weights_test_content):
                    if fnmatch.fnmatch(pth_file, f"*{remark}.pth"):
                        dir_model_weights = dir_model_weights_test_content + "\\" + pth_file
                        print("PTH-FILE: ", pth_file)
                        print("DIR-MODEL-WEIGHTS: ", dir_model_weights)

                        model = YPNetNew
                        n_classes = 2

                        pred_loader = initialize_pred_set(dataset, index=partition_index)
                        net, device = initialize_model(model, n_classes, dir_model_weights)
                        data_preds, data_labels, data_preds_ref, data_labels_ref, data_paths = prediction_model(
                            net, device, pred_loader, class_mode=class_mode)

                        # # # Stage 1
                        # compute performance metrics
                        data_accuracy, data_precision, data_recall, data_f1 = compute_performance_metrics(
                            data_preds_ref, data_labels_ref, class_mode=class_mode)

                        # # # Stage 2
                        # compute gradients/inclinations
                        df_grads = compute_gradients_from_prediction(data_preds, data_labels, data_paths)

                        df_grads_specific = compute_specific_output_gradients_class_dependent(
                            df_grads, threshold_low=0.9, threshold_high=1.0,
                            output_grads=f"data_grads_specific_{remark}_{partition_name}.csv")

                        df_grads_hit, df_grads_miss = compute_hit_and_miss_gradients_class_independent(df_grads)
                        df_grads_hit_bats, df_grads_miss_bats, df_grads_hit_noise, df_grads_miss_noise = \
                            compute_hit_and_miss_gradients_class_dependent(df_grads)
                        # compute_histogram(df_grads_hit, df_grads_miss, "hit", "missed")
                        print("SHAPE - df_grads: ", df_grads_hit_bats.shape)
                        print("SHAPE - df_grads: ", df_grads_hit_noise.shape)

                        compute_histogram(df_grads_hit_bats, df_grads_miss_bats,
                                          f"bats hit {remark}", f"bats missed {remark}",
                                          output_1=os.path.join(dir_model_weights_test_content, partition_name),
                                          output_2=os.path.join(dir_model_weights_test_content, partition_name))
                        compute_histogram(df_grads_hit_noise, df_grads_miss_noise,
                                          f"noise hit {remark}", f"noise missed {remark}",
                                          output_1=os.path.join(dir_model_weights_test_content, partition_name),
                                          output_2=os.path.join(dir_model_weights_test_content, partition_name))

                        # collect DataFrames of class dependent gradients
                        bats_grads_collector.append([df_grads_hit_bats, df_grads_miss_bats])
                        noise_grads_collector.append([df_grads_hit_noise, df_grads_miss_noise])

            # include averaging of histograms, because csv files are randomly ordered every new run, but hist is sorting
            # print("1", bats_grads_collector[0][0]["delta x_1"].iloc[0:].values)
            # print("2", np.histogram(bats_grads_collector[0][0]["delta x_1"].iloc[0:], bins=20))

            hist_by_run_for_bats_hit_list = [np.histogram(df_grads_bats[0]["delta x_1"].iloc[0:].values, bins=10)[0]
                                             for df_grads_bats in bats_grads_collector]
            hist_by_run_for_bats_miss_list = [np.histogram(df_grads_bats[1]["delta x_1"].iloc[0:].values, bins=10)[0]
                                              for df_grads_bats in bats_grads_collector]
            hist_by_run_for_noise_hit_list = [np.histogram(df_grads_noise[0]["delta x_1"].iloc[0:].values, bins=10)[0]
                                              for df_grads_noise in noise_grads_collector]
            hist_by_run_for_noise_miss_list = [np.histogram(df_grads_noise[1]["delta x_1"].iloc[0:].values, bins=10)[0]
                                               for df_grads_noise in noise_grads_collector]
            # df_average_hist_bats_hit = pd.DataFrame(np.mean(hist_by_run_for_bats_hit_list, axis=0))
            # df_average_hist_bats_miss = pd.DataFrame(np.mean(hist_by_run_for_bats_miss_list, axis=0))
            # df_average_hist_noise_hit = pd.DataFrame(np.mean(hist_by_run_for_noise_hit_list, axis=0))
            # df_average_hist_noise_miss = pd.DataFrame(np.mean(hist_by_run_for_noise_miss_list, axis=0))
            # print("SHAPE - numpy_hist: ", np.mean(hist_by_run_for_bats_hit_list, axis=0).shape)
            # print("FORM - numpy_hist: ", np.mean(hist_by_run_for_bats_hit_list, axis=0))
            # print("SHAPE - df_hist: ", df_average_hist_bats_hit.shape)
            # print("SHAPE - df_hist: ", df_average_hist_noise_hit.shape)

            compute_histogram_avg(hist_by_run_for_bats_hit_list, hist_by_run_for_bats_miss_list,
                                  bins_interval_1=[-0.5, 0, 0.05], bins_interval_2=[-1, -0.5, 0.05],
                                  title_1="bats hit average", title_2="bats miss average",
                                  output_1=os.path.join(dir_model_weights_test_content, partition_name),
                                  output_2=os.path.join(dir_model_weights_test_content, partition_name))

            compute_histogram_avg(hist_by_run_for_noise_hit_list, hist_by_run_for_noise_miss_list,
                                  bins_interval_1=[0, 0.5, 0.05], bins_interval_2=[0.5, 1, 0.05],
                                  title_1="noise hit average", title_2="noise miss average",
                                  output_1=os.path.join(dir_model_weights_test_content, partition_name),
                                  output_2=os.path.join(dir_model_weights_test_content, partition_name))
