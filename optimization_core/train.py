import os
import time
from tqdm import tqdm
import logging
import argparse

import torch
import torch.optim as optim
import torch.nn as nn

from optimization_core.validate import eval_net
from optimization_core.metrics import prediction_refactoring, label_refactoring, standard_metrics, \
    multi_class_label_translator, hard_case_image_path_tracing, species_to_genus_translator, compute_confusion_matrix, \
    performance_overview_collector
from optimization_core.runbuilder import RunBuilder, RunManager, OrderedDict
from optimization_core.logger_maker import LoggerMaker

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description="Parameter Management")
parser.add_argument("-af", "--arg_file", default=None, type=str,
                    help="txt-file of various arguments")
parser.add_argument("-mp", "--main_paths", default=None, type=str,
                    help="system-related main paths")
# # # Grid-Search (some parameters have no effect based on optimizer)
# parser.add_argument("-mt", "--model_type", default=None, type=str)
parser.add_argument("-opt", "--optimizer", default=None, type=str)
parser.add_argument("-lr", "--learning_rate", default=None, type=list)
parser.add_argument("-mom", "--momentum", default=None, type=list)
parser.add_argument("-wd", "--weight_decay", default=None, type=list)
parser.add_argument("-do", "--dropout", default=None, type=list)
parser.add_argument("-bs", "--batch_size", default=None, type=list)
parser.add_argument("-nho", "--n_hidden_out", default=None, type=list)
parser.add_argument("-ep", "--epochs", default=[10], type=list)
# # # Custom Combinations
parser.add_argument("-cc", "--custom_combo", default="False", type=str,
                    help="Switch to unique customized parameter combinations.")
parser.add_argument("-cp", "--custom_parameters", default=None, type=list,
                    help="List of lists where the first list contains the parameter names. "
                         "Follow up lists have the corresponding parameter values")
args = parser.parse_args()


def get_training_dict(arg_file=None, main_paths=None, optimizer=None, learning_rate=None, momentum=None,
                      weight_decay=None, dropout=None, batch_size=None, n_hidden_out=None, epochs=None,
                      custom_combo=None, custom_parameters=None):
    dict_tmp = {'arg_file': arg_file,
                'main_paths': main_paths,
                'optimizer': optimizer,
                'learning_rate': learning_rate,
                'momentum': momentum,
                'weight_decay': weight_decay,
                'dropout': dropout,
                'batch_size': batch_size,
                'n_hidden_out': n_hidden_out,
                'epochs': epochs,
                'custom_combo': custom_combo,
                'custom_parameters': custom_parameters
                }
    return dict_tmp


# Define automated training procedure (first for single net, single dataset and single task)
def train_net_auto(args, net, dataset, device, n_classes=2, class_mode="binary", species_or_genus="species",
                   bat_class_list=None, save_cp=True, meta_logger=None, remark="",
                   val_2nd_set=True, val_2nd_set_name="test", system_mode="local_mode",
                   dir_checkpoint=None, dir_final_model=None, path_prefix="results", test_content="",
                   score_threshold=None, score_by_majority=False):
    """ species_or_genus: translate and summarize a set of related species into a mutual genus to change labels """

    if args.custom_combo == "True":
        # Define hyperparameters for testing in case of custom combinations of parameters
        parameters = args.custom_parameters
        # parameters = [['lr', 'weight_decay', 'momentum', 'batch_size', 'epochs'],
        #               [0.000001, 0.0009, 0.90, 1, 10]
        #               ]
    else:
        # Define hyperparameters for testing in case of cartesian product
        parameters = OrderedDict(
            lr=args.learning_rate,
            weight_decay = args.weight_decay,
            momentum = args.momentum,
            batch_size = args.batch_size,
            epochs = args.epochs,
            dropout = args.dropout,
            n_hidden_out = args.n_hidden_out
        )

    runs = RunBuilder.get_runs(parameters, custom_combo=args.custom_combo)
    run_manager = RunManager()
    run_count = 0
    for run in runs:
        # Create new network instance to not pass the trained weights to the next run!
        model = net(n_classes=n_classes, dropout=run.dropout, n_hidden_out=run.n_hidden_out)
        model.to(device)
        with tqdm(total=run.epochs, desc=f'Run {run_count + 1}/{len(runs)}', unit='img') as rbar:
            run_count += 1

            if val_2nd_set:
                train_set, val_set, eval_2nd_set = dataset[0], dataset[1], dataset[2]
            else:
                train_set, val_set = dataset[0], dataset[1]
            train_loader = DataLoader(train_set, batch_size=run.batch_size, shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_set, batch_size=run.batch_size, shuffle=True, num_workers=0, drop_last=True)
            if val_2nd_set:
                eval_2nd_set_loader = DataLoader(eval_2nd_set, batch_size=run.batch_size, shuffle=True, num_workers=0,
                                                 drop_last=True)
                n_eval_2nd_set = len(eval_2nd_set)
            n_train, n_val = len(train_set), len(val_set)

            run_manager.begin_run(run, model)

            # Time string for time stamp in stored filenames
            timestr = time.strftime("%Y%m%d-%H%M%S")

            # Define configuration name for logistics
            config_name = run_manager.get_config_name(name_optimizer=args.optimizer)

            # Define SummaryWriter for Tensorboard
            os.makedirs(f"{path_prefix}/runs/{test_content}", exist_ok=True)
            writer = SummaryWriter(log_dir=f"{path_prefix}/runs/{test_content}/"
                                           f"{timestr}_{system_mode}_{config_name}_{remark}")

            # Define logger per configuration run
            os.makedirs(f"{path_prefix}/logs/train_metrics/{args.optimizer}/{test_content}", exist_ok=True)
            train_metrics_logger_maker = LoggerMaker(f"{path_prefix}/logs/train_metrics/{args.optimizer}/"
                                                     f"{test_content}/{config_name}{remark}",
                                                     level=logging.INFO, filelevel=logging.INFO, filehandling=True)
            os.makedirs(f"{path_prefix}/logs/val_metrics/{args.optimizer}/{test_content}", exist_ok=True)
            val_metrics_logger_maker = LoggerMaker(f"{path_prefix}/logs/val_metrics/{args.optimizer}/"
                                                   f"{test_content}/{config_name}{remark}",
                                                   level=logging.INFO, filelevel=logging.INFO, filehandling=True)
            os.makedirs(f"{path_prefix}/logs/{val_2nd_set_name}_metrics/{args.optimizer}/{test_content}",
                        exist_ok=True)
            eval_2nd_set_metrics_logger_maker = LoggerMaker(f"{path_prefix}/logs/{val_2nd_set_name}_metrics/"
                                                            f"{args.optimizer}/{test_content}/{config_name}{remark}",
                                                            level=logging.INFO, filelevel=logging.INFO,
                                                            filehandling=True)

            os.makedirs(f"{path_prefix}/logs/train_hard_cases/{args.optimizer}/{test_content}", exist_ok=True)
            train_hard_cases_logger_maker = LoggerMaker(f"{path_prefix}/logs/train_hard_cases/{args.optimizer}/"
                                                        f"{test_content}/{config_name}{remark}",
                                                        level=logging.INFO, filelevel=logging.INFO, filehandling=True)
            os.makedirs(f"{path_prefix}/logs/val_hard_cases/{args.optimizer}/{test_content}", exist_ok=True)
            val_hard_cases_logger_maker = LoggerMaker(f"{path_prefix}/logs/val_hard_cases/{args.optimizer}/"
                                                      f"{test_content}/{config_name}{remark}",
                                                      level=logging.INFO, filelevel=logging.INFO, filehandling=True)
            os.makedirs(f"{path_prefix}/logs/{val_2nd_set_name}_hard_cases/{args.optimizer}/{test_content}",
                        exist_ok=True)
            eval_2nd_set_hard_cases_logger_maker = LoggerMaker(f"{path_prefix}/logs/{val_2nd_set_name}_hard_cases/"
                                                               f"{args.optimizer}/{test_content}/{config_name}{remark}",
                                                               level=logging.INFO, filelevel=logging.INFO,
                                                               filehandling=True)

            os.makedirs(f"{path_prefix}/logs/loss/{args.optimizer}/{test_content}", exist_ok=True)
            loss_logger_maker = LoggerMaker(f"{path_prefix}/logs/loss/{args.optimizer}/"
                                            f"{test_content}/{config_name}{remark}",
                                            level=logging.INFO, filelevel=logging.INFO, filehandling=True)

            train_metrics_logger = train_metrics_logger_maker.logger
            val_metrics_logger = val_metrics_logger_maker.logger
            eval_2nd_set_metrics_logger = eval_2nd_set_metrics_logger_maker.logger
            loss_logger = loss_logger_maker.logger

            train_hard_cases_logger = train_hard_cases_logger_maker.logger
            val_hard_cases_logger = val_hard_cases_logger_maker.logger
            eval_2nd_set_hard_cases_logger = eval_2nd_set_hard_cases_logger_maker.logger

            path_confusion_matrix_train = f"{path_prefix}/logs/train_confusion_matrices/{args.optimizer}/{test_content}" \
                                          f"/{remark}"
            path_confusion_matrix_val = f"{path_prefix}/logs/val_confusion_matrices/{args.optimizer}/{test_content}" \
                                        f"/{remark}"
            path_confusion_matrix_eval_2nd = f"{path_prefix}/logs/{val_2nd_set_name}_confusion_matrices/" \
                                             f"{args.optimizer}/{test_content}/{remark}"
            os.makedirs(path_confusion_matrix_train, exist_ok=True)
            os.makedirs(path_confusion_matrix_val, exist_ok=True)
            os.makedirs(path_confusion_matrix_eval_2nd, exist_ok=True)

            # Define loss function
            criterion = nn.BCEWithLogitsLoss()
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1, 0.9], device=device))
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([90.0], device=device))
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 3, 50, 1], device=device))  # genera-4
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 10, 500], device=device))  # species-3
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 10, 500, 15], device=device))  # species-4
            # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1, 10, 500, 15, 2000], device=device))  # species-5

            # Define optimizer
            if args.optimizer == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=run.lr, momentum=run.momentum,
                                      weight_decay=run.weight_decay)
            elif args.optimizer == "ADAM":
                optimizer = optim.Adam(model.parameters(), lr=run.lr)

            # Training loop over epochs
            i_tot = 0  # global steps
            for epoch in range(run.epochs):
                model.train()
                run_manager.begin_epoch()
                epoch_loss = 0
                epoch_val_loss = 0
                val_counter = 0
                with tqdm(total=n_train, desc=f'Run: {run_count} - Epoch: {epoch + 1}/{run.epochs}', unit='img') as pbar:

                    for i, batch in enumerate(train_loader, 0):
                        images = batch["image"].unsqueeze(1).to(device, dtype=torch.float32)  # Shape [N,H,W]->[N,C,H,W]
                        bin_labels = batch["label"]
                        species_labels = batch["text_label"]  # 'list' of 'str'
                        if class_mode == "binary":
                            labels = bin_labels.to(device, dtype=torch.float32)
                            bat_class_list = ['Bat', 'Noise']
                        else:
                            if species_or_genus == "genus":
                                # Currently hardcoded for our case of Pipistrellus species
                                # bat_class_list = ['Nyctaloid', 'Myotis', 'Psuper']
                                bat_class_list = ['Nyctaloid', 'Myotis', 'Plecotus', 'Psuper']
                                species_to_combine = ['Ppip', 'Ptief', 'Phoch', 'Pnat', 'Ppyg', 'Pipistrelloid']
                                genus_name = 'Psuper'
                                species_labels = species_to_genus_translator(
                                    species_labels, species_to_combine, genus_name)

                            # # translate species string labels to one-hot-encoded vectors
                            trans_dict = multi_class_label_translator(bat_class_list)  # 'dict'
                            species_labels = [trans_dict[species] for species in species_labels]  # 'list' of 'ndarray'
                            labels = torch.as_tensor(species_labels)  # 'torch.Tensor' of 'torch.Tensor'
                            labels = labels.to(device, dtype=torch.float32)

                        if images.shape[3] == 586:
                            # reset gradients
                            optimizer.zero_grad()
                            # forward pass
                            outputs = model(images)
                            # loss computation
                            loss = criterion(outputs, labels)
                            # backward pass
                            loss.backward()
                            # weight update via gradient descent
                            optimizer.step()

                            epoch_loss += loss
                            pbar.set_postfix(**{'loss (batch)': loss})  # add loss of this batch to pbar information
                            pbar.update(run.batch_size)

                            i_tot += run.batch_size

                            loss_logger.info(f"TRAINING phase at step number {i_tot}. Epoch {epoch + 1}.")
                            loss_logger.info(f"-> Training Loss: {loss}.")
                            writer.add_scalar("Loss/train", loss, i_tot)

                            if (n_train - run.batch_size) <= (i+1) * run.batch_size < n_train:

                                train_loss, tr_preds, tr_labels, tr_image_paths, tr_sp_labels, tr_sp_t_labels = eval_net(
                                    model, train_loader, device, class_mode=class_mode,
                                    species_or_genus=species_or_genus, bat_class_list=bat_class_list,
                                    step_tot=i_tot, subset="train", optim_name=args.optimizer, config_name=config_name,
                                    path_prefix=path_prefix, test_content=test_content, remark=remark)

                                val_loss, v_preds, v_labels, v_image_paths, v_sp_labels, v_sp_t_labels = eval_net(
                                    model, val_loader, device, class_mode=class_mode,
                                    species_or_genus=species_or_genus, bat_class_list=bat_class_list,
                                    step_tot=i_tot, subset="val", optim_name=args.optimizer, config_name=config_name,
                                    path_prefix=path_prefix, test_content=test_content, remark=remark)

                                if val_2nd_set:
                                    eval_2nd_set_loss, e2s_preds, e2s_labels, e2s_image_paths, e2s_sp_labels, e2s_sp_t_labels = eval_net(
                                        model, eval_2nd_set_loader, device, class_mode=class_mode,
                                        species_or_genus=species_or_genus, bat_class_list=bat_class_list,
                                        step_tot=i_tot, subset="test", optim_name=args.optimizer,
                                        config_name=config_name, path_prefix=path_prefix, test_content=test_content,
                                        remark=remark)

                                loss_logger.info(f"EVALUATION phase at step number {i_tot}. Epoch {epoch + 1}.")
                                loss_logger.info(f"-> Validation loss: {val_loss}")
                                if val_2nd_set:
                                    loss_logger.info(f"-> {val_2nd_set_name} loss: {eval_2nd_set_loss}")
                                epoch_val_loss += val_loss
                                val_counter += 1

                                # NOTE: logging of metrics for multi-class gives you the joined-one-vs-rest scores
                                # see metrics.py for that i.e. label_refactoring() and prediction_refactoring()
                                tr_hard_case_collector, tr_counter = hard_case_image_path_tracing(
                                    tr_preds, tr_labels, tr_sp_labels, tr_sp_t_labels, tr_image_paths,
                                    class_mode=class_mode, n_classes=n_classes, score_threshold=score_threshold,
                                    score_by_majority=score_by_majority)
                                tr_preds = prediction_refactoring(tr_preds, class_mode=class_mode, n_classes=n_classes,
                                                                  threshold=score_threshold, majority=score_by_majority)
                                tr_labels = label_refactoring(tr_labels, tr_sp_labels, class_mode=class_mode,
                                                              n_classes=n_classes)
                                tr_accuracy, tr_precision, tr_recall, tr_f1, tr_f1_cb = standard_metrics(
                                    tr_labels, tr_preds, class_mode=class_mode)
                                train_metrics_logger.info(
                                    f"Accuracy: {round(tr_accuracy, 3)}, Precision: {round(tr_precision, 3)}, "
                                    f"Recall: {round(tr_recall, 3)}, F1: {round(tr_f1, 3)}"
                                    f"\nF1 (class-based): {[round(f1_cb, 3) for f1_cb in tr_f1_cb]}, "
                                    f"{bat_class_list}")
                                train_collector_list = [f"Pred:{entry[0]} Label:{entry[1]} Text-Label:{entry[2]} "
                                                        f"Filepath:{entry[3]}"
                                                        for entry in tr_hard_case_collector]
                                train_hard_cases_logger.info(f"\n EPOCH: {epoch}, "
                                                             f"COUNT(hard_cases): {tr_counter}, "
                                                             f"COUNT(all_cases): {n_train}")
                                train_hard_cases_logger.info("\n".join(train_collector_list))
                                conf_mat_train = compute_confusion_matrix(tr_labels, tr_preds, bat_class_list, n_classes,
                                                                          store="True", mode="Train", epoch=epoch,
                                                                          output_path=path_confusion_matrix_train)

                                v_hard_case_collector, v_counter = hard_case_image_path_tracing(
                                    v_preds, v_labels, v_sp_labels, v_sp_t_labels, v_image_paths,
                                    class_mode=class_mode, n_classes=n_classes, score_threshold=score_threshold,
                                    score_by_majority=score_by_majority)
                                v_preds = prediction_refactoring(v_preds, class_mode=class_mode, n_classes=n_classes,
                                                                 threshold=score_threshold, majority=score_by_majority)
                                v_labels = label_refactoring(v_labels, v_sp_labels, class_mode=class_mode,
                                                             n_classes=n_classes)
                                v_accuracy, v_precision, v_recall, v_f1, v_f1_cb = standard_metrics(
                                    v_labels, v_preds, class_mode=class_mode)
                                val_metrics_logger.info(
                                    f"Accuracy: {round(v_accuracy, 3)}, Precision: {round(v_precision, 3)}, "
                                    f"Recall: {round(v_recall, 3)}, F1: {round(v_f1, 3)}"
                                    f"\nF1 (class-based): {[round(f1_cb, 3) for f1_cb in v_f1_cb]}, "
                                    f"{bat_class_list}")
                                val_collector_list = [f"Pred:{entry[0]} Label:{entry[1]} Text-Label:{entry[2]} "
                                                      f"Filepath:{entry[3]}"
                                                      for entry in tr_hard_case_collector]
                                val_hard_cases_logger.info(f"\n EPOCH: {epoch}, "
                                                           f"COUNT(hard_cases): {v_counter}, "
                                                           f"COUNT(all_cases): {n_val}")
                                val_hard_cases_logger.info("\n".join(val_collector_list))
                                conf_mat_val = compute_confusion_matrix(v_labels, v_preds, bat_class_list, n_classes,
                                                                        store="True", mode="Val", epoch=epoch,
                                                                        output_path=path_confusion_matrix_val)

                                if val_2nd_set:  # for either eval_2nd_set or real test sets
                                    e2s_hard_case_collector, e2s_counter = hard_case_image_path_tracing(
                                        e2s_preds, e2s_labels, e2s_sp_labels, e2s_sp_t_labels, e2s_image_paths,
                                        class_mode=class_mode, n_classes=n_classes, score_threshold=score_threshold,
                                        score_by_majority=score_by_majority)
                                    e2s_preds = prediction_refactoring(e2s_preds, class_mode=class_mode,
                                                                       n_classes=n_classes, threshold=score_threshold,
                                                                       majority=score_by_majority)
                                    e2s_labels = label_refactoring(e2s_labels, e2s_sp_labels, class_mode=class_mode,
                                                                   n_classes=n_classes)
                                    e2s_accuracy, e2s_precision, e2s_recall, e2s_f1, e2s_f1_cb = standard_metrics(
                                        e2s_labels, e2s_preds, class_mode=class_mode)
                                    eval_2nd_set_metrics_logger.info(
                                        f"Accuracy: {round(e2s_accuracy, 3)}, Precision: {round(e2s_precision, 3)}, "
                                        f"Recall: {round(e2s_recall, 3)}, F1: {round(e2s_f1, 3)}"
                                        f"\nF1 (class-based): {[round(f1_cb, 3) for f1_cb in e2s_f1_cb]}, "
                                        f"{bat_class_list}")
                                    eval_2nd_collector_list = [f"Pred:{entry[0]} Label:{entry[1]} Text-Label:{entry[2]}"
                                                               f" Filepath:{entry[3]}"
                                                               for entry in tr_hard_case_collector]
                                    eval_2nd_set_hard_cases_logger.info(f"\n EPOCH: {epoch}, "
                                                                        f"COUNT(hard_cases): {e2s_counter}, "
                                                                        f"COUNT(all_cases): {n_eval_2nd_set}")
                                    eval_2nd_set_hard_cases_logger.info("\n".join(eval_2nd_collector_list))
                                    conf_mat_eval_2nd_set = compute_confusion_matrix(
                                        e2s_labels, e2s_preds, bat_class_list, n_classes, store="True", mode="Test",
                                        epoch=epoch, output_path=path_confusion_matrix_eval_2nd)

                                if val_2nd_set:
                                    dir_df_performance_overview = f"{path_prefix}/logs/performance_overview/" \
                                                                  f"{args.optimizer}/{test_content}/" \
                                                                  f"{config_name}{remark}.csv"
                                    os.makedirs(os.path.dirname(dir_df_performance_overview), exist_ok=True)
                                    performance_overview_collector(
                                        dir_df_performance_overview, epoch, tr_f1_cb, v_f1_cb, e2s_f1_cb,
                                        conf_mat_train, conf_mat_val, conf_mat_eval_2nd_set, bat_class_list)

                                writer.add_scalar("Loss/train", train_loss, i_tot)
                                writer.add_scalar("Metrics/train/accuracy", tr_accuracy, i_tot)
                                writer.add_scalar("Metrics/train/precision", tr_precision, i_tot)
                                writer.add_scalar("Metrics/train/recall", tr_recall, i_tot)
                                writer.add_scalar("Metrics/train/f1", tr_f1, i_tot)

                                writer.add_scalar("Loss/validate", val_loss, i_tot)
                                writer.add_scalar("Metrics/validate/accuracy", v_accuracy, i_tot)
                                writer.add_scalar("Metrics/validate/precision", v_precision, i_tot)
                                writer.add_scalar("Metrics/validate/recall", v_recall, i_tot)
                                writer.add_scalar("Metrics/validate/f1", v_f1, i_tot)

                                if val_2nd_set:
                                    writer.add_scalar(f"Loss/{val_2nd_set_name}", eval_2nd_set_loss, i_tot)
                                    writer.add_scalar(f"Metrics/{val_2nd_set_name}/accuracy", e2s_accuracy, i_tot)
                                    writer.add_scalar(f"Metrics/{val_2nd_set_name}/precision", e2s_precision, i_tot)
                                    writer.add_scalar(f"Metrics/{val_2nd_set_name}/recall", e2s_recall, i_tot)
                                    writer.add_scalar(f"Metrics/{val_2nd_set_name}/f1", e2s_f1, i_tot)
                                if val_2nd_set:
                                    run_manager.track_inter_loss(eval_2nd_set_loss, val_loss)
                                    run_manager.track_inter_performance_metrics(e2s_f1, e2s_precision, e2s_recall)
                                run_manager.track_epoch_loss(epoch_loss/i, epoch_val_loss/val_counter)
                                run_manager.track_epoch_performance_metrics(tr_f1, tr_precision, tr_recall,
                                                                            v_f1, v_precision, v_recall)
                t_epoch, t_interm_run = run_manager.end_epoch()
                rbar.set_postfix(**{'time duration of epoch': t_epoch, 'time duration of run': t_interm_run})
                rbar.update(1)
                if save_cp:  # only saves .pth files when meta_logger is given
                    run_manager.save_checkpoint(meta_logger, dir_checkpoint, dir_final_model,
                                                config_name, remark, test_content)
            writer.close()
            run_manager.end_run()
            # run_manager.save(run if args.custom_combo is "True" else "Results_of_all_runs", system_mode=system_mode)
            run_manager.save(file_name=config_name, system_mode=system_mode, remark=remark, test_content=test_content)


if __name__ == "__main__":
    from data.dataset import NoiseDataset, BatDataset, bat_class_list, partition_dataset
    from data.path_provider import dir_results, dir_results_remote, provide_paths
    from optimization_core.models import YPNet, YPNetNew
    from torch.utils.data import ConcatDataset

    # Define mode and set up main paths
    system_mode = input("Enter system mode. Choose either 'local_mode' or 'remote_mode': ")
    if system_mode == "local_mode":
        path_prefix = dir_results
        main_paths = "argparse_configs/mp_local_2019.txt"
        local_or_remote = "local"
    elif system_mode == "remote_mode":
        path_prefix = dir_results_remote
        main_paths = "argparse_configs/mp_remote_2019.txt"
        local_or_remote = "remote"
    else:
        path_prefix = "CHANGE_DIR"

    system_mode, dir_main, dir_bats_txt, dir_noise_txt, dir_data_txts, csv_path_main, dir_raw_data = provide_paths(
        local_or_remote=local_or_remote, year=2019)

    dir_checkpoint = f'{path_prefix}/pth_files/checkpoints/'
    dir_final_model = f'{path_prefix}/pth_files/final_model/'

    # Define logger
    meta_logger_maker = LoggerMaker(f"{path_prefix}/logs/meta_data", logging.INFO, filelevel=logging.INFO,
                                    filehandling=True)
    meta_logger = meta_logger_maker.logger  # Contains all meta data of the runs

    # Define dataset instance
    full_set_Noise = NoiseDataset(dir_main, dir_noise_txt, "W_05", "MFCC")
    full_set_Bats = BatDataset(dir_main, dir_bats_txt, ["Myotis"], sec=0 + 1, attr_oh="W_05", spec_type="MFCC")
    full_set = ConcatDataset([full_set_Bats, full_set_Noise])
    partitions = partition_dataset(full_set, train_size=0.6, val_size=0.2, train_sub_size=0.05, toy_factor=0.3)
    dataset = partitions[0:3]

    # Define model
    net = YPNetNew
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define parameters
    optimizer = "ADAM"
    model_type = YPNetNew
    train_max_epochs = [10]
    batch_size = [32]
    learning_rate = [0.0001]
    momentum = [0.9]
    weight_decay = [0.00003]
    dropout = [0.2]
    n_hidden_out = [128]
    score_threshold = 0.5
    n_classes = 2

    dict_tmp = get_training_dict(None, main_paths, optimizer, learning_rate, momentum,
                                 weight_decay, dropout, batch_size,
                                 n_hidden_out, train_max_epochs,
                                 custom_combo="False", custom_parameters=None)
    argparse_train_dict = vars(args)
    argparse_train_dict.update(dict_tmp)

    train_net_auto(args, net, dataset[0:3], device, n_classes=2,
                   class_mode="binary", species_or_genus="genus",
                   bat_class_list=bat_class_list,
                   save_cp=True, meta_logger=meta_logger, remark="Remark_TEST_01",
                   val_2nd_set=True, val_2nd_set_name="test", system_mode=system_mode,
                   dir_checkpoint=dir_checkpoint, dir_final_model=dir_final_model,
                   path_prefix=path_prefix, test_content="Test_Content",
                   score_threshold=score_threshold, score_by_majority=False)
