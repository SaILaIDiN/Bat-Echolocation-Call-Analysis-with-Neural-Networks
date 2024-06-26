""" This script is for an automated model training. For check_years_binary.py.
    It imports the training function and its parser and updates the parser in order to adaptively execute multiple runs.
"""


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


if __name__ == '__main__':
    import torch
    import logging
    from optimization_core.train import train_net_auto
    from optimization_core.train import parser as parser_train
    from optimization_core.models import YPNet, YPNetNew
    from optimization_core.logger_maker import LoggerMaker
    from supervised_analysis.check_years_binary import dataset_01, dataset_02, dataset_03, dataset_04, \
        dataset_05, dataset_06, dataset_07, dataset_08, bat_class_list
    from data.path_provider import dir_results, dir_results_remote

    args_train = parser_train.parse_args()
    argparse_train_dict = vars(args_train)

    system_mode = input("Enter system mode. Choose either 'local_mode' or 'remote_mode': ")
    if system_mode == "local_mode":
        path_prefix = dir_results
        main_paths = "argparse_configs/mp_local_2019.txt"
    elif system_mode == "remote_mode":
        path_prefix = dir_results_remote
        main_paths = "argparse_configs/mp_remote_2019.txt"
    else:
        path_prefix = "CHANGE_DIR"

    dir_checkpoint = f'{path_prefix}/pth_files/checkpoints/binary/'
    dir_final_model = f'{path_prefix}/pth_files/final_model/binary/'

    # Set up loggers
    meta_logger_maker = LoggerMaker(f"{path_prefix}/logs/meta_data", logging.INFO, filelevel=logging.INFO,
                                    filehandling=True)
    meta_logger = meta_logger_maker.logger  # Contains all meta data of the runs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up training loop
    list_remark = [f"_run_{i+1}" for i in range(0, 5)]

    test_content_list = ["01_Test_1_NE19_train_NW19_test", "02_Test_1_NW19_train_NE19_test",
                         "03_Test_2_NE20_train_NW20_test", "04_Test_2_NW20_train_NE20_test",
                         "05_Test_3_NE19_train_NE20_test", "06_Test_3_NE20_train_NE19_test",
                         "07_Test_4_NW19_train_NW20_test", "08_Test_4_NW20_train_NW19_test"]

    dataset_list = [dataset_01, dataset_02, dataset_03, dataset_04,
                    dataset_05, dataset_06, dataset_07, dataset_08]

    dict_all_seeds_random = {"data_aug_seeds": [0, 5, 13, 27, 43], "data_sample_seeds": [0, 5, 13, 27, 43],
                             "weights_seeds": [0, 5, 13, 27, 43]}

    seed_dict_list = [dict_all_seeds_random, dict_all_seeds_random, dict_all_seeds_random, dict_all_seeds_random,
                      dict_all_seeds_random, dict_all_seeds_random, dict_all_seeds_random, dict_all_seeds_random]

    optimizer = "ADAM"
    list_model_type = [YPNetNew]
    train_max_epochs = [10]
    list_batch_size = [[32]]
    list_learning_rate = [[0.0001]]
    list_momentum = [[0.9]]
    list_weight_decay = [[0.00003]]
    list_dropout = [[0.2]]
    n_hidden_out = [128]
    score_threshold = 0.5
    n_classes = 2

    for test_content, seed_dict, dataset in zip(test_content_list, seed_dict_list, dataset_list):
        for remark, seed_data, seed_weights in zip(list_remark,
                                                   seed_dict["data_sample_seeds"],
                                                   seed_dict["weights_seeds"]):
            for model_type in list_model_type:
                for batch_size in list_batch_size:
                    for learning_rate in list_learning_rate:
                        for momentum in list_momentum:
                            for weight_decay in list_weight_decay:
                                for dropout in list_dropout:
                                    torch.manual_seed(seed_weights)
                                    torch.cuda.manual_seed_all(seed_weights)
                                    torch.backends.cudnn.deterministic = True
                                    torch.backends.cudnn.benchmark = False
                                    dict_tmp = get_training_dict(None, main_paths, optimizer, learning_rate, momentum,
                                                                 weight_decay, dropout, batch_size,
                                                                 n_hidden_out, train_max_epochs,
                                                                 custom_combo="False", custom_parameters=None)
                                    argparse_train_dict.update(dict_tmp)
                                    train_net_auto(args_train, model_type, dataset[0:3], device, n_classes=n_classes,
                                                   class_mode="binary", bat_class_list=bat_class_list,
                                                   save_cp=True, meta_logger=meta_logger, remark=remark,
                                                   val_2nd_set=True, val_2nd_set_name="test", system_mode=system_mode,
                                                   dir_checkpoint=dir_checkpoint, dir_final_model=dir_final_model,
                                                   path_prefix=path_prefix, test_content=test_content,
                                                   score_threshold=score_threshold, score_by_majority=False)
