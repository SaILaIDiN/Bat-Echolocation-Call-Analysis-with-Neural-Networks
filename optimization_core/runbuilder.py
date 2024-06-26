""" This file contains the following
    1. RunBuilder class which helps to create all the possible runs from your set of hyperparameters efficiently
    2. RunManager class which manages each run that comes from the RunBuilder class
"""
# For RunBuilder
from collections import OrderedDict
from collections import namedtuple  # is a function that returns a subclass of tuple with named fields
from itertools import product  # cartesian product, returns as tuple of combined values, odometer-like, right to left
# For RunManager
import torch
import os
import pandas as pd
import time
import json
from data.path_provider import dir_results, dir_results_remote


class RunBuilder:
    """ class for efficient hyperparameter experimentation """

    @staticmethod
    def get_runs(params, custom_combo):
        """ return all possible hyperparameter combinations as a list of named tuples.
            params (OrderedDict if custom_combo false, 2D list if custom_combo true)
            custom_combo (str): decides the mode for namedtuple-creation for the Run instances
        """

        runs = []

        if custom_combo == "True":
            Run = namedtuple("Run", params[0])  # Note this is a subclass!!! Note params[0] contains the parameter names
            # print: Run(attr1= , attr2= , attr3= ) a single run, single sample of hyperparameters

            for v in params[1:]:
                runs.append(Run(*v))
        else:
            Run = namedtuple("Run", params.keys())  # Note this is a subclass!!! Note params."keys()" not ."values()"
            # print: Run(attr1= , attr2= , attr3= ) a single run, single sample of hyperparameters

            for v in product(*params.values()):  # the * is unpacking the list of hyperparameters category lists
                # make [[], [], [],...] into [], [], [],... to be readable for the product() function
                runs.append(Run(*v))
                # "Run" as a subclass means "Run()" is a constructor
                # Each time "Run(*v)" is called, an instance is created with the current entry of the cartesian product
                # the asterisk * is again used to unpack the parameter value for each class from the container tuple

        return runs


class RunManager:
    """ This class manages each run that comes from the RunBuilder class.
        All important results are tracked and prepared to a readable csv and json file.
        Only tensorboard-related features are covered directly inside the training loop.
        The idea is to take a look at the csv file and quickly see if any model has good values.
        After that if one wants to go deeper into a certain good-looking run, one can switch to the tensorboard run file.
    """
    def __init__(self):

        self.epoch_count = 0

        self.epoch_loss = 0
        self.epoch_train_f1_score = 0
        self.epoch_train_precision = 0
        self.epoch_train_recall = 0

        self.epoch_val_loss = 0
        self.epoch_val_f1_score = 0
        self.epoch_val_precision = 0
        self.epoch_val_recall = 0

        self.train_subset_loss = 0
        self.epoch_train_subset_f1_score = 0
        self.epoch_train_subset_precision = 0
        self.epoch_train_subset_recall = 0

        self.val_loss = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None

    def begin_run(self, run, network):
        """ Extracts the code needed to prepare a run """
        self.run_start_time = time.time()
        self.run_data = []

        self.run_params = run
        self.run_count += 1

        self.network = network

    def end_run(self):
        self.epoch_count = 0

    def begin_epoch(self):
        """ Reset necessary parameters of epoch before starting the next one and save the start time """
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_train_f1_score = 0
        self.epoch_train_precision = 0
        self.epoch_train_recall = 0

        self.epoch_val_loss = 0
        self.epoch_val_f1_score = 0
        self.epoch_val_precision = 0
        self.epoch_val_recall = 0

        self.train_subset_loss = 0
        self.epoch_train_subset_f1_score = 0
        self.epoch_train_subset_precision = 0
        self.epoch_train_subset_recall = 0

        self.val_loss = 0  # This is the val loss of the current validation step. Can happen multiple times per epoch

    def end_epoch(self):
        """ Compute all values inside an epoch and track the time.
            Note: When a specific value/parameter in results is computed more than once per epoch,
                  only the last assigned value is stored in results.
                  The tracking of values with code outside of RunManager is not bound to this condition.
                  This should be considered when tracking values with multiple procedures. (Logger, Tensorboard) """

        epoch_duration = time.time() - self.epoch_start_time
        interm_run_duration = time.time() - self.run_start_time

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["train loss"] = round(self.epoch_loss, 3)
        results["val loss"] = round(self.epoch_val_loss, 3)
        results["test loss"] = round(self.train_subset_loss, 3)

        results["train f1-score"] = round(self.epoch_train_f1_score, 3)
        results["val f1-score"] = round(self.epoch_val_f1_score, 3)
        results["test f1-score"] = round(self.epoch_train_subset_f1_score, 3)

        results["train precision"] = round(self.epoch_train_precision, 3)
        results["val precision"] = round(self.epoch_val_precision, 3)
        results["test precision"] = round(self.epoch_train_subset_precision, 3)

        results["train recall"] = round(self.epoch_train_recall, 3)
        results["val recall"] = round(self.epoch_val_recall, 3)
        results["test recall"] = round(self.epoch_train_subset_recall, 3)

        # results["val loss"] = round(self.val_loss, 3)
        # results["epoch duration [s]"] = round(epoch_duration, 2)
        results["interm. run duration [s]"] = round(interm_run_duration, 2)
        for k, v in self.run_params._asdict().items():
            # run_params contains a single run instance like Run(lr=0.01, batch_size=1000), a namedtuple
            # by using _asdict() the object is turned into a dictionary
            # then one can use items() to store all pairs of key and value as tuples inside a list
            # can we use another more simple way to get an iterable for k and v?
            # the only advantage is that _asdict() can convert many different data types
            # the items() function stores the tuples inside a special list where you automatically
            # iterate over keys on first index and value in second index, but without cartesian product!
            results[k] = v

        del results["epochs"]
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient="columns")

        return epoch_duration, interm_run_duration

    def track_inter_loss(self, train_subset_loss, val_loss):
        """ (UNUSED ATM) Called after each intermediate step that computes validation and training_subset loss.
            Has to be called inside the training loop. """
        self.train_subset_loss = train_subset_loss.item()
        self.val_loss = val_loss.item()

    def track_inter_performance_metrics(self, train_subset_f1_score, train_subset_precision, train_subset_recall):
        """ Called after each epoch to track the last scores of the test/train-subset from the current epoch.
            Has to be called inside the training loop. """
        self.epoch_train_subset_f1_score = train_subset_f1_score.item()
        self.epoch_train_subset_precision = train_subset_precision.item()
        self.epoch_train_subset_recall = train_subset_recall.item()

    def track_epoch_loss(self, train_loss, val_loss):
        """ Called after each epoch to track the loss over the current epoch.
            Has to be called inside the training loop. """
        self.epoch_loss = train_loss.item()
        self.epoch_val_loss = val_loss.item()

    def track_epoch_performance_metrics(self, train_f1_score, train_precision, train_recall,
                                        val_f1_score, val_precisison, val_recall):
        """ Called after each epoch to track the last scores of the train and val set from the current epoch.
            Has to be called inside the training loop. """
        self.epoch_train_f1_score = train_f1_score.item()
        self.epoch_train_precision = train_precision.item()
        self.epoch_train_recall = train_recall.item()

        self.epoch_val_f1_score = val_f1_score.item()
        self.epoch_val_precision = val_precisison.item()
        self.epoch_val_recall = val_recall.item()

    def get_config_name(self, name_optimizer):
        """ Returns a string that contains all necessary configurations of a run by the given optimizer name. """
        if name_optimizer == "SGD":
            config_name = f"LR_{self.run_params.lr}_MOM_{self.run_params.momentum}_L2_{self.run_params.weight_decay}" \
                          f"_BS_{self.run_params.batch_size}_DO_{self.run_params.dropout}"
        elif name_optimizer == "ADAM":
            config_name = f"LR_{self.run_params.lr}"\
                          f"_BS_{self.run_params.batch_size}_DO_{self.run_params.dropout}"
        else:
            print("No configuration name given!")

        return config_name

    def save_checkpoint(self, train_logger, dir_checkpoint, dir_final_model,
                        config_name="", remark="", test_content=""):
        """ Save .pth-files """
        os.makedirs(dir_checkpoint, exist_ok=True)
        try:
            train_logger.info('Meta-Logger is verified! Ready to track storing of .pth-files.')
        except TypeError:
            print("No meta logger defined to track saving of .pth-files.")
            pass

        timestr = time.strftime("%Y%m%d-%H%M%S")
        if self.epoch_count < self.run_params.epochs:
            torch.save(self.network.state_dict(),
                       dir_checkpoint + f'CP_epoch{self.epoch_count}_{config_name}_{remark}.pth')
            train_logger.info(f'Checkpoint {self.epoch_count} saved !')
        if self.epoch_count == self.run_params.epochs:
            os.makedirs(dir_final_model + test_content, exist_ok=True)
            torch.save(self.network.state_dict(), dir_final_model + test_content + "/" +
                       f'{timestr}_Model_{config_name}_{remark}.pth')
            train_logger.info(f'Final model after epoch {self.epoch_count} saved !')

    def save(self, file_name, system_mode, remark="", test_content=""):
        """ Saves the run_data in two formats (JSON, CSV) """
        if system_mode == "local_mode":
            os.makedirs(f'{dir_results}/runs/run_tables/{test_content}',
                        exist_ok=True)
            pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(
                f'{dir_results}/runs/run_tables/{test_content}/'
                f'{file_name}_{remark}.csv')
            with open(f'{dir_results}/runs/run_tables/{test_content}/'
                      f'{file_name}_{remark}.json',
                      'w', encoding='utf-8') as f:
                json.dump(self.run_data, f, ensure_ascii=False, indent=4)

        elif system_mode == "remote_mode":
            os.makedirs(f'{dir_results_remote}/runs/run_tables/{test_content}',
                        exist_ok=True)
            pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(
                f'{dir_results_remote}/runs/run_tables/{test_content}/'
                f'{file_name}_{remark}.csv')

            with open(f'{dir_results_remote}/runs/run_tables/{test_content}/'
                      f'{file_name}_{remark}.json', 'w',
                      encoding='utf-8') as f:
                json.dump(self.run_data, f, ensure_ascii=False, indent=4)
