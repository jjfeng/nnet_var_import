"""
Main file for fitting neural networks and getting variable importance.
This will fit a single augmented MTL network to estimate all the needed conditional means
"""
import sys
import argparse
import logging
import tensorflow as tf

import numpy as np

from data_generator import DataGenerator
from estimate_variable_importance import calculate_var_imports_no_refit
from common import *

import data_gen_funcs
import sim_func_var_importance

def parse_args():
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed',
        type=int,
        help='Random number generator seed for replicability',
        default=1)
    parser.add_argument('--num-p',
        type=int,
        help="Dimension of X",
        default=6)
    parser.add_argument('--num-train',
        type=int,
        help="Num training samples",
        default=2500)
    parser.add_argument('--num-test',
        type=int,
        help="Num testing samples",
        default=5000)
    parser.add_argument('--max-iters',
        type=int,
        help="max training iters",
        default=3000)
    parser.add_argument('--num-inits',
        type=int,
        help="num random initializations",
        default=1)
    parser.add_argument('--sim-func',
        type=str,
        help="simulation function. If None, then supposes there is input data",
        default=None,
        choices=["six_func", "eight_additive"])
    parser.add_argument('--sim-noise-sd',
        type=float,
        help="simulation noise std dev",
        default=0)
    parser.add_argument('--input-data',
        type=str,
        help="input data file name, needs to be a pickle file containing a Dataset obj (see data_generator.py)",
        default=None)
    parser.add_argument('--var-import-idx',
        type=str,
        help="""
        semi-colon separated lists of comma-separated lists of variables that we want the var import of.
        if None, then calculates the var import of each variable individually.
        Example: 1;2;3,4,5
        """,
        default=None)
    parser.add_argument('--var-import-group-sizes',
        type=int,
        help="""
        Specifies we are interested in all var import groups up to this size
        Example: 4 means in all var import groups up to size 4
        """,
        default=None)
    parser.add_argument('--ridge-params',
        type=str,
        help="comma separated list of ridge params (for tuning over)",
        default="0.005")
    parser.add_argument('--layer-sizes',
        type=str,
        help="""
        semi-colon separated lists of comma-separated lists of layer sizes (for tuning over)
        Example: 12,2,1;12,2,2,1
        """,
        default="12,3,1")
    parser.add_argument('--act-func',
        type=str,
        help="activiation function",
        default="relu",
        choices=["tanh", "relu"])
    parser.add_argument('--output-act-func',
        type=str,
        help="activiation function",
        default=None,
        choices=["sigmoid"])
    parser.add_argument('--cv',
        type=int,
        help="num cross-validation folds",
        default=3)
    parser.add_argument('--out-dir',
        type=str,
        help="""
        Name of output directory. will make a new folder for this seed.
        Will add log file, data file if simulating data, fitted model, and confidence interval csv file
        """,
        default="_output")
    parser.add_argument('--sgd-sample-size',
        type=int,
        help="sgd sample size",
        default=2000)
    parser.add_argument('--nan-fill-config-file',
        type=str,
        help="""
        Name of config file for filling in nan values.
        nan values in the input data represent missing values.
        The config file should be a json indicating the range of the uniform distribution to draw from for filling in these nans.
        nans in the ICU data are missing data that are likely not missing at random -- this flag instructs how to impute missing values
        for different features.
        If None, then we will use the usual process for filling in missing values (see --missing-values-fill-rv)
        """,
        default=None)
    parser.add_argument('--missing-value-fill',
        type=float,
        help="""
        What prob distribution to draw from to input values for the missing values
        (W_s in the manuscript)
        Note that we use the same distribution for all missing values (does not depend on the variable importance group s)
        and they are all iid.
        * None -- standard normal rv
        * or a float that will be the constant value (typically zero)
        """,
        default=None)

    parser.set_defaults(nan_fill_config = None)
    args = parser.parse_args()
    args.nn_struct = "3"

    if args.sim_func == "eight_additive":
        args.sim_func_reduced = "eight_additive_reduced"
    else:
        args.sim_func_reduced = None

    print(args)

    if args.var_import_idx is None:
        if args.var_import_group_sizes is None:
            args.var_import_idx = [[i] for i in range(args.num_p)]
        else:
            args.var_import_idx = nonempty_powerset(range(args.num_p), max_size=args.var_import_group_sizes)
    else:
        args.var_import_idx = [process_params(substr, int) for substr in args.var_import_idx.split(";")]

    # Parse parameters to CV over
    args.ridge_params = process_params(args.ridge_params, float)
    args.layer_sizes = [process_params(substr, int) for substr in args.layer_sizes.split(";")]

    if args.nan_fill_config_file is not None:
        args.nan_fill_config = read_nan_fill_config_file(args.nan_fill_config_file)

    # create output file
    args.out_dir_inner = make_inner_outdir_name(args)
    print("output directory %s" % args.out_dir_inner)
    args.data_file = make_data_file_name(args)
    args.model_file = make_model_file_name(args)
    args.var_import_file = make_var_import_file_name(args)
    args.log_file = make_log_file_name(args)
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    logging.basicConfig(format="%(message)s",filename=args.log_file, level=logging.DEBUG)
    logging.info(args)
    
    # This random seed thing seems to only apply to the data-generation process.
    # The initialization of the neural net doesn't seem to be affected by this
    # ... which is really annoying.
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.sim_func is not None:
        # Create data
        data_gen = DataGenerator(
            func_name=args.sim_func,
            n_train = args.num_train,
            n_test = args.num_test,
            num_p = args.num_p,
            noise_sd = args.sim_noise_sd,
        )
        data = data_gen.create_data()

        # Write data to file
        pickle_to_file(data, args.data_file)
    else:
        # Read data
        data = pickle_from_file(args.input_data)

    param_grid = [{
        'layer_sizes': args.layer_sizes,
        'ridge_param': args.ridge_params,
        'max_iters': [args.max_iters],
        'num_inits': [args.num_inits],
        'act_func': [args.act_func],
        'output_act_func': [args.output_act_func],
        'var_import_idxs': [args.var_import_idx],
        'sgd_sample_size': [args.sgd_sample_size],
        'nan_fill_config': [args.nan_fill_config],
        'missing_value_fill': [args.missing_value_fill],
    }]
    
    # Fit neural network and calculate variable importance
    reduced_func = getattr(data_gen_funcs, args.sim_func_reduced) if args.sim_func_reduced else None
    var_imports, fitted_model = calculate_var_imports_no_refit(
        data,
        param_grid=param_grid,
        cv=args.cv,
        reduced_func=reduced_func)

    # Save model
    pickle_to_file(fitted_model, args.model_file)
    # Store var import results
    pickle_to_file(var_imports, args.var_import_file)
    
    # Print output for each var import estimate
    coverage = []
    for i, var_group in enumerate(args.var_import_idx):
        v = var_imports[i]["std-True"]
        if i == 0:
            logging.info("full final r2 %f (1 is best)", v["r2.full"])
            logging.info("full final r2 test %f", v["r2.test.full"])
            logging.info("full final mse train %f", v["mse.train.full"])
            logging.info("full final mse test %f", v["mse.test.full"])
        logging.info(" --- small --- %s ---- ", var_group)
        logging.info("small final r2: %f", v["r2.small"])
        logging.info("small final r2 test: %f", v["r2.test.small"])
        if v["mse.train.small"] is not None:
            logging.info("small final mse train: %f", v["mse.train.small"])
            logging.info("small final mse test: %f", v["mse.test.small"])
        logging.info("one step est std=True: %f, %s", v["onestep"], v["onestep.ci"])
        v_not_std = var_imports[i]["std-False"]
        logging.info("one step est std=False: %f, %s",
                v_not_std["onestep"],
                v_not_std["onestep.ci"])

    if args.sim_func_reduced:
        logging.info("Average coverage over all the groups: %f (%d/%d)", np.mean(coverage), np.sum(coverage), len(coverage))

if __name__ == "__main__":
    main(sys.argv[1:])
