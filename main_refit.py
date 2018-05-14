"""
Main file for fitting neural networks and getting variable importance.
This will fit a separate network for each conditional mean
"""
import sys
import argparse
import logging
import tensorflow as tf

import numpy as np

from data_generator import DataGenerator
from estimate_variable_importance import calculate_var_imports_refits
from common import *

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
        help="Num training samples (for simulations only)",
        default=2500)
    parser.add_argument('--num-test',
        type=int,
        help="Num testing samples (for simulations only)",
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
        Indexing starts at zero!
        Example: 0;1;2;3
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
        Example: 6,2,1;6,2,2,1
        """,
        default="6,3,1")
    parser.add_argument('--cond-layer-sizes',
        type=str,
        help="""
        Semi-colon separated lists of comma-separated lists of layer sizes (for tuning over)
        This supposes that we search over the same set of structures for estimating all reduced
        conditional means.
        Example: 6,2,1;6,2,2,1
        """,
        default=None)
    parser.add_argument('--cond-layer-sizes-separate',
        type=str,
        help="""
        Plus-separated, semi-colon separated lists of comma-separated lists of layer sizes (for tuning over).
        The plus delimiter is to separate the network structures to search over for each var importance group.
        (Either specify --cond-layer-sizes or --cond-layer-sizes-separate, not both)
        Example: 6,2,1;6,2,2,1+4,2,1+4,10,1
        """,
        default=None)
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
    parser.set_defaults(aws=False)
    args = parser.parse_args()
    args.nn_struct = "basic"

    # Parse var import idxs
    args.var_import_group_sizes = None
    if args.var_import_idx is None:
        args.var_import_idx = [[i] for i in range(args.num_p)]
    else:
        args.var_import_idx = [process_params(substr, int) for substr in args.var_import_idx.split(";")]

    # Parse parameters to CV over
    args.ridge_params = process_params(args.ridge_params, float)
    args.layer_sizes = [process_params(substr, int) for substr in args.layer_sizes.split(";")]
    if args.cond_layer_sizes:
        # Use the same for all reduced conditional NNs
        args.cond_layer_sizes_separate = [
                [process_params(substr, int) for substr in args.cond_layer_sizes.split(";")]
                for _ in args.var_import_idx]
    else:
        # Different layer sizes for different var importance groups
        args.cond_layer_sizes_separate = [
            [process_params(substr, int) for substr in cond_layer_sizes.split(";")]
            for cond_layer_sizes in args.cond_layer_sizes_separate.split("+")]

    # Check that the neural net layer sizes are appropriate
    assert(np.all([lay_size[0] == args.num_p for lay_size in args.layer_sizes]))

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
    print(args)
    logging.info(args)
    
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
        
        # Write data
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
    }]
    
    # Fit neural network and calculate variable importance
    var_imports, fitted_models = calculate_var_imports_refits(
        data,
        param_grid=param_grid,
        cond_layer_sizes=args.cond_layer_sizes_separate,
        var_import_idxs=args.var_import_idx)

    # Save model
    pickle_to_file(fitted_models, args.model_file)
    # Store var import results
    pickle_to_file(var_imports, args.var_import_file)

    # Print output
    for i in range(len(var_imports)):
        v = var_imports[i]["std-True"]
        if i == 0:
            logging.info("full final r2 %f (1 is best)", v["r2.full"])
            logging.info("full final r2 test %f", v["r2.test.full"])
        logging.info("small final r2 %d : %f", i, v["r2.small"])
        logging.info("small final r2 test %d : %f", i, v["r2.test.small"])
        logging.info("one step est std=True %d : %f, %s", i, v["onestep"], v["onestep.ci"])
        v_not_std = var_imports[i]["std-False"]
        logging.info("one step est std=False %d : %f, %s",
                i,
                v_not_std["onestep"],
                v_not_std["onestep.ci"])

if __name__ == "__main__":
    main(sys.argv[1:])
