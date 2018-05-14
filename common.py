import os
import numpy as np
import pickle
import json
from itertools import chain, combinations

THRES = 1e-7
ALMOST_ZERO = 0 #1e-8

def make_inner_outdir_name(args):
    if args.sim_func:
        if args.var_import_group_sizes:
            var_import_idx_str = "upto%d" % args.var_import_group_sizes
        else:
            var_import_idx_str = "_".join(["-".join([str(v) for v in vs]) for vs in args.var_import_idx])

        prefix = "%s-%d-%s/noise-sd-%.2f/%s" % (
                args.sim_func, 
                args.num_p, 
                var_import_idx_str,
                args.sim_noise_sd,
                args.num_train)
    else:
        # if relative import, get rid of '/'
        # want to keep the first "." though, for 
        prefix = args.input_data.split("/")[-1].split(".")
        if prefix != '':
            tmp = prefix[:(len(prefix)-1)]
            # if length more than 1, must have had a train fraction
            if len(tmp) > 1:
                prefix = '.'.join(tmp)
            else:
                prefix = tmp[0]
        else:
            prefix = ''.join(args.input_data.split(".")[-1])
    
    outdir_full = "%s/%s/nn_struct_%s/%s" % (args.out_dir, prefix, args.nn_struct, args.act_func)
    if not os.path.exists(outdir_full):
        os.makedirs(outdir_full)
    outdir_full_seed = "%s/%d" % (outdir_full, args.seed)
    if not os.path.exists(outdir_full_seed):
        os.makedirs(outdir_full_seed)
    return outdir_full_seed

def make_data_file_name(args):
    return "%s/data.pkl" % (args.out_dir_inner)

def make_model_file_name(args):
    return "%s/model.pkl" % (args.out_dir_inner)

def make_log_file_name(args):
    return "%s/log.txt" % (args.out_dir_inner)

def make_var_import_file_name(args):
    return "%s/var_import.pkl" % (args.out_dir_inner)

def pickle_to_file(obj, file_name):
    with open(file_name, "w") as f:
        pickle.dump(obj, f, protocol=-1)

def pickle_from_file(file_name):
    with open(file_name, "r") as f:
        out = pickle.load(f)
    return out

def process_params(param_str, dtype):
    if param_str:
        return [dtype(r) for r in param_str.split(",")]
    else:
        return []

def process_seeds(seed_range_str, seeds_str):
    if seed_range_str:
        seed_range = process_params(seed_range_str, int)
        return range(seed_range[0], seed_range[1])
    else:
        return process_params(seeds_str, int)

def nonempty_powerset(iterable, max_size=None):
    """
    @return all subsets of iterable that are not empty
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    subset_sizes = range(1, max_size + 1 if max_size else len(xs) + 1)
    assert(max_size <= len(xs))
    powerset_iterable = chain.from_iterable(combinations(xs,n) for n in subset_sizes)
    return [s for s in powerset_iterable]

def read_nan_fill_config_file(filename):
    """
    @return the dictionary corresponding to the json at `filename`
    """
    with open(filename, "r") as f:
        return json.load(f)
