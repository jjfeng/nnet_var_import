import logging
import numpy as np

from sklearn.model_selection import GridSearchCV

from data_generator import Dataset
from neural_network_basic import NeuralNetworkBasic
from neural_network_aug_mtl import NeuralNetworkAugMTL
import variable_importance as vi
import variable_importance_se as se
import variable_importance_ci as ci

def calculate_var_imports_refits(dataset, param_grid, cond_layer_sizes, var_import_idxs=None, cv=3):
    """
    Estimate variable importance, assumes we need to refit for each set of variable groups

    @param dataset: Dataset
    @param param_grid: dictionary to CV over, contains all values for initializing NeuralNetworkAugMTL
                    (see docs for GridSearchCV from scikit)
    @param cond_layer_sizes: a list of list of network structures, each list of network structures is what we search over
                            for estimating the reduced conditional means, ordering according to param_grid[0]["var_import_idxs"]
    @param cv: number of folds for Cross validation
    @return tuple with:
        1. list of dicts, in the order of the variable groups in param_grid[0]["var_import_idxs"]
            Each dict contains: {
                "std-True": dict corresponding to naive and one-step estimates of var importance (and conf intervals) for standardized variable importance,
                "std-False": dict corresponding to naive and one-step estimates of var importance (and conf intervals) for not-standardized variable importance,
            }
            
    """
    # Pick best parameters via cross validation
    best_params, cv_results = _get_best_params(NeuralNetworkBasic, param_grid, dataset, cv=cv)
    logging.info("Best params %s", str(best_params))

    # Fit for the full conditional mean
    final_nn = NeuralNetworkBasic(**best_params)
    final_nn.fit(dataset.x_train, dataset.y_train)
    
    # Calculate some stats on our fitted network
    full_fit = final_nn.predict(dataset.x_train)
    r2_full = 1 - np.sum((dataset.y_train - full_fit) ** 2)/np.sum((dataset.y_train - np.mean(dataset.y_train)) ** 2)
    full_fit_test = final_nn.predict(dataset.x_test)
    r2_full_pred = 1 - np.sum((dataset.y_test - full_fit_test) ** 2)/np.sum((dataset.y_test - np.mean(dataset.y_test)) ** 2)

    var_imports = []
    num_p = dataset.x_train.shape[1]
    fitted_models = {"full": final_nn.model_params, "cond": {}, "cv_results": cv_results, "cond_cv_results": {}}

    # set up which var importance values to calculate if not passed in
    if var_import_idxs is None:
        var_import_idxs = range(dataset.x_train.shape[1])

    # Calculate some stats about our network regarding each of the variable groups
    # Get the estimated variable importance values
    for i, del_idx_group in enumerate(var_import_idxs):
        # Prepare dataset without the particular variables
        cond_x_train = np.delete(dataset.x_train, del_idx_group, axis=1)
        cond_x_test = np.delete(dataset.x_test, del_idx_group, axis=1)
        cond_dataset = Dataset(cond_x_train, dataset.y_train, dataset.y_train_true, cond_x_test, dataset.y_test, dataset.y_train_true)
        cond_param_grid = param_grid
        cond_param_grid[0]["layer_sizes"] = cond_layer_sizes[i]

        # Fit for reduced conditional means
        best_cond_params, cv_results_cond = _get_best_params(NeuralNetworkBasic, cond_param_grid, cond_dataset, cv=cv)
        logging.info("Best cond params %s", str(best_cond_params))
        cond_nn = NeuralNetworkBasic(**best_cond_params)

        # Refit!
        cond_nn.fit(cond_x_train, dataset.y_train)
        fitted_models["cond"][str(del_idx_group)] = cond_nn.model_params
        fitted_models["cond_cv_results"][str(del_idx_group)] = cv_results_cond

        # Get new fitted values
        small_fit = cond_nn.predict(cond_x_train)
        small_fit_test = cond_nn.predict(cond_x_test)

        ## calculate R^2
        r2_small = 1 - np.sum((dataset.y_train - small_fit) ** 2)/np.sum((dataset.y_train - np.mean(dataset.y_train)) ** 2)

        ## calculate predicted R^2
        r2_small_pred = 1 - np.sum((dataset.y_test - small_fit_test) ** 2)/np.sum((dataset.y_test - np.mean(dataset.y_test)) ** 2)
        logging.info("==== %s =======", str(del_idx_group))
        logging.info("r2 small: %f", r2_small)
        logging.info("r2 small pred: %f", r2_small_pred)

        ## calculate estimators both standardized and unstandardized
        var_import_ret = {}
        for std in [True, False]:
            ests = vi.variableImportance(full_fit, small_fit, dataset.y_train, std)
            naive = np.array([ests[0]])
            onestep = np.array([ests[1]])

            ## calculate standard error for one-step
            onestep_se = se.variableImportanceSE(full_fit, small_fit, dataset.y_train, std)

            ## calculate CI for one-step
            onestep_ci = ci.variableImportanceCI(onestep, onestep_se, level = 0.95)


            ret = {
                    'naive':np.array(naive), # naive estimate
                    'onestep':onestep, # one-step estimate
                    'onestep.se':onestep_se, # std error of one-step est
                    'onestep.ci':onestep_ci, # conf int for var import
                    'r2.full': r2_full, #R^2 for the full conditional mean on train data
                    'r2.small': r2_small, # R^2 for the reduced conditional mean  on train data
                    'r2.test.full': r2_full_pred, #R^2 for the full conditional mean on test data
                    'r2.test.small':r2_small_pred} # R^2 for the reduced conditional mean  on test data
            var_import_ret["std-%s" % std] = ret
        var_imports.append(var_import_ret)
    return var_imports, fitted_models

def calculate_var_imports_no_refit(dataset, param_grid, cv=3, reduced_func=None):
    """
    Estimate variable importance, assumes we do not need to refit for each set of variable groups
    It relies on a neural network structure that has an ability to estimate E[y|X_{-s}]

    @param dataset: Dataset
    @param param_grid: dictionary to CV over, contains all values for initializing NeuralNetworkAugMTL
                    (see docs for GridSearchCV from scikit)
    @param cv: number of folds for Cross validation
    @param reduced_func: a function that returns the reduced conditional mean
                        used to calculate coverage
    @return Tuple with:
        1. list of dicts, in the order of the variable groups in param_grid[0]["var_import_idxs"]
            Each dict contains: {
                "std-True": dict corresponding to naive and one-step estimates of var importance (and conf intervals) for standardized variable importance,
                "std-False": dict corresponding to naive and one-step estimates of var importance (and conf intervals) for not-standardized variable importance,
            }
        2. dictionary containing the fitted models
    """
    # Do cross validation
    best_params, cv_results = _get_best_params(NeuralNetworkAugMTL, param_grid, dataset, cv=cv)
    logging.info("Best params %s", str(best_params))

    # Fit the network with the chosen parameters from CV
    final_nn = NeuralNetworkAugMTL(**best_params)
    final_nn.fit(dataset.x_train, dataset.y_train)

    # Calculate some stats about our network
    full_fit = final_nn.predict(dataset.x_train, filter_idx=None)
    r2_full = 1 - np.sum((dataset.y_train - full_fit) ** 2)/np.sum((dataset.y_train - np.mean(dataset.y_train)) ** 2)
    full_fit_test = final_nn.predict(dataset.x_test, filter_idx=None)
    r2_full_pred = 1 - np.sum((dataset.y_test - full_fit_test) ** 2)/np.sum((dataset.y_test - np.mean(dataset.y_test)) ** 2)
    mse_pred = np.mean((dataset.y_test_true - full_fit_test) ** 2)
    mse_train = np.mean((dataset.y_train_true - full_fit) ** 2)

    # Calculate some stats about our network regarding each of the variable groups
    # Get the estimated variable importance values
    var_imports = []
    num_p = dataset.x_train.shape[1]
    for filter_idx in best_params["var_import_idxs"]:
        small_fit = final_nn.predict(dataset.x_train, filter_idx)
        small_fit_test = final_nn.predict(dataset.x_test, filter_idx)

        if reduced_func:
            y_train_small = np.reshape(reduced_func(dataset.x_train, filter_idx), (dataset.x_train.shape[0], 1))
            y_test_small = np.reshape(reduced_func(dataset.x_test, filter_idx), (dataset.x_test.shape[0], 1))

            mse_small_train = np.mean((y_train_small - small_fit) ** 2)
            mse_small_test = np.mean((y_test_small - small_fit_test) ** 2)
        else:
            mse_small_train = None
            mse_small_test = None

        ## calculate R^2
        r2_small = 1 - np.sum((dataset.y_train - small_fit) ** 2)/np.sum((dataset.y_train - np.mean(dataset.y_train)) ** 2)
        
        ## calculate predicted R^2
        r2_small_pred = 1 - np.sum((dataset.y_test - small_fit_test) ** 2)/np.sum((dataset.y_test - np.mean(dataset.y_test)) ** 2)
            
        std_ret = {"var_import": filter_idx}
        for std in [True, False]:
            ## calculate estimators
            ests = vi.variableImportance(full_fit, small_fit, dataset.y_train, std)
            naive = np.array([ests[0]])
            onestep = np.array([ests[1]])
            
            ## calculate standard error for one-step
            onestep_se = se.variableImportanceSE(full_fit, small_fit, dataset.y_train, std)
            
            ## calculate CI for one-step
            onestep_ci = ci.variableImportanceCI(onestep, onestep_se, level = 0.95)
            
            ret = {
                    'naive':np.array(naive), # naive estimate
                    'onestep':onestep, # one-step estimate
                    'onestep.se':onestep_se, # std error of one-step est
                    'onestep.ci':onestep_ci, # conf int for var import
                    'r2.full': r2_full, #R^2 for the full conditional mean on train data
                    'r2.small': r2_small, # R^2 for the reduced conditional mean  on train data
                    'r2.test.full': r2_full_pred, #R^2 for the full conditional mean on test data
                    'r2.test.small':r2_small_pred, # R^2 for the reduced conditional mean  on test data
                    'mse.train.full': mse_train, #MSE to full conditional mean (not observed resps) on the training data
                    'mse.test.full': mse_pred, # MSE to full conditional mean (not observed resps) on the test data
                    'mse.train.small': mse_small_train, # MSE to reduced conditional mean (not observsed responses), train data
                    'mse.test.small': mse_small_test} # MSE to reduced conditional mean (not observed responses), test data
            std_ret["std-%s" % std] = ret
        var_imports.append(std_ret)
    fitted_models = {"final_nn": final_nn.model_params, "cv_results": cv_results}
    return var_imports, fitted_models

def _get_best_params(model_cls, param_grid, dataset, cv = 3):
    """
    Runs cross-validation if needed
    @return best params chosen by CV in dict form, `cv_results_` attr from GridSearchCV
    """
    if np.all([len(v) == 1 for k,v in param_grid[0].iteritems()]):
        # Don't run CV if there is nothing to tune
        return {k:v[0] for k,v in param_grid[0].iteritems()}, None
    else:
        # grid search CV to get argmins
        # HACK: half the number of initializations for CV
        param_grid[0]["num_inits"][0] = max(param_grid[0]["num_inits"][0]/2, 1)
        grid_search_cv = GridSearchCV(
                model_cls(),
                param_grid = param_grid,
                cv = cv,
                refit=False)

        ### do cross validation
        grid_search_cv.fit(dataset.x_train, dataset.y_train)
        logging.info("Completed CV")
        logging.info(grid_search_cv.cv_results_)

        # HACK: un-half the number of initializations for the final fitting
        grid_search_cv.best_params_["num_inits"] *= 2
        return grid_search_cv.best_params_, grid_search_cv.cv_results_
