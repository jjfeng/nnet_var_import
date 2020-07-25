# Variable importance in neural networks

Feng, Jean, Brian Williamson, Noah Simon, and Marco Carone. 2018. “Nonparametric Variable Importance Using an Augmented Neural Network with Multi-Task Learning.” International Conference on Machine Learning 80: 1496–1505. http://proceedings.mlr.press/v80/feng18a.html.

## Installation instructions
Runs with python 2.7 (other versions are not tested).
See `requirements.txt` for python libraries/versions

## Main files
* `main_norefit.py`: Estimates variable importance by estimating the full and reduced conditional means simultaneously in a single neural network
* `main_refit.py`: Estimates variable importance by estimating the full and reduced conditional means using separate neural networks
* `neural_network_aug_mtl.py`: The augmented MTL neural network for estimating full and reduced conditional means simultaneously
* `neural_network_basic.py`: A basic neural network for estimating a single conditional mean function
* `data_generator.py`: Contains objects for storing/simulating datasets
* `variable_importance*`: Computes the variable importance estimates from the estimated conditional means
* `process_icu_data.py`: Code for pre-processing the ICU data example in the paper

## Examples
To estimate the variable importance of each of the six variables in the six-variable function in the paper, run
```
python main_norefit.py --sim-func six_func --cond-layer-sizes 5,2,1
```
To estimate importance of variables groups 1,2;3,4;5,6, run
```
python main_norefit.py --var-import 0,1;2,3;4,5 --sim-func six_func --cond-layer-sizes 5,2,1
```

## Paper Examples
### Six-variable simulation example in the paper
For the simulation where we fit separate networks, we did cross validation over the network structures in the dictionaries below.
```
# Network sizes for estimating the full conditional mean
# Key indicates training set size
LAYER_SIZES = {
    500: "6,5,5,1;6,10,5,1;6,10,10,1;6,20,10,1;6,20,20,1",
    1000: "6,5,5,1;6,10,5,1;6,20,10,1;6,20,20,1",
    2000: "6,10,5,1;6,10,10,1;6,20,10,1;6,20,20,1",
    4000: "6,20,10,1;6,20,20,1;6,40,20,1;6,40,40,1;6,20,20,20,1",
    8000: "6,20,10,1;6,20,20,1;6,40,20,1;6,40,40,1;6,20,20,20,1",
    16000: "6,40,20,1;6,40,40,1;6,20,20,20,1;6,40,20,20,1",
}

# Network sizes for estimating the reduced conditional mean for s={4,5}
# Key indicates training set size
COND_LAYER_SIZES = {
    500: "4,20,20,1;4,40,40,1;4,80,40,1",
    1000: "4,20,20,1;4,40,40,1;4,80,40,1",
    2000: "4,40,40,1;4,80,40,1;4,20,20,20,1",
    4000: "4,40,40,1;4,80,40,1;4,20,20,20,1",
    8000: "4,80,40,1;4,20,20,20,1;4,40,20,20,1",
    16000: "4,20,20,20,1;4,40,20,20,1;4,40,40,20,1",
}
```
We then ran:
```
python main_refit.py --sim-func six_func --num-p 6 --var-import-idx 0,1;2,3;4,5 --sim-noise 0.25 --act-func relu --max-iters 40000 --num-train TRAINING_SIZE --layer-sizes LAYER_SIZES[TRAINING_SIZE] --cond-layer-sizes-separate "4,5,5,1;4,10,5,1+4,5,5,1;4,10,5,1+%s" % COND_LAYER_SIZES[TRAINING_SIZE] --ridge 0.0001,0.000001,0.00000001 --num-inits 2 --cv 5
```

For the simulation where we fit a single network, we did cross validation over the network structures in the dictionary below.
```
LAYER_SIZES = {
    500: "12,5,5,1;12,10,5,1;12,10,10,1;12,20,10,1;12,20,20,1",
    1000: "12,20,10,1;12,20,20,1;12,40,20,1;12,40,40,1",
    2000: "12,20,20,1;12,40,20,1;12,40,40,1;12,80,40,1",
    4000: "12,40,20,1;12,40,40,1;12,80,40,1",
    8000: "12,40,40,1;12,40,20,20,1;12,40,40,20,1;12,40,40,40,1",
    16000: "12,40,40,20,1;12,40,40,40,1;12,80,40,40,1",
}
```
We then ran:
```
python main_norefit.py --sim-func six_func --num-p 6 --var-import-idx 0,1;2,3;4,5 --sim-noise 0.25 --act-func relu --max-iters 40000 --num-train TRAINING_SIZE --layer-sizes LAYER_SIZES[TRAINING_SIZE] --ridge 0.0001,0.000001,0.00000001 --num-inits 2 --cv 5
```

### Eight variable function
We used the following network sizes and ridge penalty paraemters for fitting the eight-variable function example in the paper:
```
LAYER_SIZES = {
    500: "16,20,20,2,1",
    1000: "16,20,20,2,1",
    2000: "16,20,20,2,1",
    4000: "16,20,20,5,1",
    8000: "16,20,20,20,1",
    16000: "16,20,20,20,1"
}
RIDGE_PARAMS = {
    500: "0.01",
    1000: "0.001",
    2000: "0.0001",
    4000: "0.0001",
    8000: "0.00001",
    16000: "0.000001",
}
```
We then ran:
```
python main_norefit.py --sim-func eight_additive --num-p 8 --var-import-group-sizes 4 --sim-noise 2.5 --act-func relu --max-iters 40000 --num-train TRAINING_SIZE --layer-sizes LAYER_SIZES[TRAINING_SIZE] --ridge RIDGE[TRAINING_SIZE] --num-inits 1
```

### ICU Data analysis
The icu data was fit using the augmented MTL network, i.e. using `python main_norefit.py`, with options `---max-iter 10000 --sgd-sample-size 12000 --layer-sizes "116,4,3,2,1"`.
