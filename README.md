# Stego-SandBox

## dataset
[download url](https://cloud.tsinghua.edu.cn/f/534297a363764ad698d8/?dl=1)


## methods
The implementation is based on the repo [GraphSaint](https://github.com/GraphSAINT/GraphSAINT)

### requirements
- python == 3.8.11
- pytorch == 1.8.0
- cython == 0.29.24
- g++ == 5.4.0
- numpy == 1.20.3
- scipy == 1.6.2
- scikit-learn == 0.24.2
- pyyaml == 5.4.1

### before running
We have a cython module which need compilation before training can start. Compile the module by running the following from the root directory:

```
python setup.py build_ext --inplace
```

### baseline

```
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/no_graph.yml --no_graph --repeat_time 1 
```

### using context information

```
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_with_graph.yml --repeat_time 1
```

### relative_data_dir
decompress the downloaded data, then relative_data_dir can be any one of the sub_dir, for example:  
```
relative_data_dir=./data/reddit-ac-1-onlyends_with_isolated_bi_10percent_hop1
```


## For details of the methods and results, please refer to our paper.

