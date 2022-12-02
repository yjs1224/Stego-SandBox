# Stego-SandBox

## data 
[download url](https://cloud.tsinghua.edu.cn/f/534297a363764ad698d8/?dl=1)


## codes
The implementation is based on the repo [GraphSaint](https://github.com/GraphSAINT/GraphSAINT)

### requirements
- python == 3.8.11
- pytoch == 1.8.0
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
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/no_graph.yml --no_graph
```

### using context information

```
python main.py --data_prefix ${relative_data_dir} --train_config ./train_config/gat_192_8_with_graph.yml
```
