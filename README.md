# Molecular Generation with GDB-13 dataset

# Setup
```
cd ..
mkdir GDB_Generation_project
mv Molecular_Generation_with_GDB13 GDB_Generation_project/
```

Setup a conda envirement:
```
cd Molecular_Generation_with_GDB13
conda env create --file=environment.yml
conda activate gdb
```

setup metaseq:
```
## Install Megatron 
cd ..
git clone --branch fairseq_v3 https://github.com/ngoyal2707/Megatron-LM.git
cd Megatron-LM
pip install six regex
pip install -e .

## Install fairscale 
cd ..
git clone https://github.com/facebookresearch/fairscale.git
cd fairscale
git checkout fixing_memory_issues_with_keeping_overlap_may24
pip install -e .

## Install metaseq 
cd..
git clone https://github.com/YerevaNN/metaseq.git
cd metaseq
git checkout scaling_racm3
pip install -e .


# turn on pre-commit hooks
pre-commit install
```

1. Download data_subsets from [here](https://disk.yandex.ru/d/Qmqhz2nXeYCR3Q) 