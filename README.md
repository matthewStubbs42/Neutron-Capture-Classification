# Neutron Capture Classification

## Description

Classification of neutron vs background events in a Water Cherenkov (WC) detector using Graph Neural Networks (GCN, AGNN, SG), standard NN (MLP) and XGBoost

## Table of Contents

### 1. [Installation](#installation)
### 2. [Usage](#usage)

## Installation <a id="installation"></a>

To download the repository use :
'git clone https://github.com/matthewStubbs42/Neutron-Capture-Classification.git'

## Usage

# Run Examples:

```
# GCN, distance weighted, fully connected, graph network
python main.py --dw=True --model=GCN --epochs=100 --lr=0.003
```

```
# MLP model
python main_agg.py --model=MLP --epochs=1000 --lr=0.005 --lrdecay=0.98
```
