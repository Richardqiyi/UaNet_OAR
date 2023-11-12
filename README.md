# UaNet_OAR
For original repo, see [UaNet](https://github.com/uci-cbcl/UaNet#clinically-applicable-deep-learning-framework-for-organs-at-risk-delineation-in-ct-images)

#### Dataset

See [HaN-Seg](https://doi.org/10.1002/mp.16197)

#### Environment

Python = 3.7, Pytorch = 1.8.0, Torchvision = 1.9.0, CUDA = 11.1 

#### GPU

NVIDIA GeForce RTX 3090(24 GB memory)

#### Docker

```
docker pull qiyi007/uanet:2.0
```

```
conda activate UaNet
```
#### Install dependencies
Please make sure your working directory is src/

```
cd src
```

```
cd build/box
python setup.py install
```

#### File Directory
```

|-- UaNet_OAR
    |-- data
        |-- raw
            |-- case_01
                |-- img.nrrd (your image)
                |-- structures
                    |-- <organ at risk>.nrrd
                    |-- ......
        |-- preprocessed

```
                



