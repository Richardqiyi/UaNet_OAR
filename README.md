# UaNet_OAR
For original repo, see [UaNet](https://github.com/uci-cbcl/UaNet#clinically-applicable-deep-learning-framework-for-organs-at-risk-delineation-in-ct-images)

#### Dataset

See [HaN-Seg](https://doi.org/10.1002/mp.16197)

#### Environment

Python = 3.7, Pytorch = 1.8.0, Torchvision = 1.9.0, CUDA = 11.1 

#### GPU

1 NVIDIA GeForce RTX 3090 24G

#### Docker

```
docker pull qiyi007/uanet:2.0
```

```
conda activate UaNet
```
#### Install dependencies
Please make sure your working directory is src

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
                    |-- <name of the first mask (no index needed)>.nrrd
                    |-- <name of the second mask (no index needed)>.nrrd
                    |-- ......
        |-- preprocessed
    |-- src

```
                
#### Preprocess

1. Use `scale.py` to scale up the orginal images (some original shape is (D, 512, 512)) to (D, 1024, 1024)
2. Use `utils/preprocess.py` to preprocess the converted data.

#### Train
Change training configuration and data configuration in `config.py`, especially the path to your preprocessed data.

You can change network configuration in `net/config.py`, then run training script:

```
python train.py
```

#### Test

```
python mytest.py test --weight $PATH_TO_WEIGHT --nrrd-path $DICOM_PATH --out-dir $OUTPUT_DIR
```

Can only process one image at a time.


