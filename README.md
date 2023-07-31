# HSTTN
This is a PyTorch implementation of **Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer**. 

If you use this code for your research, please cite our paper:

```
@inproceedings{zhang2023long,
  title={Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer},
  author={Zhang, Yang and Liu, Lingbo and Xiong, Xinyu and Li, Guanbin and Wang, Guoli and Lin, Liang},
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```
### Data Download
- [OneDrive](https://1drv.ms/f/s!Asbmu3Fgg-sHhRDShL--3wtsMcBd)
- [BaiDuYun](https://pan.baidu.com/s/1s8ZUzkCQMaa1xMVyyEzR8Q?pwd=9lpp )
  
### Train with kddcup dataset
* put the kddcup dataset in /data/kddcup or your own path
* python -m main_kddcup.py --data KDDCUP

### Train with engie dataset
* put the engie dataset in /data/engie or your own path
* python -m main_engie.py --data ENGIE

To be continued.
