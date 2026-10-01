<p align="center">

  <h3 align="center">GeoLink: A 3D-aware Framework to Improve Generalization for Cross-view Geo-localization</h3>

</p>

<h5 align="center">
  If you like our project, please give us a star ⭐️ for the continuous updates.
</h5>

<p align="center">
  <a href="https://hrt00.github.io/hyzhang.github.io/" target="_blank">Hongyang Zhang<sup>1,*</sup></a>,&nbsp;
  Yinhao Liu<sup>2,*</sup>,&nbsp;
  Haitao Zhang<sup>2</sup>,&nbsp;
  Zhongyi Wen<sup>3</sup>,&nbsp;
  Zhenyu Kuang<sup>4</sup>,&nbsp;
  Shuxian Liang<sup>5,†</sup>;
  Xian-Sheng Hua<sup>5,†</sup>
</p>

<p align="center">
  <sup>1</sup>CUHK-Shenzhen; 
  <sup>2</sup>Xiamen Univeristy; 
  <sup>3</sup>UESTC;
  <sup>4</sup>Foshan Univeristy;
  <sup>5</sup>Tongji Univeristy;
</p>

<p align="center">
  <sup>*</sup>Equal contribution. <sup>†</sup>Corresponding author.
</p>

## <a id="news"></a> 🔥 News
- 🎉[October 1, 2026]: The source code and [GeoLink-3D dataset](https://huggingface.co/datasets/ZhangHY/GeoLink-3D) have been released.
- 🎉[July 10, 2026]: GeoLink is accepted by ACMMM'26. See you in Rio de Janeiro, Brazil!
- 🚩[April 13, 2026]: The preprint version has been released in [Paper Link](https://arxiv.org/pdf/2604.13183).  

## 🚀 How to Use
(1) Environment

To set up the environment, run:
```
# python 3.10
conda create -n cvgl python=3.10 -y
conda activate cvgl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

(2) Dataset

The GeoLink-3D point-cloud dataset is available on
[Hugging Face](https://huggingface.co/datasets/ZhangHY/GeoLink-3D). Download and
place the point clouds under the corresponding dataset's training directory.
For the University-1652 layout, the expected structure is:

```
University-1652/
└── train/
    ├── drone/
    ├── satellite/
    └── drone_3D/
        ├── 0839_group0/
        │   └── points3D.txt
        ├── 0839_group1/
        │   └── points3D.txt
        └── 0839_group2/
            └── points3D.txt
```

In other words, put the `*_group0`, `*_group1`, and `*_group2` sample folders
directly inside `train/drone_3D/`; do not add another nested `drone_3D/`
directory. Each sample folder contains its COLMAP `points3D.txt`. The dataset
loader reads XYZ and RGB from that file. The University example at
`/media/lscsc/nas2/hongyang/dataset/University-1652/train/drone_3D/` follows
this layout. For SUES and DenseUAV, use the corresponding point-cloud training
folder configured in their training scripts, with one sample folder containing
`points3D.txt` per sample.

(3) Training and Inference

The source code includes the following training and inference scripts:

- Training: `train_university.py`, `train_sues.py`, and `train_denseuav.py`.
- Inference: `eval_sues.py`.

## Cite
If you find our paper and code useful in your research, please consider citing our work 📝:
```bibtex
@article{zhang2026geolink,
  title={GeoLink: A 3D-Aware Framework Towards Better Generalization in Cross-View Geo-Localization},
  author={Zhang, Hongyang and Liu, Yinhao and Zhang, Haitao and Wen, Zhongyi and Kuang, Zhenyu and Liang, Shuxian and Hua, Xiansheng},
  journal={arXiv preprint arXiv:2604.13183},
  year={2026}
}
```
## Contact
If you have any questions about this project, please feel free to contact hongyangzhang1@link.cuhk.edu.cn.
