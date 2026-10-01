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

### (1) Environment

To set up the environment, run:

```bash
# python 3.10
conda create -n cvgl python=3.10 -y
conda activate cvgl
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the commands below from the directory containing `requirements.txt` and the training scripts.

### (2) Dataset

**Download.** Prepare the original image datasets for University-1652, SUES-200, and DenseUAV. Our reconstructed 3D point clouds are available at [GeoLink-3D on Hugging Face](https://huggingface.co/datasets/ZhangHY/GeoLink-3D).

**Directory structure.** For University-1652, extract the point clouds into `train/drone_3D/`, alongside the image folders:

```text
University-1652/
└── train/
    ├── drone/
    │   └── 0839/
    │       └── ...
    ├── satellite/
    │   └── 0839/
    │       └── ...
    └── drone_3D/
        ├── 0839_group0/
        │   └── points3D.txt
        ├── 0839_group1/
        │   └── points3D.txt
        ├── 0839_group2/
        │   └── points3D.txt
        └── ...
```

Here, `0839` is the location ID shared with the image folders, and `group0`, `group1`, and `group2` identify its point-cloud groups. Preserve these names when extracting the dataset. Each group contains a COLMAP `points3D.txt` file.

Place the point clouds for each dataset as follows (paths are relative to its dataset root):

| Dataset | Point-cloud directory |
| --- | --- |
| University-1652 | `train/drone_3D/` |
| SUES-200, 300 m | `Training/300/drone300_colmap_output/` |
| DenseUAV | `train/denseUAV_colmap_output/` |

Set `pointcloud_folder_train` to the directory that directly contains the group folders. For the University-1652 layout above, use:

```python
config.pointcloud_folder_train = f'{config.data_folder}/train/drone_3D'
```

**Dataset paths.** Set the roots to your local dataset locations before training:

```bash
export UNIVERSITY_DATA_ROOT=/path/to/University-1652
export SUES_DATA_ROOT=/path/to/SUES-200
export DENSEUAV_DATA_ROOT=/path/to/DenseUAV
```

`SUES_DATA_ROOT` should contain the `Training/` and `Testing/` folders. Configure the image and evaluation paths in the selected training script to match your prepared data; the current University entry point also uses `train/bev_3d_new/` for BEV images.

### (3) Training

The release provides three training entry points. Their default training and evaluation datasets are:

| Training script | Training dataset | Evaluation dataset |
| --- | --- | --- |
| [train_university.py](train_university.py) | University-1652 | SUES-200, 150 m, satellite-to-drone with weather gallery |
| [train_sues.py](train_sues.py) | SUES-200, 300 m | DenseUAV, drone-to-satellite |
| [train_denseuav.py](train_denseuav.py) | DenseUAV | SUES-200, 300 m, drone-to-satellite |

Choose the entry point for your experiment:

```bash
# Train on University-1652
python train_university.py

# Train on SUES-200
python train_sues.py

# Train on DenseUAV
python train_denseuav.py
```

Set `CUDA_VISIBLE_DEVICES` to select your GPUs and `OUTPUT_DIR` to choose where checkpoints and training logs are saved. Batch size, learning rate, and epochs are defined in each script's `Configuration` class.

### (4) Inference

Use [eval_sues.py](eval_sues.py) to extract image descriptors and evaluate retrieval from a saved checkpoint. For example, the University-to-SUES evaluation uses the following query and gallery folders:

```bash
python eval_sues.py \
  --checkpoint /path/to/checkpoint.pth \
  --query-dir "$SUES_DATA_ROOT/Testing/150/query_satellite" \
  --gallery-dir "$SUES_DATA_ROOT/Testing/weather_gallery_drone/150" \
  --batch-size 64 \
  --device cuda
```

Query and gallery images should be organized in subfolders named by location ID. The evaluation reports retrieval recall and average precision. Point clouds are used for data preparation and training; this inference entry point takes images only.

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
