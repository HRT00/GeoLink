"""Evaluate a train_university.py checkpoint on the configured SUES-200 split."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cvgl_base.dataset.sues import SUESDatasetEval, get_transforms
from cvgl_base.evaluate.university import evaluate
from cvgl_base.model_club import TimmModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path,
                        help="SUES satellite query directory (for example Testing/150/query_satellite)")
    parser.add_argument("--gallery-dir", required=True, type=Path,
                        help="SUES drone gallery directory (for example Testing/weather_gallery_drone/150)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    img_size = (448, 448)
    model = TimmModel(
        args=args,
        model_name="dinov2_vitb14_MixVPR",
        pretrained=True,
        img_size=img_size[0],
        backbone_arch="dinov2_vitb14",
        agg_arch="MixVPR",
        agg_config={"in_channels": 768, "in_h": 32, "in_w": 32,
                    "out_channels": 1024, "mix_depth": 2,
                    "mlp_ratio": 1, "out_rows": 4},
        layer1=7,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(args.device).eval()

    val_transform, _, _ = get_transforms(img_size, **model.get_config())
    query_set = SUESDatasetEval(str(args.query_dir), "query", transforms=val_transform)
    gallery_set = SUESDatasetEval(
        str(args.gallery_dir), "gallery", transforms=val_transform,
        sample_ids=query_set.get_sample_ids(),
    )
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

    print(f"Query images: {len(query_set)}; gallery images: {len(gallery_set)}")
    evaluate(config=args, model=model, query_loader=query_loader,
             gallery_loader=gallery_loader, ranks=[1, 5, 10], cleanup=True)


if __name__ == "__main__":
    main()
