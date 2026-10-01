import torch
import numpy as np
from tqdm import tqdm
import os
import gc
from cvgl_base.trainer import predict, predict3d, predict_tta
import torch.nn.functional as F


def evaluate_tta(config,
                 model,
                 query_loader,
                 gallery_loader,
                 ranks=[1, 5, 10],
                 step_size=1000,
                 cleanup=True,
                 tta_transforms=None):

    def extract_features_tta(loader):

        if tta_transforms is None:
            feats, ids = predict_tta(config, model, loader, transform=None)
            return feats.cpu(), ids.cpu()

        feature_chunks = []
        ids_ref = None

        transforms_to_run = tta_transforms if isinstance(tta_transforms, list) else [tta_transforms]

        for transform in transforms_to_run:

            feats, ids = predict_tta(config, model, loader, transform=transform)

            feats = feats.cpu()

            feature_chunks.append(feats)

            if ids_ref is None:
                ids_ref = ids.cpu()

            del feats
            torch.cuda.empty_cache()

        stacked = torch.stack(feature_chunks, dim=0)

        averaged = torch.mean(stacked, dim=0)

        averaged = F.normalize(averaged, dim=1)

        del stacked
        gc.collect()

        return averaged, ids_ref

    print("Extract Features:")

    img_features_query, ids_query = extract_features_tta(query_loader)
    img_features_gallery, ids_gallery = extract_features_tta(gallery_loader)

    gl = ids_gallery.numpy()
    ql = ids_query.numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    if cleanup:
        del img_features_query
        del img_features_gallery
        del ids_query
        del ids_gallery
        gc.collect()
        torch.cuda.empty_cache()

    return CMC[0]

def evaluate(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)
   
    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]

def evaluate_logs(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True,
             output_error_file="error_pairs.txt"):
    print("Extract Features:")
    # Get list of feature/ID tensors (chunked)
    features_query_chunks, ids_query_chunks = predict(config, model, query_loader)
    features_gallery_chunks, ids_gallery_chunks = predict(config, model, gallery_loader)

    # Concatenate all chunks into single tensor
    # img_features_query = torch.cat(features_query_chunks, dim=0)  # [Nq, D]
    # ids_query = torch.cat(ids_query_chunks, dim=0)               # [Nq]
    img_features_query = features_query_chunks
    ids_query = ids_query_chunks

    # img_features_gallery = torch.cat(features_gallery_chunks, dim=0)  # [Ng, D]
    # ids_gallery = torch.cat(ids_gallery_chunks, dim=0)               # [Ng]
    img_features_gallery = features_gallery_chunks
    ids_gallery = ids_gallery_chunks

    # Get file paths (assuming dataset has get_paths())
    query_paths = query_loader.dataset.get_paths()
    gallery_paths = gallery_loader.dataset.get_paths()

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0

    # Open error file for writing
    os.makedirs(os.path.dirname(output_error_file) if os.path.dirname(output_error_file) else '.', exist_ok=True)
    with open(output_error_file, 'w') as f_err:
        f_err.write("Query_Path\tGallery_Top1_Path\tCorrect_Gallery_Paths\n")

        for i in tqdm(range(len(ids_query))):
            q_feat = img_features_query[i]
            q_id = ql[i]

            # Compute distances
            dists = torch.linalg.norm(img_features_gallery - q_feat, dim=1)
            indices = torch.argsort(dists)

            # Top-1 gallery index
            top1_idx = indices[0].item()
            top1_id = gl[top1_idx]
            top1_path = gallery_paths[top1_idx]

            # Check if correct
            is_correct = (top1_id == q_id)

            # Compute AP and CMC (reuse your existing eval_query if it returns CMC_tmp)
            ap_tmp, CMC_tmp = eval_query(q_feat, q_id, img_features_gallery, gl)
            if CMC_tmp[0] == -1:
                continue
            CMC += CMC_tmp
            ap += ap_tmp

            # If top-1 is wrong, record the pair
            if not is_correct:
                # Find all correct gallery paths for this query (for reference)
                correct_gallery_paths = [gallery_paths[j] for j, gid in enumerate(gl) if gid == q_id]
                correct_str = ";".join(correct_gallery_paths)
                f_err.write(f"{query_paths[i]}\t{top1_path}\t{correct_str}\n")

    AP = ap / len(ids_query) * 100
    CMC = CMC.float() / len(ids_query)
    top1_percent = round(len(ids_gallery) * 0.01)

    string = []
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))
    string.append('Recall@top1%: {:.4f}'.format(CMC[top1_percent] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()

    return CMC[0]

def evaluate3d(config,
             model,
             query_loader,
             gallery_loader,
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    print("Extract Features:")
    img_features_query, pc_features_query, ids_query = predict3d(config, model, query_loader)
    img_features_gallery, ids_gallery = predict(config, model, gallery_loader)

    gl = ids_gallery.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_gallery)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query3d(img_features_query[i], pc_features_query[i], ql[i], img_features_gallery, gl)
        if CMC_tmp[0] == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    AP = ap / len(ids_query) * 100

    CMC = CMC.float()
    CMC = CMC / len(ids_query)  # average CMC

    # top 1%
    top1 = round(len(ids_gallery) * 0.01)

    string = []

    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i - 1] * 100))

    string.append('Recall@top1: {:.4f}'.format(CMC[top1] * 100))
    string.append('AP: {:.4f}'.format(AP))

    print(' - '.join(string))

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_gallery, ids_gallery
        gc.collect()
        # torch.cuda.empty_cache()

    return CMC[0]

def eval_query3d(qf, qf3d, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score3d = gf @ qf3d.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()
    score3d = score3d.squeeze().cpu().numpy()

    score += 0.1*score3d

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp

def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)

    score = score.squeeze().cpu().numpy()

    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl == -1)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
