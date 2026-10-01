import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch.nn.functional as F
from .loss.loss import relational_distillation_3d_loss


def cal_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss

def compute_mmd(features_a, features_b, sigma=1.0):
    """
    计算两组特征之间的 Maximum Mean Discrepancy (MMD)
    features_a, features_b: [B, D] 张量
    """
    def rbf_kernel(x, y, sigma):
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * sigma ** 2))
    
    K_xx = rbf_kernel(features_a, features_a, sigma)
    K_yy = rbf_kernel(features_b, features_b, sigma)
    K_xy = rbf_kernel(features_a, features_b, sigma)
    
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

def train(train_config, model, club_estimator, dataloader, loss_function, optimizer, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    mmds = AverageMeter()
    losses_sim_g2s = AverageMeter()
    losses_sim_pc2g = AverageMeter()
    losses_sim_pc2s = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
          
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    # for loop over one epoch
    for query, reference, pc, ids, labels in bar:
        
        if scaler:
            with autocast():
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
                pc = pc.to(train_config.device)
                labels = labels.to(train_config.device)
            
                # Forward pass
                features1, features2, features3 = model(query, reference, pc)
                loss_uav = club_estimator(features1, features3)
                loss_sat = club_estimator(features2, features3)
                club_loss = loss_uav + loss_sat
                club_loss = club_loss.mean()

                loss_sim_self1 = loss_function['InfoNCE'](features1, features1, model.logit_scale.exp())
                loss_sim_self2 = loss_function['InfoNCE'](features2, features2, model.logit_scale.exp())
                loss_sim_self3 = loss_function['InfoNCE'](features3, features3, model.logit_scale.exp())
                    
                loss_sim_g2s = loss_function['InfoNCE'](features1, features2, model.logit_scale.exp())
                loss_sim_pc2g = loss_function['InfoNCE'](features3, features1, model.logit_scale.exp())
                loss_sim_pc2s = loss_function['InfoNCE'](features3, features2, model.logit_scale.exp())
                    
                loss_distill = relational_distillation_3d_loss(features1, features3) + relational_distillation_3d_loss(features2, features3)
                mmd_val = compute_mmd(features1, features2)
                   
                loss = club_loss + loss_distill + (loss_sim_g2s + loss_sim_pc2g + loss_sim_pc2s) + 4.0*(loss_sim_self1 + loss_sim_self2 + loss_sim_self3)
                losses.update(loss.item())
                mmds.update(mmd_val.item())
                losses_sim_g2s.update(loss_sim_g2s.item())
                losses_sim_pc2g.update(loss_sim_pc2g.item())
                losses_sim_pc2s.update(loss_sim_pc2s.item())
                  
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler == "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()

        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg, losses_sim_g2s.avg, losses_sim_pc2g.avg, losses_sim_pc2s.avg, mmds.avg


def predict(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

def predict_eval(train_config, model, dataloader):
    model.eval()
    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    # output_shape = model(dummy_input)[0].shape[1:]
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size)
    imgs = torch.zeros((len(dataloader.dataset), 3, 448, 448), dtype=torch.float32, device=train_config.device)
    img_features = torch.zeros((len(dataloader.dataset), *output_shape), dtype=torch.float32, device=train_config.device)
    ids = torch.zeros(len(dataloader.dataset), dtype=torch.long, device=train_config.device)
    qnames = []
    with torch.no_grad(), autocast():
        for i, (img, ids_current, img_pth) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            # img_feature = model(img)[0]
            img_feature = model(img)

            # normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current
            imgs[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img
            qnames.extend(list(img_pth))

    return img_features, ids, imgs, qnames

def predict_vigor(train_config, model, dataloader):
    model.eval()

    # Get output shape from a dummy input
    dummy_input = torch.randn(1, *dataloader.dataset[0][0].shape, device=train_config.device)
    output_shape = model(dummy_input).shape[1:]

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, *output_shape), dtype=torch.float32, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids
    ids = torch.zeros((total_samples, 4), dtype=torch.long, device=train_config.device)  # 修改为二维张量以匹配ids_current的形状

    with torch.no_grad(), autocast():
        for i, (img, ids_current) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature = model(img)

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids

# def predict(train_config, model, dataloader):
#     model.eval()
#
#     # wait before starting progress bar
#     time.sleep(0.1)
#
#     if train_config.verbose:
#         bar = tqdm(dataloader, total=len(dataloader))
#     else:
#         bar = dataloader
#
#     img_features_list = []
#
#     ids_list = []
#     with torch.no_grad():
#
#         for img, ids in bar:
#
#             ids_list.append(ids)
#
#             with autocast():
#
#                 img = img.to(train_config.device)
#                 img_feature = model(img)
#
#                 # normalize is calculated in fp32
#                 if train_config.normalize_features:
#                     img_feature = F.normalize(img_feature, dim=-1)
#
#             # save features in fp32 for sim calculation
#             img_features_list.append(img_feature.to(torch.float32))
#
#         # keep Features on GPU
#         img_features = torch.cat(img_features_list, dim=0)
#         ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
#
#     if train_config.verbose:
#         bar.close()
#
#     return img_features, ids_list