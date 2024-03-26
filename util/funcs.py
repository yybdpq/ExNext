import os
import random
import numpy as np
import json
import torch
import logging
from tqdm import tqdm
import os.path as osp
from torch import nn

def logset(args):
    """
    Write logs to checkpoint and console
    """
    if args.do_train:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'train.log')
    else:
        log_file = osp.join(args.log_path or args.init_checkpoint, 'test.log')

    # Remove all handlers associated with the root logger object
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w+'
    )

def pathget():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("ExNext")
    dirname = "/".join(dirname_split[:index + 1])
    # dirname = "your dirname"
    return dirname

def seedset(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

   
def recall(lab, prd, k):
    return torch.sum(torch.sum(lab == prd[:, :k], dim=1)) / lab.shape[0]

def ndcg(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    dcg = 1 / torch.log2(exist_pos.float() + 1)
    return torch.sum(dcg) / lab.shape[0]

def map_k(lab, prd, k):
    exist_pos = torch.nonzero(prd[:, :k] == lab, as_tuple=False)[:, 1] + 1
    map_tmp = 1 / exist_pos
    return torch.sum(map_tmp) / lab.shape[0]

def mrr(lab, prd):
    exist_pos = torch.nonzero(prd == lab, as_tuple=False)[:, 1] + 1
    mrr_tmp = 1 / exist_pos
    return torch.sum(mrr_tmp) / lab.shape[0]

def save_model(model, optimizer, save_variable_list, run_args, argparse_dict):
    """
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    """
    with open(osp.join(run_args.log_path, 'config.json'), 'w') as fjson:
        for key, value in argparse_dict.items():
            if isinstance(value, torch.Tensor):
                argparse_dict[key] = value.numpy().tolist()
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        osp.join(run_args.save_path, 'checkpoint.pt')
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_step(model, data, ks=(1, 5, 10, 20)):
    model.eval()
    loss_list = []
    pred_list = []
    label_list = []
    with torch.no_grad():
        for row in tqdm(data):
            split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)

            input_data = {
                'x': row.x,
                'edge_index': row.adjs_t,
                'edge_attr': row.edge_attrs,
                'split_index': split_index,
                'delta_ts': row.edge_delta_ts,
                'delta_ss': row.edge_delta_ss,
                'edge_type': row.edge_types
            }
            out, _, _= model(input_data, label=row.y[:, 0], mode='test')
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(out, row.y[:, 0].long())
                      
            loss_list.append(loss.cpu().detach().numpy().tolist())
            ranking = torch.sort(out, descending=True)[1]
            pred_list.append(ranking.cpu().detach())
            label_list.append(row.y[:, :1].cpu())
    pred_ = torch.cat(pred_list, dim=0)
    label_ = torch.cat(label_list, dim=0)
    recalls, NDCGs, MAPs = {}, {}, {}
    logging.info(f"[Evaluating] Average loss: {np.mean(loss_list)}")
    for k_ in ks:
        recalls[k_] = recall(label_, pred_, k_).cpu().detach().numpy().tolist()
        NDCGs[k_] = ndcg(label_, pred_, k_).cpu().detach().numpy().tolist()
        MAPs[k_] = map_k(label_, pred_, k_).cpu().detach().numpy().tolist()
        logging.info(f"[Evaluating] Recall@{k_} : {recalls[k_]},\tNDCG@{k_} : {NDCGs[k_]},\tMAP@{k_} : {MAPs[k_]}")
    mrr_res = mrr(label_, pred_).cpu().detach().numpy().tolist()
    logging.info(f"[Evaluating] MRR : {mrr_res}")
    return recalls, NDCGs, MAPs, mrr_res, np.mean(loss_list)
