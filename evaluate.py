import os
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import tempfile
import argparse
import time
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from models.registry import get_model


def get_object_size(obj, unit='bits'):
    assert unit == 'bits'
    with tempfile.TemporaryFile() as fp:
        pickle.dump(obj, fp)
        num_bits = os.fstat(fp.fileno()).st_size * 8
    return num_bits


def evaluate_model(model, args):
    test_transform = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    testset = tv.datasets.ImageFolder(root=args.data_root, transform=test_transform)
    testloader = DataLoader(
        testset, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, drop_last=False
    )

    device = next(model.parameters()).device

    pbar = tqdm(testloader)
    stats_accumulate = defaultdict(float)
    for im, labels in pbar:
        im = im.to(device=device)
        nB, imC, imH, imW = im.shape

        # sender side: compress and save
        # start_time = time.time()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,], with_flops=True) as p:
            compressed_obj, encoder_latency = model.send(im)
        print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        # end_time = time.time()
        # latency = (end_time - start_time) * 1000
        # sender side: compute bpp
        num_bits = get_object_size(compressed_obj)
        bpp = num_bits / float(imH * imW)

        # receiver side: class prediction
        p = model.receive(compressed_obj)
        # reduce to top5
        _, top5_idx = torch.topk(p, k=5, dim=1, largest=True)
        # compute top1 and top5 matches (true positives)
        correct5 = torch.eq(labels.view(nB, 1), top5_idx.cpu())

        stats_accumulate['count'] += float(nB)
        stats = {
            'top1': correct5[:, 0].float().sum().item(),
            'top5': correct5.any(dim=1).float().sum().item(),
            'bpp': bpp,
            # 'latency': latency,
            'network_latency': encoder_latency[0],
            'encoder_latency': encoder_latency[1],
            'compression_time': encoder_latency[2],
            'total_latency': encoder_latency[3]
            # 'latency_conv1': encoder_latency[0],
            # 'latency_conv2': encoder_latency[1],
            # 'latency_conv3': encoder_latency[2],
            # 'latency_emb': encoder_latency[3],
            # 'latency_compression': encoder_latency[4],
            # 'latency_total': encoder_latency[5],
        }
        for k, v in stats.items():
            stats_accumulate[k] += float(v)

        # logging
        _cnt = stats_accumulate['count']
        msg = ', '.join([f'{k}={v/_cnt:.4g}' for k,v in stats_accumulate.items() if (k != 'count')])
        pbar.set_description(msg)
    pbar.close()

    # compute total statistics and return
    total_count = stats_accumulate.pop('count')
    results = {k: v/total_count for k,v in stats_accumulate.items()}
    return results


def evaluate_all_bit_rate(model_name, args):
    device = torch.device('cuda')
    checkpt_name = 'ours_condautoencoder_44_conv1'
    checkpoint_root = Path(f'checkpoints/{checkpt_name}')
    # checkpoint_root = Path(f'checkpoints/{model_name}')
    os.makedirs("results", exist_ok=True)
    # save_json_path = Path(f'results/{model_name}.json')
    save_json_path = Path(f'results/{checkpt_name}_FLOPs.json')
    if save_json_path.is_file():
        print(f'==== Warning: {save_json_path} already exists. Will overwrite it! ====')
    else:
        print(f'Will save results to {save_json_path} ...')

    # checkpoint_path = checkpoint_root.rglob('*.pt')
    checkpoint_path = os.path.join(checkpoint_root, 'last_ema.pt')
    print(checkpoint_path)
    # checkpoint_paths = list(checkpoint_root.rglob('*.pt'))
    # checkpoint_paths.sort()
    # print(f'Find {len(checkpoint_paths)} checkpoints in {checkpoint_root}. Evaluating them ...')

    results_of_all_models = defaultdict(list)
    # log_lmb_min, log_lmb_max = math.log(0.0000001), math.log(10.24)
    # bpp_lmb_list = torch.exp(torch.tensor(np.linspace(log_lmb_min, log_lmb_max, num=10))).tolist()
    # print(bpp_lmb_list)
    bpp_lmb_list = [1e-7, 0.0001, 0.01, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 8.12, 10.24]                 # conv1
    # bpp_lmb_list = [1.28]
    print(bpp_lmb_list)
    # bpp_lmb_list = [0.00000000001]
    ################## fix the codes below
    # for ckptpath in checkpoint_paths:
    for bpp_lmb in bpp_lmb_list:
        model = get_model(model_name)(bpp_lmb=bpp_lmb)       #(teacher=False)
        # checkpoint = torch.load(ckptpath)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])

        model = model.to(device=device)
        model.eval()
        model.update()

        results = evaluate_model(model, args)
        print(results)
        # results['checkpoint'] = str(ckptpath.relative_to(checkpoint_root))
        results['lambda'] = str(bpp_lmb)
        for k,v in results.items():
            results_of_all_models[k].append(v)

        with open(save_json_path, 'w') as f:
            json.dump(results_of_all_models, fp=f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, nargs='+',
        default=['ours_condautoencoder'])                 # ours_autoencoder   'ours_n0', 'ours_n4', 'ours_n8', 'ours_n0_enc', 'ours_n4_enc', 'ours_n8_enc'
    parser.add_argument('-d', '--data_root',  type=str, default='/data/home/hossai34/datasets/imagenet/val')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-w', '--workers',    type=int, default=1)
    # parser.add_argument('--bpp_lmb',  type=float, default=1.28)
    args = parser.parse_args()

    # torch.set_num_threads(1)
    for model_name in args.models:
        evaluate_all_bit_rate(model_name, args)
        print()


if __name__ == '__main__':
    main()

# import numpy as np
# import matplotlib.pyplot as plt
# ​
# ​
# def _poly_of_log_integral(poly, xmin, xmax, N=1000):
#     degree = poly.shape[0] - 1
#     x = np.linspace(xmin, xmax, num=N)
#     y = sum([poly[i] * np.log(x)**(degree-i) for i in range(degree + 1)])
#     assert y.shape == (N,)
#     # approximated area under the curve (ie, definite integral)
#     auc = np.sum(y) * float(1/N)
#     return auc, (x, y)
# ​
# ​
# def bd_accuracy(stats1, stats2, visualize=False):
#     bpp1 = stats1['bpp']
#     acc1 = stats1['acc']
#     name1 = stats1.get('name', 'input 1')
#     bpp2 = stats2['bpp']
#     acc2 = stats2['acc']
#     name2 = stats2.get('name', 'input 2')
# ​
#     logb1 = np.log(bpp1)
#     logb2 = np.log(bpp2)
# ​
#     degree = 4
#     poly1 = np.polyfit(logb1, acc1, deg=degree)
#     poly2 = np.polyfit(logb2, acc2, deg=degree)
# ​
#     bppmin = max(min(bpp1), min(bpp2))
#     bppmax = min(max(bpp1), max(bpp2))
#     auc1, (_x1,_y1) = _poly_of_log_integral(poly1, xmin=bppmin, xmax=bppmax, N=10000)
#     auc2, (_x2,_y2) = _poly_of_log_integral(poly2, xmin=bppmin, xmax=bppmax, N=10000)
# ​
#     bd_acc = (auc2 - auc1) / (bppmax - bppmin)
# ​
#     if visualize:
#         x = np.linspace(np.log(bppmin), np.log(bppmax), num=200)
#         fig1, ax = plt.subplots(1, 2, figsize=(12,5))
#         plt.setp(ax, ylim=(min(min(acc1), min(acc2))-1, max(max(acc1), max(acc2))+1))
#         # left figure: log space
#         l1 = ax[0].plot(logb1, acc1, label=f'data - {name1}', marker='.', markersize=12, linestyle='none')
#         ax[0].plot(x, np.polyval(poly1, x), label=f'polyfit - {name1}', color=l1[0].get_color())
#         l2 = ax[0].plot(logb2, acc2, label=f'data - {name2}', marker='.', markersize=12, linestyle='none')
#         ax[0].plot(x, np.polyval(poly2, x), label=f'polyfit - {name2}', color=l2[0].get_color())
#         ax[0].set_xlabel('$\log R$')
#         ax[0].set_ylabel('$A$')
#         ax[0].set_title('Polynomial fitting in the log rate space', fontdict = {'fontsize' : 14})
#         ax[0].legend(loc='lower right')
#         # right figure: normal space
#         l1 = ax[1].plot(bpp1,  acc1, label=f'data - {name1}', marker='.', markersize=12, linestyle='none')
#         ax[1].plot(_x1, _y1, label=f'polyfit - {name1}', color=l1[0].get_color())
#         l2 = ax[1].plot(bpp2,  acc2, label=f'data - {name2}', marker='.', markersize=12, linestyle='none')
#         ax[1].plot(_x2, _y2, label=f'polyfit - {name2}', color=l2[0].get_color())
#         ax[1].set_xlabel('$R$')
#         ax[1].set_ylabel('$A$')
#         ax[1].set_title('Mapping back to the normal space', fontdict = {'fontsize' : 14})
#         ax[1].legend(loc='lower right')
#     else:
#         pass
# ​
#     return bd_acc
