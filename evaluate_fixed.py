import os
import json
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, OrderedDict
import time
import tempfile
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision as tv

from models.registry import get_model


def key_transformation(old_key):
    if old_key.endswith('offset') or old_key.endswith('quantized_cdf') or old_key.endswith('cdf_length'):
        new_key = old_key
    elif old_key.startswith('backbone'):
        _, new_key = old_key.split('backbone.')
    else:
        new_key = old_key
    return new_key


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
        start_time = time.perf_counter()
        compressed_obj = model.send(im)
        end_time = time.perf_counter()
        encoder_latency = (end_time - start_time) * 1000
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
            'latency': encoder_latency
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
    checkpoint_root = Path(f'checkpoints/{model_name}_mod')
    save_json_path = Path(f'results_final/{model_name}.json')
    if save_json_path.is_file():
        print(f'==== Warning: {save_json_path} already exists. Will overwrite it! ====')
    else:
        print(f'Will save results to {save_json_path} ...')

    checkpoint_paths = list(checkpoint_root.rglob('*.pt'))
    checkpoint_paths.sort()
    print(f'Find {len(checkpoint_paths)} checkpoints in {checkpoint_root}. Evaluating them ...')

    results_of_all_models = defaultdict(list)
    for ckptpath in checkpoint_paths:
        model = get_model(model_name)(teacher=False)

        checkpoint = torch.load(ckptpath)
        new_state_dict = OrderedDict()

        for key, value in checkpoint['model'].items():
            new_key = key_transformation(key)
            new_state_dict[new_key] = value

        checkpoint['model'] = new_state_dict

        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # print("##############################################")
        # for key, value in checkpoint['model'].items():
        #     print('key: ', key, '\t', 'value: ', checkpoint['model'][key].shape)    # value[key].shape

        model.load_state_dict(checkpoint['model'], strict=False)

        model = model.to(device=device)
        model.eval()
        model.update()

        results = evaluate_model(model, args)
        print(results)
        results['checkpoint'] = str(ckptpath.relative_to(checkpoint_root))
        for k,v in results.items():
            results_of_all_models[k].append(v)

        with open(save_json_path, 'w') as f:
            json.dump(results_of_all_models, fp=f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, nargs='+',
        default=['ours_n8'])                                                                     # 'ours_n0', 'ours_n4', 'ours_n8'
    parser.add_argument('-d', '--data_root',  type=str, default='val_1000')
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    parser.add_argument('-w', '--workers',    type=int, default=1)
    args = parser.parse_args()

    # torch.set_num_threads(1)
    for model_name in args.models:
        evaluate_all_bit_rate(model_name, args)
        print()


if __name__ == '__main__':
    main()
