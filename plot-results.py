import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
# Importing libraries
import numpy as np
import math
import re

from delta_accuracy import bd_accuracy


default_font = {
    'size': 18,
}

def plot(stat, ax=None):
    x = stat['bpp']
    y = [f*100 for f in stat['top1']]
    label = stat['name']
    module = ax or plt
    p = module.plot(x, y, label=label,
        marker='.', markersize=10, linewidth=1.6,
    )
    return p

def plot_del(stat, ax=None):
    x = stat['bpp']
    y = [f*100 for f in stat['top1']]
    label = stat['name']
    module = ax or plt
    p = module.plot(x, y, label=label,
        marker='.', markersize=10, linewidth=1.6,
    )
    return p


def main():
    fig1, ax = plt.subplots(figsize=(8,8))

    results_dir = Path('results')
    # results_dir = [os.path.join(results_dir, 'ours_condautoencoder_1.json'), os.path.join(results_dir, 'ours_condautoencoder_2.json'), os.path.join(results_dir, 'ours_condautoencoder_3.json'), os.path.join(results_dir, 'ours_n4.json')]
    all_methods_results = []
    # model_list = ['results_cond_final/ours_condautoencoder_42_Config.3.json', 'results_cond_final/ours_condautoencoder_67_Config.4.json']
    for fpath in results_dir.rglob('*.json'):
    # for fpath in results_dir:
        print(fpath)
        # if str(fpath) in model_list:
            # print(str(fpath))
        with open(fpath, 'r') as f:
            results = json.load(f)
        # results['name'] = fpath.stem
        results['name'] = re.split('_+', fpath.stem)[-1]
        if results['name'] == 'conv1x':
            reference_results = results
        print(results['name'])
        all_methods_results.append(results)
    all_methods_results = sorted(all_methods_results, key=lambda d: d['name']) 

    # print(all_methods_results[0])
    for results in all_methods_results:
        # print(results)
        # print(type(results))
        del_accuracy = bd_accuracy(reference_results, results, visualize=False)
        # del_accuracy = bd_accuracy(reference_results, results, visualize=False)
        print(f"Model {results['name']}: {del_accuracy}")
        # print(type(results['latency']))
        # avg_latency = results['latency']
        plot(results, ax=ax)


    plt.title('Rate-accuracy trade-off on ImageNet', fontdict=default_font)
    plt.grid(True, alpha=0.32)
    plt.legend(loc='lower right', prop={'size': 18})
    plt.xlabel('Bits per pixel (bpp)', fontdict=default_font)
    plt.ylabel('Top-1 acc. (%)', fontdict=default_font)
    x_ticks = [(i) / 10 for i in range(22)]
    plt.xticks(x_ticks)
    y_ticks = [i for i in range(54, 78, 2)]
    plt.yticks(y_ticks)
    plt.xlim(min(x_ticks), max(x_ticks))
    plt.ylim(min(y_ticks), max(y_ticks))
    plt.tight_layout()
    plt.savefig("images/RAC_exp.jpg")
    plt.show(block=True)







    # # Using Numpy to create an array X
    # X = np.arange(0, math.pi*2, 0.05)
    
    # # Assign variables to the y axis part of the curve
    # y = np.sin(X)
    # z = np.cos(X)
    
    # # Plotting both the curves simultaneously
    # plt.plot(X, y, color='r', label='sin')
    # plt.plot(X, z, color='g', label='cos')
    
    # # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Angle")
    # plt.ylabel("Magnitude")
    # plt.title("Sine and Cosine functions")
    
    # # Adding legend, which helps us recognize the curve according to it's color
    # plt.legend()
    
    # # To load the display window
    # plt.show()

    

if __name__ == '__main__':
    main()
    


