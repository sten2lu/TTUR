from __future__ import absolute_import, division, print_function
import numpy as np
import os

import fid


def check_paths(paths):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" %p)

def get_activation(path):
    if path.endswith('.npz'):
        f = np.load(path)
        try:
            act = f['act']
            return act
        except:
            raise RuntimeError('Invalid .npz it needs activations instead of mu and sigma')

    else:
        raise RuntimeError('Invalid file, only works with .npz to save time')
    

def bootstrap_sample(act1, act2, iterations=10, same_index=False):
    l = len(act1)
    output = []
    size = len(act2)
    if l > size:
        raise RuntimeError('subsampling length is bigger than sample')

    m1, s1 = get_moments(act1)
    iter_array = np.arange(size)
    for i in range (iterations):
        iter_array = np.random.shuffle(iter_array)
        iters, iter_array = iter_array[:l], iter_array[l:]

        temp2 = act2[iters,:]
        m2, s2 = get_moments(temp2)
        output.append(fid.calculate_frechet_distance(m1,s1, m2,s2))
        if len(iter_array)<l:
            iter_array = np.arange(size)
    output=np.array(output)
    return output

def get_moments(act):
    return np.mean(act, axis=0), np.cov(act, rowvar=False)

def main(path, iterations, same_index):
    np.random.seed(1)
    #check_paths(paths)
    path_val = 'path/to/val'
    act1 = get_activation(path)
    act2 = get_activation(path_val)
    print((path.split('/')[-1]))
    
    print('FID score:')
    fid_scores = bootstrap_sample(act1, act2, iterations=iterations, same_index=same_index)
    print(fid_scores)
    print('Mean')
    print(fid_scores.mean())
    print('std')
    print(fid_scores.std())
    

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("--iterations", default=100 ,type=int, nargs=2,
        help='How many bootstrap sampling operations are performed')
    parser.add_argument('--same_index', action='store_true',
        help='Whether the same random sampled indices are used for the computations of m1,s2 & m2,s2')
    args = parser.parse_args()
    main(args.path, args.iterations, args.same_index)
    