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
    if l != len(act2):
        raise RuntimeError('only works when length of both arrays is identical')
    for i in range (iterations):
        iter_array = np.random.randint(low=0, high=l, size=l)
        temp1 = act1[iter_array,:]
        if same_index is False:
            iter_array = np.random.randint(low=0, high=l, size=l)
        temp2 = act2[iter_array,:]
        m1, s1 = get_moments(temp1)
        m2, s2 = get_moments(temp2)

        output.append(fid.calculate_frechet_distance(m1,s1, m2,s2))
    output=np.array(output)
    return output

def get_moments(act):
    return np.mean(act, axis=0), np.cov(act, rowvar=False)

def main(paths, iterations, same_index):
    check_paths(paths)

    act1 = get_activation(paths[0])
    act2 = get_activation(paths[1])
    #print(act1.shape)
    #print(act2.shape)
    print((paths[0].split('/')[-1]))
    print('Same index:', same_index)
    print('FID score:')
    m1, s1 = get_moments(act1)
    m2, s2 = get_moments(act2)
    print(fid.calculate_frechet_distance(m1,s1, m2,s2))

    fid_scores = bootstrap_sample(act1, act2, iterations=iterations, same_index=same_index)
    print(fid_scores)
    print('Mean')
    print(fid_scores.mean())
    print('std')
    print(fid_scores.std())
    

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("paths", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("--iterations", default=100 ,type=int, nargs=2,
        help='How many bootstrap sampling operations are performed')
    parser.add_argument('--same_index', action='store_true',
        help='Whether the same random sampled indices are used for the computations of m1,s2 & m2,s2')
    args = parser.parse_args()
    main(args.paths, args.iterations, args.same_index)
    