from __future__ import print_function

import os
import sys
import cPickle
import numpy as np
import itertools as it
import irsg_utils as iu

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects



#===============================================================================
BASE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(BASE_DIR, os.pardir)
BASE_DIR = os.path.abspath(BASE_DIR)

N_PAIRS = 5
N_DENSITY_BINS = 20
N_PROB_BINS = 20

#===============================================================================
class BinaryRelation (object):
    def __init__(self, sub_bboxes, obj_bboxes, gmm):
        self.box_pairs = np.array([x for x in it.product(sub_bboxes, obj_bboxes)])
        self.box_vec = iu.get_gmm_features(self.box_pairs, in_format='xywh')
        self.density = iu.gmm_pdf(self.box_vec, gmm.gmm_weights, gmm.gmm_mu, gmm.gmm_sigma)
        self.prob = 1. / (1. + np.exp(-(gmm.platt_a * self.density + gmm.platt_b)))
        self.sort_ixs = np.argsort(self.prob)[::-1]



class ImageData (object):
    def __init__(self, csv_dir, cls_names, rel_map, gmms):
        self.unary = {}
        for cls in cls_names:
            fname = os.path.join(csv_dir, cls, image + '.csv')
            unary_data = np.genfromtxt(fname, delimiter=',')
            boxes = unary_data[:, 0:4]
            scores = unary_data[:, 4]
            self.unary[cls] = (boxes, scores)
        
        self.binary = {}
        for rel in relations:
            self.binary[rel] = {}
            for instance in rel_map[rel]:
                sub_cls = instance[0]
                sub_tuple = self.unary[sub_cls]
                sub_bboxes = sub_tuple[0]
                
                obj_cls = instance[1]
                obj_tuple = self.unary[obj_cls]
                obj_bboxes = obj_tuple[0]
                
                self.binary[rel] = BinaryRelation(sub_bboxes, obj_bboxes, gmms[rel])



#===============================================================================
def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Binary Relation generation')
    parser.add_argument('--model', dest='model_type', choices=['dog_walking', 'stanford_dw', 'pingpong', 'handshake'])
    parser.add_argument('--dataset', dest='dataset', choices=['pos', 'hard_neg', 'full_neg'])
    parser.add_argument('--gmm', dest='gmm_fname')
    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset
    gmm_fname = args.gmm_fname
    
    output_dir = None
    anno_dir = None
    imageset_file = None
    anno_fn = None
    cls_names = None
    
    gmm_dir = os.path.join(BASE_DIR, 'data/')
    gmm_fname = os.path.join(gmm_dir, gmm_fname)
    
    if model_type == 'dog_walking':
        csv_dir_map = {
            'pos' : os.path.join(BASE_DIR, 'run_results/dw_fullpos/'),
            'hard_neg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/'),
            'full_neg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/')
        }
        image_dir_map = {
            'pos' : os.path.join(BASE_DIR, 'data/dog_walking'),
            'hard_neg' : os.path.join(BASE_DIR, ''),
            'full_neg' : os.path.join(BASE_DIR, '')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        imageset_files = {
            'pos': os.path.join(BASE_DIR, 'data/dogwalkingtest_fnames_test.txt'),
            'hard_neg': None,
            'full_neg': None
        }
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        rel_map = {
            'holding' : [('dog_walker', 'leash')],
            'attached_to' : [('leash', 'dog')],
            'walked_by' : [('dog', 'dog_walker')],
            'is_walking' : [('dog_walker', 'dog')]
        }
    elif model_type == 'stanford_dw':
        csv_dir_map = {
            'pos' : os.path.join(BASE_DIR, 'run_results/stanford_dog_walking/'),
            'hard_neg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/'),
            'full_neg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/')
        }
        image_dir_map = {
            'pos' : os.path.join(BASE_DIR, ''),
            'hard_neg' : os.path.join(BASE_DIR, ''),
            'full_neg' : os.path.join(BASE_DIR, '')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/stanford_dog_walking/')
        imageset_files = {
            'pos': os.path.join(BASE_DIR, 'data/stanford_fnames_test.txt'),
            'hard_neg': None,
            'full_neg': None
        }
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        rel_map = {
            'holding' : [('dog_walker', 'leash')],
            'attached_to' : [('leash', 'dog')],
            'walked_by' : [('dog', 'dog_walker')],
            'is_walking' : [('dog_walker', 'dog')]
        }
    elif model_type == 'pingpong':
        csv_dir_map = {
            'pos' : os.path.join(BASE_DIR, 'run_results/pingpong/'),
            'hard_neg' : os.path.join(BASE_DIR, 'run_results/pingpong_hardneg/'),
            'full_neg' : os.path.join(BASE_DIR, 'run_results/pingpong_fullneg/')
        }
        image_dir_map = {
            'pos' : os.path.join(BASE_DIR, ''),
            'hard_neg' : os.path.join(BASE_DIR, ''),
            'full_neg' : os.path.join(BASE_DIR, '')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        imageset_files = {
            'pos': os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt'),
            'hard_neg': None,
            'full_neg': None
        }
        cls_names = ['player', 'net', 'table']
        cls_counts = [2, 1, 1]
        rel_map = {
            'at' : [('player__1', 'table'), ('player__2', 'table')],
            'on' : [('net', 'table')],
            'playing_pingpong_with' : [('player__1', 'player__2'), ('player__2', 'player__1')]
        }
    elif model_type == 'handshake':
        csv_dir_map = {
            'pos' : os.path.join(BASE_DIR, 'run_results/hs_fullpos/'),
            'hard_neg' : os.path.join(BASE_DIR, 'run_results/hs_hardneg/'),
            'full_neg' : os.path.join(BASE_DIR, 'run_results/hs_fullneg/')
        }
        image_dir_map = {
            'pos' : os.path.join(BASE_DIR, ''),
            'hard_neg' : os.path.join(BASE_DIR, ''),
            'full_neg' : os.path.join(BASE_DIR, '')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        imageset_files = {
            'pos': os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt'),
            'hard_neg': None,
            'full_neg': None
        }
        cls_names = ['person', 'handshake']
        cls_counts = [2, 1]
        rel_map = {
            'extending' : [('person__1', 'handshake'), ('person__2', 'handshake')],
            'handshaking' : [('person__1', 'person__2'), ('person__2', 'person__1')]
        }
    else:
        pass
    
    return model_type, gmm_fname, csv_dir_map[dataset], image_dir_map[dataset], output_dir, imageset_files[dataset], rel_map, cls_names, cls_counts



"""
    model: which model to process
    gmm: the .pkl gmm for sampling

python viz_gmm_performance.py --model handshake --dataset pos --gmm handshake_gmms.pkl
python viz_gmm_performance.py --model dog_walking --dataset pos --gmm dw_gmms_unlog.pkl
python viz_gmm_performance.py --model pingpong --dataset pos --gmm pingpong_gmms.pkl
"""
if __name__ == '__main__':
    # read config
    cfg = get_cfg()
    
    model_type    = cfg[0]
    gmm_fname     = cfg[1]
    csv_dir       = cfg[2]
    image_dir     = cfg[3]
    output_dir    = cfg[4]
    imageset_file = cfg[5]
    rel_map       = cfg[6]
    cls_names     = cfg[7]
    cls_counts    = cfg[8]
    
    # create output dir, if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # prep the imageset (imageXXX.jpg\r\n -> imageXXX)
    imageset = []
    if imageset_file is not None:
        f = open(imageset_file)
        imageset = f.readlines()
        f.close()
        imageset = [fname.rstrip('\n') for fname in imageset]
        imageset = [fname.rstrip('\r') for fname in imageset]
    else:
        imageset = os.listdir(anno_dir)
        imageset = filter(lambda x: '.labl' in x, imageset)
    imageset = [fname.split('.')[0] for fname in imageset]
    imageset = ['dog-walking394']
    n_images = len(imageset)
        
    # get the gmms
    f = open(gmm_fname, 'rb')
    gmms = cPickle.load(f)
    
    relations = gmms.keys()
    
    # read in image class data
    image_data = {}
    for image_ix, image in enumerate(imageset):
        print('loading {} ({}/{})\r'.format(image, image_ix+1, n_images), end='')
        sys.stdout.flush()
        
        img = ImageData(csv_dir, cls_names, rel_map, gmms)
        image_data[image] = img
    
    # TODO: pickle the image_data?
    import pdb; pdb.set_trace()
    
    # draw the top bbox pairs, density hist, and prob hist
    sub_color = [0.9, 0.1, 0.1]
    obj_color = [0.1, 0.1, 0.9]
    
    for rel in rel_map:
        for rel_instance in rel_map[rel]:
            for image in imageset:
                image_fname = os.path.join(image_dir, image + '.jpg')
                image_pil = Image.open(image_fname)
                image_bytes = np.array(image_pil, dtype=np.uint8)
                
                plt.figure(0)
                plt.suptitle('{} {} {}'.format(rel_instance[0], rel, rel_instance[1]))
                
                image_axes   = plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
                density_axis = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
                prob_axis    = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)
                
                # draw image and and boxes
                image_axes.imshow(image_bytes)
                image_axes.get_xaxis().set_visible(False)
                image_axes.get_yaxis().set_visible(False)
                bin_rel = image_data[image].binary[rel]
                top_ixs = bin_rel.sort_ixs[0:N_PAIRS]
                for ix in top_ixs:
                    sub_bbox = bin_rel.box_pairs[ix][0]
                    sx = sub_bbox[0]
                    sy = sub_bbox[1]
                    sw = sub_bbox[2]
                    sh = sub_bbox[3]
                    box = patches.Rectangle((sx, sy), sw, sh, linewidth=3, edgecolor=sub_color, facecolor='None')
                    image_axes.add_patch(box)
                    
                    obj_bbox = bin_rel.box_pairs[ix][1]
                    ox = obj_bbox[0]
                    oy = obj_bbox[1]
                    ow = obj_bbox[2]
                    oh = obj_bbox[3]
                    box = patches.Rectangle((ox, oy), ow, oh, linewidth=3, edgecolor=obj_color, facecolor='None')
                    image_axes.add_patch(box)

                    sub_cx = int(sx + sw * 0.5)
                    sub_cy = int(sy + sh * 0.5)
                    obj_cx = int(ox + ow * 0.5)
                    obj_cy = int(oy + oh * 0.5)
                    dx = obj_cx - sub_cx
                    dy = obj_cy - sub_cy
                    image_axes.arrow(sub_cx, sub_cy, dx, dy, linewidth=4, color='k', head_width=10, head_length=10)
                
                # draw density hist
                d = density_axis.hist(bin_rel.density, bins=N_DENSITY_BINS, log=True)
                density_axis.set_title('density histogram')
                
                # overlay sigmoid
                
                # draw prob hist
                p = prob_axis.hist(bin_rel.prob, bins=N_PROB_BINS, log=True)
                prob_axis.set_title('probability histogram')
                plt.xlim([0.0, 1.0])
                
                plt.tight_layout(pad=0.2)
                plt.show()
    
    # save the matrix plot
