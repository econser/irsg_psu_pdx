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



#===============================================================================
class BinaryRelation (object):
    def __init__(self, sub_bboxes, obj_bboxes, gmm):
        box_pairs = np.array([x for x in it.product(sub_bboxes, obj_bboxes)])
        box_vec = iu.get_gmm_features(box_pairs, in_format='xywh')
        density = iu.gmm_pdf(box_vec, gmm.gmm_weights, gmm.gmm_mu, gmm.gmm_sigma)
        density = np.reshape(density, (len(sub_bboxes), len(obj_bboxes)))
        self.prob = 1. / (1. + np.exp(-(gmm.platt_a * density + gmm.platt_b)))
        #self.box_pairs = np.array([x for x in it.product(sub_bboxes, obj_bboxes)])
        #self.box_vec = iu.get_gmm_features(self.box_pairs, in_format='xywh')
        #self.density = iu.gmm_pdf(self.box_vec, gmm.gmm_weights, gmm.gmm_mu, gmm.gmm_sigma)
        #self.density = np.reshape(self.density, (len(sub_bboxes), len(obj_bboxes)))
        #self.prob = 1. / (1. + np.exp(-(gmm.platt_a * self.density + gmm.platt_b)))
        #self.sort_ixs = np.argsort(self.prob)[::-1]



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
        for rel in gmms.keys():
            if rel not in rel_map:
                continue
            
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
    parser.add_argument('--dataset', dest='dataset', choices=['postest', 'hardneg', 'fullneg'])
    parser.add_argument('--gmm', dest='gmm_fname')
    args = parser.parse_args()
    
    model_type = args.model_type
    dataset = args.dataset
    gmm_fname = args.gmm_fname
    
    output_dir = None
    imageset_file = None
    anno_fn = None
    cls_names = None
    
    gmm_dir = os.path.join(BASE_DIR, 'data/')
    gmm_fname = os.path.join(gmm_dir, gmm_fname)
    
    if model_type == 'dog_walking':
        csv_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/dw_fullpos/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/')
        }
        image_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/dw_fullpos/dog/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/dog/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/dog/')
        }
        best_bbox_map = {
            'postest' : os.path.join(BASE_DIR, 'output/full_runs/dog_walking/dw_cycle_postest_pgm_{}_bboxes.csv'),
            'hardneg': os.path.join(BASE_DIR, 'output/full_runs/dog_walking/dw_cycle_hardneg_pgm_{}_bboxes.csv'),
            'fullneg': os.path.join(BASE_DIR, 'output/full_runs/dog_walking/dw_cycle_fullneg_pgm_{}_bboxes.csv')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        imageset_files = {
            'postest': os.path.join(BASE_DIR, 'data/dogwalkingtest_fnames_test.txt'),
            'hardneg': None,
            'fullneg': None
        }
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        rel_map = {
            'holding' : [('dog_walker', 'leash')],
            'attached_to' : [('leash', 'dog')],
            'walked_by' : [('dog', 'dog_walker')]#, 'is_walking' : [('dog_walker', 'dog')]
        }
    elif model_type == 'stanford_dw':
        csv_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/stanford_dog_walking/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/')
        }
        image_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/stanford_dog_walking/dog/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/dw_hardneg/dog/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/dw_fullneg/dog/')
        }
        best_bbox_map = {
            'postest' : os.path.join(BASE_DIR, 'full_runs/stanford_dog_walking/dw_cycle_postest_pgm_{}_bboxes.csv'),
            'hardneg': os.path_join(BASE_DIR, 'full_runs/stanford_dog_walking/dw_cycle_hardneg_pgm_{}_bboxes.csv'),
            'fullneg': os.path_join(BASE_DIR, 'full_runs/stanford_dog_walking/dw_cycle_fullneg_pgm_{}_bboxes.csv')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/stanford_dog_walking/')
        imageset_files = {
            'postest': os.path.join(BASE_DIR, 'data/stanford_fnames_test.txt'),
            'hardneg': None,
            'fullneg': None
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
            'postest' : os.path.join(BASE_DIR, 'run_results/pingpong/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/pingpong_hardneg/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/pingpong_fullneg/')
        }
        image_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/pingpong/player/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/pingpong_hardneg/player/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/pingpong_fullneg/player/')
        }
        best_bbox_map = {
            'postest' : os.path.join(BASE_DIR, 'full_runs/pingpong/pingpong_postest_pgm_{}_bboxes.csv'),
            'hardneg': os.path_join(BASE_DIR, 'full_runs/pingpong/pingpong_hardneg_pgm_{}_bboxes.csv'),
            'fullneg': os.path_join(BASE_DIR, 'full_runs/pingpong/pingpong_fullneg_pgm_{}_bboxes.csv')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        imageset_files = {
            'postest': os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt'),
            'hardneg': None,
            'fullneg': None
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
            'postest' : os.path.join(BASE_DIR, 'run_results/hs_fullpos/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/hs_hardneg/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/hs_fullneg/')
        }
        image_dir_map = {
            'postest' : os.path.join(BASE_DIR, 'run_results/hs_fullpos/person/'),
            'hardneg' : os.path.join(BASE_DIR, 'run_results/hs_hardneg/person/'),
            'fullneg' : os.path.join(BASE_DIR, 'run_results/hs_fullneg/person/')
        }
        best_bbox_map = {
            'postest' : os.path.join(BASE_DIR, 'full_runs/handshake/handshake_postest_pgm_{}_bboxes.csv'),
            'hardneg': os.path_join(BASE_DIR, 'full_runs/handshake/handshake_hardneg_pgm_{}_bboxes.csv'),
            'fullneg': os.path_join(BASE_DIR, 'full_runs/handshake/handshake_fullneg_pgm_{}_bboxes.csv')
        }
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        imageset_files = {
            'postest': os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt'),
            'hardneg': None,
            'fullneg': None
        }
        cls_names = ['person', 'handshake']
        cls_counts = [2, 1]
        rel_map = {
            'extending' : [('person__1', 'handshake'), ('person__2', 'handshake')],
            'handshaking' : [('person__1', 'person__2'), ('person__2', 'person__1')]
        }
    else:
        pass
    
    return model_type, gmm_fname, csv_dir_map[dataset], image_dir_map[dataset], best_bbox_map[dataset], output_dir, imageset_files[dataset], rel_map, cls_names, cls_counts, dataset



"""
    model: which model to process
    gmm: the .pkl gmm for sampling

python gen_probability_factors.py --model dog_walking --dataset postest --gmm dw_gmms_revised.pkl
python gen_probability_factors.py --model handshake --dataset pos --gmm handshake_gmms.pkl
python gen_probability_factors.py --model pingpong --dataset pos --gmm pingpong_gmms.pkl
"""
if __name__ == '__main__':
    # read config
    cfg = get_cfg()
    
    model_type     = cfg[0]
    gmm_fname      = cfg[1]
    csv_dir        = cfg[2]
    image_dir      = cfg[3]
    best_bbox_file = cfg[4]
    output_dir     = cfg[5]
    imageset_file  = cfg[6]
    rel_map        = cfg[7]
    cls_names      = cfg[8]
    cls_counts     = cfg[9]
    dataset        = cfg[10]
    
    # create output dir, if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # prep the imageset
    imageset = []
    if imageset_file is not None:
        f = open(imageset_file)
        imageset = f.readlines()
        f.close()
        imageset = [fname.rstrip('\n') for fname in imageset]
        imageset = [fname.rstrip('\r') for fname in imageset]
    else:
        imageset = os.listdir(image_dir)
        imageset = filter(lambda x: '.csv' in x, imageset)
    imageset = [fname.split('.')[0] for fname in imageset]
    #imageset = ['dog-walking394']
    #imageset = ['5462007065_b94d86008a_b']
    n_images = len(imageset)
    
    # get the gmms
    f = open(gmm_fname, 'rb')
    gmms = cPickle.load(f)
    relations = gmms.keys()
    
    # process each best_bbox csv file
    #   best_bboxes[image_filename] => bbox_dict[class_name] => [bbox]
    best_bboxes = {}
    for cls_name in cls_names:
        f = open(best_bbox_file.format(cls_name), 'rb')
        for line in f.readlines():
            line = line.rstrip('\n')
            csv = line.split(', ')
            src_fname = csv[0].split('.')[0]
            
            x = int(csv[1])
            y = int(csv[2])
            w = int(csv[3])
            h = int(csv[4])
            bbox = np.array((x, y, w, h))
            
            if src_fname not in best_bboxes:
                best_bboxes[src_fname] = {}
            
            if cls_name not in best_bboxes[src_fname]:
                best_bboxes[src_fname][cls_name] = []
            
            best_bboxes[src_fname][cls_name].append(bbox)
        f.close()
    
    # read in image class data
    image_data = {}
    for image_ix, image in enumerate(imageset):
        print('loading {} ({}/{})\r'.format(image, image_ix+1, n_images), end='')
        sys.stdout.flush()
        
        img = ImageData(csv_dir, cls_names, rel_map, gmms)
        image_data[image] = img
    
    # store the unary and binary probs for the best match boxes
    results = []
    header = ['filename']
    header_prepared = False
    for image_ix, image in enumerate(imageset):
        print('calculating {} ({}/{})\r\n'.format(image, image_ix+1, n_images), end='')
        result = [image]
        best_bbox_ixs = {}
        for cls in cls_names:
            best_bbox_coords = best_bboxes[image][cls][0]
            coords_match = image_data[image].unary[cls][0] == best_bbox_coords
            coords_match = coords_match.all(axis=1)
            best_bbox_ix = np.where(coords_match)[0][0]
            top_cfg_prob = image_data[image].unary[cls][1][best_bbox_ix]
            result.append(top_cfg_prob)
            
            best_bbox_ixs[cls] = best_bbox_ix
            
            if not header_prepared:
                header.append(cls)
        
        for rel in rel_map:
            for rel_instance in rel_map[rel]:
                bin_rel = image_data[image].binary[rel]
                
                sub_name = rel_instance[0]
                sub_ix = best_bbox_ixs[sub_name]
                
                obj_name = rel_instance[1]
                obj_ix = best_bbox_ixs[obj_name]
                
                top_rel_prob = bin_rel.prob[sub_ix][obj_ix]
                
                sub_bbox = image_data[image].unary[cls][0][sub_ix]
                obj_bbox = image_data[image].unary[cls][0][obj_ix]
                # run through the gmm to check
                
                result.append(top_rel_prob)
                if not header_prepared:
                    header.append(rel)
        
        header_prepared = True
        results.append(result)
        
        header_str = ''
        np_fmt_str = ''
        n_items = len(header)
        for n, item in enumerate(header):
            header_str += item
            
            if n == 0:
                np_fmt_str = '%s'
            else:
                np_fmt_str += '%0.3f'
                
            if n < n_items-1:
                header_str += ', '
                np_fmt_str += ', '
        
        result_array = np.array(results, dtype=np.object)
        output_fname = '{}_{}_factors.csv'.format(model_type, dataset)
        np.savetxt(os.path.join(output_dir, output_fname), result_array, header=header_str, fmt=np_fmt_str, comments='')

"""
      filename, dog_walker, leash,   dog, attached_to, walked_by, holding, -ln(P), C
dog-walking394, 0.997, 0.927, 0.991, 0.399, 0.488, 0.424

"""
