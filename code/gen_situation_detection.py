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

anno_fn_map = {
    'dog_walking'  : iu.get_dw_boxes,
    'pingpong'     : iu.get_pp_bboxes,
    'handshake'    : iu.get_hs_bboxes,
    'leadinghorse' : iu.get_lh_bboxes
    }



#===============================================================================
def get_iou(gt_bbox, obj_bbox):
    # convert to x1y1x2y2
    xyxy = np.copy(obj_bbox)
    xyxy[2] += xyxy[0]
    xyxy[3] += xyxy[1]
    
    gt_xyxy = np.copy(gt_bbox)
    gt_xyxy[2] += gt_xyxy[0]
    gt_xyxy[3] += gt_xyxy[1]
    
    # store area
    bbox_area = obj_bbox[2] * obj_bbox[3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    
    # find intersection
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]
    
    inter_x1 = np.maximum(xyxy[0], gt_xyxy[0])
    inter_y1 = np.maximum(xyxy[1], gt_xyxy[1])
    inter_x2 = np.minimum(xyxy[2], gt_xyxy[2])
    inter_y2 = np.minimum(xyxy[3], gt_xyxy[3])
    
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    
    union = bbox_area + gt_area - inter_area
    iou = inter_area / union
    
    return iou



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
    parser.add_argument('--model', dest='model_type')
    parser.add_argument('--method', dest='energy_method')
    args = parser.parse_args()
    
    model_type = args.model_type
    energy_method = args.energy_method
    rcnn_bbox_dir = None
    output_dir = None
    anno_dir = None
    imageset_file = None
    anno_fn = None
    cls_names = None
    
    if model_type == 'dog_walking':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        anno_dir = os.path.join(BASE_DIR, 'data/dog_walking')
        imageset_file = os.path.join(BASE_DIR, 'data/dogwalkingtest_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        data_prefix = 'dw_cycle'
    elif model_type == 'stanford_dw':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/stanford_dog_walking/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/stanford_dog_walking/')
        anno_dir = os.path.join(BASE_DIR, 'data/StanfordSimpleDogWalking/')
        imageset_file = os.path.join(BASE_DIR, 'data/stanford_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        data_prefix = 'dw_cycle'
    elif model_type == 'leadinghorse':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/leadinghorse/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/leadinghorse/')
        anno_dir = os.path.join(BASE_DIR, 'data/LeadingHorse/')
        imageset_file = os.path.join(BASE_DIR, 'data/leadinghorse_fnames_test.txt')
        anno_fn = anno_fn_map['leadinghorse']
        cls_names = ['horse-leader', 'horse', 'lead']
        cls_counts = [1, 1, 1]
        data_prefix = 'lh'
    elif model_type == 'pingpong':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        anno_dir = os.path.join(BASE_DIR, 'data/PingPong')
        imageset_file = os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt')
        anno_fn = anno_fn_map['pingpong']
        cls_names = ['player', 'net', 'table']
        cls_counts = [2, 1, 1]
        data_prefix = 'pingpong'
    elif model_type == 'handshake':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        anno_dir = os.path.join(BASE_DIR, 'data/Handshake')
        imageset_file = os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt')
        anno_fn = anno_fn_map['handshake']
        cls_names = ['person', 'handshake']
        cls_counts = [2, 1]
        data_prefix = 'handshake'
    else:
        pass
    
    return model_type, energy_method, best_bbox_dir, output_dir, anno_dir, imageset_file, anno_fn, cls_names, cls_counts, data_prefix



"""
    model: which model to process
    gmm: the .pkl gmm for sampling

python gen_situation_detection.py --model 'leadinghorse' --method 'pgm'
python gen_situation_detection.py --model 'leadinghorse' --method 'geo'
python gen_situation_detection.py --model 'leadinghorse' --method 'brute'
python gen_situation_detection.py --model 'handshake' --method 'pgm'
python gen_situation_detection.py --model 'handshake' --method 'geo'
python gen_situation_detection.py --model 'handshake' --method 'brute'
python gen_situation_detection.py --model 'pingpong' --method 'pgm'
python gen_situation_detection.py --model 'pingpong' --method 'geo'
python gen_situation_detection.py --model 'pingpong' --method 'brute'
python gen_situation_detection.py --model 'dog_walking' --method 'pgm'
python gen_situation_detection.py --model 'dog_walking' --method 'geo'
python gen_situation_detection.py --model 'dog_walking' --method 'brute'
"""
if __name__ == '__main__':
    # read config
    cfg = get_cfg()
    model_type = cfg[0]
    energy_method = cfg[1]
    best_bbox_dir = cfg[2]
    output_dir = cfg[3]
    anno_dir = cfg[4]
    imageset_file = cfg[5]
    anno_fn = cfg[6]
    cls_names = cfg[7]
    cls_counts = cfg[8]
    data_prefix = cfg[9]
    
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
        
    # process each image in the imageset
    #   annotations[image_filename] => object_dict[object_class] => bbox
    annotations = {}
    for anno_file in imageset:
        fname = os.path.join(anno_dir, anno_file + '.labl')
        f = open(fname, 'rb')
        lines = f.readlines()
        f.close()
        
        img_bboxes = anno_fn(lines[0])
        annotations[anno_file] = img_bboxes
        
    # process each best_bbox csv file
    #   best_bboxes[object_class] => bbox_dict[image_filename] => bbox
    best_bboxes = {}
    for cls_name in cls_names:
        fname = os.path.join(best_bbox_dir, '{}_postest_{}_{}_bboxes.csv'.format(data_prefix, energy_method, cls_name))
        f = open(fname, 'rb')
        for line in f.readlines():
            line = line.rstrip('\n')
            csv = line.split(', ')
            src_fname = csv[0].split('.')[0]
            x = int(csv[2])
            y = int(csv[3])
            w = int(csv[4])
            h = int(csv[5])
            bbox = np.array((x, y, w, h))
            
            if src_fname not in best_bboxes:
                best_bboxes[src_fname] = {}
            
            if cls_name not in best_bboxes[src_fname]:
                best_bboxes[src_fname][cls_name] = []
            
            best_bboxes[src_fname][cls_name].append(bbox)
    
    # calculate IoU for the object classes
    iou_tracker = {}
    for image_fname in annotations.keys():
        iou_tracker[image_fname] = []
        for cls_ix, cls_name in enumerate(cls_names):
            gt_bboxes = []
            cls_count = cls_counts[cls_ix]
            if cls_count > 1:
                for n in range(1, cls_count+1):
                    anno_cls_name = '{}__{}'.format(cls_name, n)
                    gt_bbox = annotations[image_fname][anno_cls_name]
                    gt_bboxes.append(gt_bbox)
            else:
                gt_bbox = annotations[image_fname][cls_name]
                gt_bboxes.append(gt_bbox)
            
            # get the model's best-choice bboxes
            img_bboxes = best_bboxes[image_fname][cls_name]
            
            # calculate IoU of gt and best bboxes
            iou = None
            if cls_count == 1:
                iou = get_iou(gt_bboxes[0], img_bboxes[0])
                iou_tracker[image_fname].append(iou)
            elif cls_count == 2:
                # if there are 2 class objects, find configuration of bboxes
                # that gives the highest IoU
                i00 = get_iou(gt_bboxes[0], img_bboxes[0])
                i01 = get_iou(gt_bboxes[0], img_bboxes[1])
                i10 = get_iou(gt_bboxes[1], img_bboxes[0])
                i11 = get_iou(gt_bboxes[1], img_bboxes[1])
                
                iou_norm = np.array((i00, i11))
                iou_swap = np.array((i01, i10))
                iou_swap_alt = np.array((i10, i01))
                
                iou = iou_norm
                if np.average(iou_swap) > np.average(iou):
                    iou = iou_swap
                
                for i in iou:
                    iou_tracker[image_fname].append(i)
    
    # calculate detection threshold counts
    #print('situation success rates for "{}", {} method'.format(model_type, energy_method))
    thresh_list = np.linspace(0., 1., num=11) #TODO: parameterize
    n_images = len(imageset)
    rates = []
    for threshold in thresh_list:
        n_successes = 0
        for image_fname in annotations.keys():
            ious = iou_tracker[image_fname]
            if min(ious) >= threshold:
                n_successes += 1
        rates.append((threshold, n_successes / float(n_images)))
        
        #print('{}: {:0.3f}'.format(threshold, n_successes / float(n_images)))
    rates = np.array(rates)
    out_fname = os.path.join(output_dir, 'sdr_{}_{}.csv'.format(model_type, energy_method))
    np.savetxt(out_fname, rates, header='threshold, hit_rate', comments='', fmt='%0.2f, %0.3f')
