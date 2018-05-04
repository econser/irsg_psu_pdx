import numpy as np
import irsg_utils as iu

#===============================================================================
import os
BASE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(BASE_DIR, os.pardir)
BASE_DIR = os.path.abspath(BASE_DIR)

anno_fn_map = {
    'dog_walking' : iu.get_dw_boxes,
    'pingpong'    : iu.get_pp_bboxes,
    'handshake'   : iu.get_hs_bboxes
    }



#===============================================================================
def get_ious(gt_bbox, rcnn_bboxes):
    # convert to x1y1x2y2
    xyxy = np.copy(rcnn_bboxes)
    xyxy[:,2] += xyxy[:,0]
    xyxy[:,3] += xyxy[:,1]
    
    gt_xyxy = np.copy(gt_bbox)
    gt_xyxy[2] += gt_xyxy[0]
    gt_xyxy[3] += gt_xyxy[1]
    
    # store areas
    bbox_areas = rcnn_bboxes[:,2] * rcnn_bboxes[:,3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    
    # find intersections
    x1 = xyxy[:,0]
    y1 = xyxy[:,1]
    x2 = xyxy[:,2]
    y2 = xyxy[:,3]
    
    inter_x1 = np.maximum(x1, gt_xyxy[0])
    inter_y1 = np.maximum(y1, gt_xyxy[1])
    inter_x2 = np.minimum(x2, gt_xyxy[2])
    inter_y2 = np.minimum(y2, gt_xyxy[3])
    
    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
    inter_area = inter_w * inter_h
    
    unions = bbox_areas + gt_area - inter_area
    ious = inter_area / unions
    
    return ious



def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Binary Relation generation')
    parser.add_argument('--model', dest='model_type')
    args = parser.parse_args()
    
    model_type = args.model_type
    rcnn_bbox_dir = None
    output_dir = None
    anno_dir = None
    imageset_file = None
    anno_fn = None
    cls_names = None
    
    if model_type == 'dog_walking':
        rcnn_bbox_dir = os.path.join(BASE_DIR, 'run_results/dw_fullpos/')
        output_dir = os.path.join(BASE_DIR, 'research/output/rel_results')
        anno_dir = os.path.join(BASE_DIR, 'research/data/dog_walking')
        imageset_file = os.path.join(BASE_DIR, 'research/data/dogwalkingtest_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
    elif model_type == 'pingpong':
        rcnn_bbox_dir = os.path.join(BASE_DIR, 'run_results/pingpong/')
        output_dir = os.path.join(BASE_DIR, 'output/rel_results')
        anno_dir = os.path.join(BASE_DIR, 'data/PingPong')
        imageset_file = os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt')
        anno_fn = anno_fn_map['pingpong']
        cls_names = ['player', 'net', 'table']
    elif model_type == 'handshake':
        rcnn_bbox_dir = os.path.join(BASE_DIR, 'run_results/hs_fullpos/')
        output_dir = os.path.join(BASE_DIR, 'output/rel_results')
        anno_dir = os.path.join(BASE_DIR, 'data/Handshake')
        imageset_file = os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt')
        anno_fn = anno_fn_map['handshake']
        cls_names = ['person', 'handshake']
    else:
        pass
    
    return model_type, rcnn_bbox_dir, output_dir, anno_dir, imageset_file, anno_fn, cls_names



"""
    best_bbox_dir: base directory for the best bboxes
    anno_dir: directory for annotation files
    imageset_file: list of filenames to generate output for (fq) (optional)
    
    model_type: which model to process
    
    output_dir: directory for storing IoU data and plots
"""
if __name__ == '__main__':
    # read config
    cfg = get_cfg()
    model_type = cfg[0]
    best_bbox_dir = cfg[1]
    output_dir = cfg[2]
    anno_dir = cfg[3]
    imageset_file = cfg[4]
    anno_fn = cfg[5]
    cls_names = cfg[6]
    
    # create output dir, if necessary
    output_model_dir = os.path.join(output_dir, model_type)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    
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
    annotations = {}
    for anno_file in imageset:
        fname = os.path.join(anno_dir, anno_file + '.labl')
        f = open(fname, 'rb')
        lines = f.readlines()
        f.close()
        
        img_bboxes = anno_fn(lines[0])
        annotations[anno_file] = img_bboxes

    # process each best_bbox csv file
    best_bboxes = {}
    for cls in cls_names:
        fname = os.path.join(data_dir, '{}_bboxes.csv'format(cls_name))
        f = open(fname, 'rb')
        for line in f.readlines():
            csv = line.split(', ')
            src_fname = csv[0]
            x = int(csv[1])
            y = int(csv[2])
            w = int(csv[3])
            h = int(csv[4])
            best_bboxes[src_fname].append(np.array(x, y, w, h))
    

        # TODO: get class names
        for cls_nm in cls_names:
            
        for relation in relations:
            sub_gt_bbox = img_bboxes[relation[0]]
            obj_gt_bbox = img_bboxes[relation[2]]
            
            # find the index of the box in relation[0] that has the largest IoU
            sub_dir = relation[0]
            if '__' in sub_dir:
                sub_dir = sub_dir.split('__')[0]
            
            sub_bbox_fname = os.path.join(rcnn_bbox_dir, sub_dir, anno_file + '.csv')
            sub_rcnn_bboxes = np.genfromtxt(sub_bbox_fname, delimiter=',')
            
            ious = get_ious(sub_gt_bbox, sub_rcnn_bboxes)
