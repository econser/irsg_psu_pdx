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
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        anno_dir = os.path.join(BASE_DIR, 'data/dog_walking')
        imageset_file = os.path.join(BASE_DIR, 'data/dogwalkingtest_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
    elif model_type == 'pingpong':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        anno_dir = os.path.join(BASE_DIR, 'data/PingPong')
        imageset_file = os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt')
        anno_fn = anno_fn_map['pingpong']
        cls_names = ['player', 'net', 'table']
        cls_counts = [2, 1, 1]
    elif model_type == 'handshake':
        best_bbox_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        anno_dir = os.path.join(BASE_DIR, 'data/Handshake')
        imageset_file = os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt')
        anno_fn = anno_fn_map['handshake']
        cls_names = ['person', 'handshake']
        cls_counts = [2, 1]
    else:
        pass
    
    return model_type, best_bbox_dir, output_dir, anno_dir, imageset_file, anno_fn, cls_names, cls_counts



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
    cls_counts = cfg[7]
    
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
        fname = os.path.join(best_bbox_dir, '{}_bboxes.csv'.format(cls_name))
        f = open(fname, 'rb')
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
    
    # calculate IoU for the object classes
    iou_by_class = {}
    image_fnames = annotations.keys()
    for cls_ix, cls_name in enumerate(cls_names):
        iou_results = []
        for image_fname in image_fnames:
            # get the ground truth bboxes for this image
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
                iou_results.append((image_fname, iou))
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
                    iou_results.append((image_fname, i))
        
        iou_results = np.array(iou_results, dtype=np.object)
        iou_fname = os.path.join(output_dir, '{}_iou.csv'.format(cls_name))
        
        np.savetxt(iou_fname, iou_results, fmt='%s, %0.3f')
        iou_by_class[cls_name] = iou_results
    
    # calculate detection threshold counts
    thresh_list = np.linspace(0., 1., num=11) #TODO: parameterize
    n_images = len(imageset)
    for cls_ix, cls_name in enumerate(cls_names):
        ious = iou_by_class[cls_name]
        max_hits = cls_counts[cls_ix] * n_images
        detections = []
        for thresh_val in thresh_list:
            hits = np.where(ious[:,1] >= thresh_val)
            n_hits = len(hits[0])
            hit_rate = n_hits / (max_hits * 1.)
            detections.append((thresh_val, hit_rate))
        detections = np.array(detections)
        detection_fname = os.path.join(output_dir, '{}_detections.csv'.format(cls_name))
        np.savetxt(detection_fname, detections, header='threshold, hit_rate', comments='', fmt='%0.2f, %0.3f')
    
    # plot the detections data
    # for plot in plots:
    #    setup plot
    #    save plot
