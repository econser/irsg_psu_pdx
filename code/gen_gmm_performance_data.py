import cPickle
import numpy as np
import irsg_utils as iu

#===============================================================================
import os
BASE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(BASE_DIR, os.pardir)
BASE_DIR = os.path.abspath(BASE_DIR)

N_SAMPLES = 10000

anno_fn_map = {
    'dog_walking' : iu.get_dw_boxes,
    'pingpong'    : iu.get_pp_bboxes,
    'handshake'   : iu.get_hs_bboxes
    }



#===============================================================================
def get_ious(gt_bbox, obj_bboxes):
    # convert to x1y1x2y2
    xyxy = np.copy(obj_bboxes)
    xyxy[:,2] += xyxy[:,0]
    xyxy[:,3] += xyxy[:,1]
    
    gt_xyxy = np.copy(gt_bbox)
    gt_xyxy[2] += gt_xyxy[0]
    gt_xyxy[3] += gt_xyxy[1]
    
    # store areas
    bbox_areas = obj_bboxes[:,2] * obj_bboxes[:,3]
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



#def get_iou(gt_bbox, obj_bbox):
#    # convert to x1y1x2y2
#    xyxy = np.copy(obj_bbox)
#    xyxy[2] += xyxy[0]
#    xyxy[3] += xyxy[1]
#    
#    gt_xyxy = np.copy(gt_bbox)
#    gt_xyxy[2] += gt_xyxy[0]
#    gt_xyxy[3] += gt_xyxy[1]
#    
#    # store area
#    bbox_area = obj_bbox[2] * obj_bbox[3]
#    gt_area = gt_bbox[2] * gt_bbox[3]
#    
#    # find intersection
#    x1 = xyxy[0]
#    y1 = xyxy[1]
#    x2 = xyxy[2]
#    y2 = xyxy[3]
#    
#    inter_x1 = np.maximum(xyxy[0], gt_xyxy[0])
#    inter_y1 = np.maximum(xyxy[1], gt_xyxy[1])
#    inter_x2 = np.minimum(xyxy[2], gt_xyxy[2])
#    inter_y2 = np.minimum(xyxy[3], gt_xyxy[3])
#    
#    inter_w = np.maximum(0.0, inter_x2 - inter_x1 + 1)
#    inter_h = np.maximum(0.0, inter_y2 - inter_y1 + 1)
#    inter_area = inter_w * inter_h
#    
#    union = bbox_area + gt_area - inter_area
#    iou = inter_area / union
#    
#    return iou



def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Binary Relation generation')
    parser.add_argument('--model', dest='model_type')
    parser.add_argument('--gmm', dest='gmm_fname')
    args = parser.parse_args()
    
    model_type = args.model_type
    gmm_fname = args.gmm_fname
    output_dir = None
    anno_dir = None
    imageset_file = None
    anno_fn = None
    cls_names = None
    
    gmm_dir = os.path.join(BASE_DIR, 'data/')
    gmm_fname = os.path.join(gmm_dir, gmm_fname)
    
    if model_type == 'dog_walking':
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/dog_walking/')
        anno_dir = os.path.join(BASE_DIR, 'data/dog_walking')
        imageset_file = os.path.join(BASE_DIR, 'data/dogwalkingtest_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        rel_map = {
            'holding' : [('dog_walker', 'leash')],
            'attached_to' : [('leash', 'dog')],
            'walked_by' : [('dog', 'dog_walker')],
            'is_walking' : [('dog_walker', 'dog')]
        }
    elif model_type == 'stanford_dw':
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/stanford_dog_walking/')
        anno_dir = os.path.join(BASE_DIR, 'data/StanfordSimpleDogWalking/')
        imageset_file = os.path.join(BASE_DIR, 'data/stanford_fnames_test.txt')
        anno_fn = anno_fn_map['dog_walking']
        cls_names = ['dog_walker', 'leash', 'dog']
        cls_counts = [1, 1, 1]
        rel_map = {
            'holding' : [('dog_walker', 'leash')],
            'attached_to' : [('leash', 'dog')],
            'walked_by' : [('dog', 'dog_walker')],
            'is_walking' : [('dog_walker', 'dog')]
        }
    elif model_type == 'pingpong':
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/pingpong/')
        anno_dir = os.path.join(BASE_DIR, 'data/PingPong')
        imageset_file = os.path.join(BASE_DIR, 'data/pingpong_fnames_test.txt')
        anno_fn = anno_fn_map['pingpong']
        cls_names = ['player', 'net', 'table']
        cls_counts = [2, 1, 1]
        rel_map = {
            'at' : [('player__1', 'table'), ('player__2', 'table')],
            'on' : [('net', 'table')],
            'playing_pingpong_with' : [('player__1', 'player__2'), ('player__2', 'player__1')]
        }
    elif model_type == 'handshake':
        output_dir = os.path.join(BASE_DIR, 'output/full_runs/handshake/')
        anno_dir = os.path.join(BASE_DIR, 'data/Handshake')
        imageset_file = os.path.join(BASE_DIR, 'data/handshake_fnames_test.txt')
        anno_fn = anno_fn_map['handshake']
        cls_names = ['person', 'handshake']
        cls_counts = [2, 1]
        rel_map = {
            'extending' : [('person__1', 'handshake'), ('person__2', 'handshake')],
            'handshaking' : [('person__1', 'person__2'), ('person__2', 'person__1')]
        }
    else:
        pass
    
    return model_type, gmm_fname, output_dir, anno_dir, imageset_file, anno_fn, rel_map, cls_names, cls_counts



"""
    model: which model to process
    gmm: the .pkl gmm for sampling

python gen_gmm_performance_data.py --model handshake --gmm handshake_gmms.pkl
python gen_gmm_performance_data.py --model dog_walking --gmm dw_gmms_unlog.pkl
python gen_gmm_performance_data.py --model pingpong --gmm pingpong_gmms.pkl
"""
if __name__ == '__main__':
    # read config
    cfg = get_cfg()
    model_type    = cfg[0]
    gmm_fname     = cfg[1]
    output_dir    = cfg[2]
    anno_dir      = cfg[3]
    imageset_file = cfg[4]
    anno_fn       = cfg[5]
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
    
    # generate samples from the gmms
    f = open(gmm_fname, 'rb')
    gmms = cPickle.load(f)
    
    relations = gmms.keys()
    sample_dict = {}
    obj_bbox_dict = {}
    
    avg_iou_results = {}
    full_iou_results = {}
    for image in imageset:
        for relation in relations:
            if relation not in avg_iou_results:
                avg_iou_results[relation] = []
                
            if relation not in full_iou_results:
                full_iou_results[relation] = []
            
            samples = gmms[relation].model.sample(N_SAMPLES) # TODO: parameterize n_samples?
            samples = samples[0]
            sample_dict[relation] = samples
            
            for sro_tuple in rel_map[relation]:
                subj_name = sro_tuple[0]
                obj_name = sro_tuple[1]
                
                sub_box = annotations[image][subj_name]
                gt_obj_bbox = annotations[image][obj_name]
                
                obj_ws = samples[:,2] * sub_box[2]
                obj_xs = sub_box[0] + 0.5 * sub_box[2] - sub_box[2] * samples[:,0] - 0.5 * obj_ws
                obj_hs = samples[:,3] * sub_box[3]
                obj_ys = sub_box[1] + 0.5 * sub_box[3] - sub_box[3] * samples[:,1] - 0.5 * obj_hs
                samp_obj_bboxes = np.vstack((obj_xs, obj_ys, obj_ws, obj_hs)).T
                
                obj_ious = get_ious(gt_obj_bbox, samp_obj_bboxes)
                full_iou_results[relation].append((image, obj_ious))
                
                avg_iou = np.average(obj_ious)
                avg_iou_results[relation].append((image, avg_iou))
    
    # save the sample IoU
    for relation in relations:
        arr = np.array(avg_iou_results[relation], dtype=np.object)
        avg = np.average(arr[:,1])
        print('{}: {:0.4}'.format(relation, avg))
    
    # calculate detection threshold counts
    thresh_list = np.linspace(0., 1., num=11) #TODO: parameterize
    n_images = len(imageset)
    for rel_name in relations:
        ious = np.array(avg_iou_results[rel_name], dtype=np.object)
        max_hits = len(rel_map[rel_name]) * n_images
        detections = []
        for thresh_val in thresh_list:
            hits = np.where(ious[:,1] >= thresh_val)
            n_hits = len(hits[0])
            hit_rate = n_hits / (max_hits * 1.)
            detections.append((thresh_val, hit_rate))
        detections = np.array(detections)
        detection_fname = os.path.join(output_dir, '{}_detections.csv'.format(rel_name))
        np.savetxt(detection_fname, detections, header='threshold, hit_rate', comments='', fmt='%0.2f, %0.3f')
    
    # plot the detections data
    # for plot in plots:
    #    setup plot
    #    save plot
