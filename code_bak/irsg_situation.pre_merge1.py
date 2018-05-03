from __future__ import print_function
#import matplotlib; matplotlib.use('agg') #when running remotely
import opengm as ogm

import sys
sys.path.append('/home/econser/School/Thesis/external/py-faster-rcnn/lib')

import numpy as np
import irsg_utils as iutl



#===============================================================================
ENERGY_METHODS = ['pgm', 'geo_mean']



#===============================================================================
def gmm_pdf(X, mixture, mu, sigma):
    from scipy.stats import multivariate_normal as mvn
    n_components = len(mixture)
    n_vals = len(X)
    
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
        mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
    
    return mixed_pdf



def get_boxes(obj_name, descriptors):
    for obj_desc in descriptors:
        if obj_name == obj_desc.name:
            return obj_desc.boxes
    return None



def get_pregen_components(image_filename, pregen_dir, gmms, classes):
    import os.path
    
    unary_components = []
    for obj_class in classes:
        obj_dir = os.path.join(pregen_dir, obj_class)
        if not os.path.isdir(obj_dir):
            continue
        
        fq_filename = os.path.join(obj_dir, image_filename.split('.')[0] + '.csv')
        class_results = np.genfromtxt(fq_filename, delimiter=',')
        
        boxes = class_results[:, 0:4]
        scores = class_results[:, 4]
        ious = class_results[:, 5]
        ocs = iutl.UnaryComponents(obj_class, boxes, scores, ious)
        
        unary_components.append(ocs)
    
    rc = iutl.RelationComponents(image_filename, unary_components, gmms)
    return rc



def viz_detections(image_filename, detections, output_filename):
    import matplotlib.pyplot as plt
    import cv2
    
    im = cv2.imread(image_filename)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for object_detection in detections:
        class_name = object_detection[0]
        bbox = object_detection[1]
        score = object_detection[2]
        
        plot_rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=3.5)
        ax.add_patch(plot_rect)
        
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')
    
    #ax.set_title(('{} detections with 'p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(output_filename)
    plt.close()



def get_boxes_and_scores(image_dir, model_file, weights, classes, gmms, cpu_flag, nms_threshold=0.3, imageset=[]):
    import os
    from fast_rcnn.config import cfg
    from fast_rcnn.test import im_detect
    from fast_rcnn.nms_wrapper import nms
    import caffe, cv2
    
    cfg.TEST.HAS_RPN = True
    if cpu_flag == True:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(model_file, weights, caffe.TEST)
    # TODO: get classes from model
    
    filenames = os.listdir(image_dir)
    filenames = filter(lambda f: '.jpg' in f, filenames)
    if len(imageset) > 0:
        filenames = filter(lambda x: x in [f.split('.')[0] for f in filenames], imageset)
    
    rc_list = []
    for filename in filenames:
        print('processing {}'.format(filename))
        file_path = os.path.join(image_dir, filename)
        image = cv2.imread(file_path)
        
        scores, boxes = im_detect(net, image)
        
        unaries = []
        for cls_ix, cls_name in enumerate(classes):
            cls_ix += 1
            cls_boxes = boxes[:, 4*cls_ix:4*(cls_ix + 1)]
            cls_scores = scores[:, cls_ix]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            
            keep = nms(dets, nms_threshold)
            dets = dets[keep, :]
            
            dets[:, 2] = dets[:, 2] - dets[:, 0]
            dets[:, 3] = dets[:, 3] - dets[:, 1]
            dets[:, 0:4] = np.round(dets[:, 0:4])
            
            unary = iutl.UnaryComponents(cls_name, dets[:, 0:4], dets[:, 4], None)
            unaries.append(unary)
        rc = iutl.RelationComponents(filename, unaries, gmms)
        rc_list.append(rc)
    return rc_list



def save_box_and_score_data(rc_list, output_dir):
    for rc in rc_list:
        unaries = rc.unary_components
        for cls in unaries:
            path = os.path.join(output_dir, cls.name)
            if not os.path.exists(path):
                os.makedirs(path)
            filename = rc.image_filename.split('.')[0] + '.csv'
            filepath = os.path.join(path, filename)
            ious = np.ones_like(cls.scores[:, np.newaxis]) * -1.
            csv_data = np.hstack((cls.boxes, cls.scores[:, np.newaxis], ious))
            np.savetxt(filepath, csv_data, fmt='%d, %d, %d, %d, %0.6f, %d')



#===============================================================================
"""
    get the gmm input vector, pdf, and calibrated probability of a bbox pair
"""
def get_relationship_csv(gmm_params, subject_bbox, object_bbox):
    bbox_pair = np.array((subject_bbox, object_bbox))
    bbox_pair = bbox_pair[np.newaxis, :, :]
    
    input_vec = iutl.get_gmm_features(bbox_pair, in_format='xywh')
    pdf_score = iutl.gmm_pdf(input_vec, gmm_params.gmm_weights, gmm_params.gmm_mu, gmm_params.gmm_sigma)
    prob_score = 1. / (1. + np.exp(-(gmm_params.platt_a * pdf_score + gmm_params.platt_b)))
    
    ret_str = '{:0.6f}, {:0.6f}, {}, {}, {}, {}, {}, {}, {}, {}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}'.format(prob_score[0], pdf_score[0], subject_bbox[0], subject_bbox[1], subject_bbox[2], subject_bbox[3], object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3], input_vec[0][0], input_vec[0][1], input_vec[0][2], input_vec[0][3])
    
    return ret_str



def get_top_rel_scores(best_box_ixs, rc, gmm):
    best_bbox_dict = {}
    for ix, box_ix in enumerate(best_box_ixs):
        cls_name = rc.unary_components[ix].name
        bbox = rc.unary_components[ix].boxes[box_ix]
        best_bbox_dict[cls_name] = bbox
    
    best_bbox_keys = best_bbox_dict.keys()
    rel_dict = {}
    
    if set(best_bbox_keys) == set(['dog', 'dog_walker', 'leash']):
        walker_bbox = best_bbox_dict['dog_walker']
        dog_bbox = best_bbox_dict['dog']
        leash_bbox = best_bbox_dict['leash']
        
        rel_str = 'holding'
        subject_bbox = walker_bbox
        object_bbox = leash_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
        rel_str = 'attached_to'
        subject_bbox = leash_bbox
        object_bbox = dog_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
        rel_str = 'walked_by'
        subject_bbox = dog_bbox
        object_bbox = walker_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
    elif set(best_bbox_keys) == set(['player__1', 'player__2', 'table', 'net']):
        player1_bbox = best_bbox_dict['player__1']
        player2_bbox = best_bbox_dict['player__2']
        table_bbox = best_bbox_dict['table']
        net_bbox = best_bbox_dict['net']
        
        rel_str = 'walked_by'
        subject_bbox = player1_bbox
        object_bbox = table_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        subject_bbox = player2_bbox
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
        rel_str = 'on'
        subject_bbox = net_bbox
        object_bbox = table_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
        rel_str = 'playing_pingpong_with'
        subject_bbox = player1_bbox
        object_bbox = player2_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
    elif set(best_bbox_keys) == set(['person__1', 'person__2', 'handshake']):
        p1_bbox = best_bbox_dict['person__1']
        p2_bbox = best_bbox_dict['person__2']
        handshake_bbox = best_bbox_dict['handshake']
        
        rel_str = 'extending'
        subject_bbox = person1_bbox
        object_bbox = handshake_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        subject_bbox = person2_bbox
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
        
        rel_str = 'handshaking'
        subject_bbox = person1_bbox
        object_bbox = person2_bbox
        g = gmm[rel_str]
        rel_dict[rel_str] = []
        rel_csv = get_relationship_csv(g, subject_bbox, object_bbox)
        rel_dict[rel_str].append(rel_csv)
    
    return rel_dict



def get_query_to_detection_map(query, model_components):
    unary_obj_descriptors = model_components.unary_components
    binary_models_dict = model_components.binary_components
    
    fg_to_sg = []
    sg_to_unary = []
    
    for sg_obj_ix, sg_object in enumerate(query.objects):
        img_obj_ix = -1
        for ix, img_obj in enumerate(unary_obj_descriptors):
            if img_obj.name == sg_object.names:
                img_obj_ix = ix
        if img_obj_ix == -1: continue
        fg_to_sg.append(sg_obj_ix)
        sg_to_unary.append(img_obj_ix)
    #return fg_to_sg
    return sg_to_unary



def get_objects_per_class(query, model_components):
    object_names = []
    for obj in query.objects:
        object_names.append(obj.names)
    
    class_names = []
    for cls in model_components.unary_components:
        class_names.append(cls.name)
    
    objects_per_class = []
    for cls_name in class_names:
        n = 0
        for obj_name in object_names:
            if cls_name == obj_name:
                n += 1
        objects_per_class.append(n)
    
    return objects_per_class


#===============================================================================
# GEOMETRIC MEAN SCORING
#
def get_geo_mean_energy(model_components, bboxes_per_class, nms_threshold=0.3):
    from fast_rcnn.nms_wrapper import nms
    
    scores = []
    best_box_ixs = []
    
    for cls_ix, unary in enumerate(model_components.unary_components):
        cls_name = unary.name
        
        orig_ixs = np.arange(len(unary.boxes))
        orig_ixs = orig_ixs[:, np.newaxis]
        detections = np.hstack((unary.boxes, unary.scores[:,np.newaxis], orig_ixs))
        detections[:,2] += detections[:,0]
        detections[:,3] += detections[:,1]
        detections = detections.astype(np.float32)
        
        keep_ixs = nms(detections, nms_threshold)
        detections = detections[keep_ixs]
        sorted_ixs = np.argsort(detections[:,4])
        # TODO: make sure that there are enough boxes left
        cls_objects_to_score = bboxes_per_class[cls_ix]
        for obj_ix in range(0, cls_objects_to_score):
            bbox_ix = sorted_ixs[::-1][obj_ix]
            orig_ix = detections[bbox_ix, 5]
            best_box_ixs.append(int(orig_ix))
            scores.append(detections[bbox_ix, 4])
    
    # calc geometric mean and convert to an energy
    scores = np.array(scores)
    geo_mean = scores.prod() ** (1.0 / len(scores))
    energy = np.exp(-geo_mean)
    return energy, best_box_ixs



#===============================================================================
# FACTOR GRAPH GENERATION AND INFERENCE CALLS
#
"""
    Generate a factor graph from a query structure the model components
"""
def gen_factor_graph(query, model_components, verbose=False, use_scaling=True):
    import itertools
    
    verbose_tab = '  '
    do_unary_xform = True
    do_binary_xform = True
    
    unary_obj_descriptors = model_components.unary_components
    binary_models_dict = model_components.binary_components
    
    n_vars = []
    fg_to_sg = []
    sg_to_unary = []
    fg_functions = []
    
    #---------------------------------------------------------------------------
    # GENERATE UNARY FUNCTIONS
    for sg_obj_ix, sg_object in enumerate(query.objects):
        if verbose: print('{}using model for object "{}"'.format(verbose_tab, sg_object.names))
        img_obj_ix = -1
        for ix, img_obj in enumerate(unary_obj_descriptors):
            if img_obj.name == sg_object.names:
                img_obj_ix = ix
        if img_obj_ix == -1: continue
        n_labels = len(unary_obj_descriptors[img_obj_ix].boxes)
        n_vars.append(n_labels)
        fg_to_sg.append(sg_obj_ix)
        sg_to_unary.append(img_obj_ix)
    gm = ogm.gm(n_vars, operator='adder')
    
    # add unary functions to gm
    for ix in fg_to_sg:
        unary_ix = sg_to_unary[ix]
        scores = np.copy(unary_obj_descriptors[unary_ix].scores)
        if do_unary_xform:
            scores += np.finfo(np.float).eps
            scores = -np.log(scores)
        fn_id = gm.addFunction(scores)
        fg_functions.append((1, fn_id, [ix]))
    
    #---------------------------------------------------------------------------
    # GENERATE BINARY FUNCTIONS
    
    # prep the relationships
    bin_relations = query.binary_triples
    relationships = []
    if isinstance(bin_relations, np.ndarray):
        for rel in bin_relations:
            relationships.append(rel)
    else:
        relationships.append(bin_relations)
    
    # generate a function for each relationship
    for rel in relationships:
        # get object boxes and generate box pairs
        subject_name = query.objects[rel.subject].names
        object_name = query.objects[rel.object].names
        
        # specific: <subject_<relationship>_<object>
        specific_rel = subject_name + '_'
        specific_rel += rel.predicate.replace(' ', '_')
        specific_rel += '_' + object_name
        
        # wildcard: *_<relationship>_*
        wildcard_rel = rel.predicate.replace(' ', '_')
        #wildcard_rel = '*_'
        #wildcard_rel += rel.predicate.replace(' ', '_')
        #wildcard_rel += '_*'
        
        # get the model string
        relationship_key = ''
        if specific_rel in binary_models_dict:
            if verbose: print('{}using relationship model for "{}"'.format(verbose_tab, specific_rel))
            relationship_key = specific_rel
        elif wildcard_rel in binary_models_dict:
            if verbose: print('{}no relationship model for "{}", using "{}"'.format(verbose_tab, specific_rel, wildcard_rel))
            relationship_key = wildcard_rel
        else:
            if verbose: print('{}no relationship models for "{}" or "{}", skipping relationship'.format(verbose_tab, specific_rel, wildcard_rel))
            continue
        
        # generate box pairs
        sub_boxes = get_boxes(subject_name, unary_obj_descriptors)
        n_sub_boxes = len(sub_boxes)
        obj_boxes = get_boxes(object_name, unary_obj_descriptors)
        n_obj_boxes = len(obj_boxes)
        
        box_pairs = np.array([x for x in itertools.product(sub_boxes, obj_boxes)])
        gmm_features = iutl.get_gmm_features(box_pairs, in_format='xywh')
        params = binary_models_dict[relationship_key]
        
        # run the features through the relationship model
        scores = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        if do_binary_xform:
            #scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
            if use_scaling and params.platt_a is not None and params.platt_b is not None:
                #scores = np.log(scores)
                scores = 1. / (1. + np.exp(-(params.platt_a * scores + params.platt_b)))
            scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
            scores = -np.log(scores)
        
        bin_fns = np.reshape(scores, (n_sub_boxes, n_obj_boxes))
        
        sub_var_ix = fg_to_sg[rel.subject]
        obj_var_ix = fg_to_sg[rel.object]
        var_ixs = [sub_var_ix, obj_var_ix]
        
        if obj_var_ix < sub_var_ix:
            bin_fns = bin_fns.T
            var_ixs = [obj_var_ix, sub_var_ix]
        
        fid = gm.addFunction(bin_fns)
        fg_functions.append((2, fid, var_ixs))
    
    #---------------------------------------------------------------------------
    # ADD FUNCTIONS TO GM
    for fn_tup in fg_functions:
        if fn_tup[0] == 1:
            gm.addFactor(fn_tup[1], fn_tup[2][0])
        else:
            gm.addFactor(fn_tup[1], fn_tup[2])
    
    return gm, sg_to_unary



""" Run belief propagation on the provided graphical model
    returns:
    energy (float): the energy of the GM
    var_indices (numpy array): indices for the best label for each variable
"""
def do_inference(gm, n_steps=120, damping=0., convergence_bound=0.001, verbose=False):
    import opengm as ogm
    ogm_params = ogm.InfParam(steps=n_steps, damping=damping, convergenceBound=convergence_bound)
    infr_output = ogm.inference.BeliefPropagation(gm, parameter=ogm_params)
    
    if verbose:
        infr_output.infer(infr_output.verboseVisitor())
    else:
        infr_output.infer()
    
    detected_vars = []
    for i in range(0, gm.numberOfVariables):
        if gm.numberOfLabels(i) > 1:
            detected_vars.append(i)
    
    #infr_marginals = infr_output.marginals(detected_vars)
    #infr_marginals = np.exp(-infr_marginals)
    
    infr_best_match = infr_output.arg()
    infr_energy = infr_output.value()
    
    return infr_energy, infr_best_match, None#infr_marginals



#===============================================================================
def save_all_binary_probs(query, query_to_model_map, best_box_ixs, rc, csv_prefix):
    # get the unaries
    unary_names = []
    for uc in rc.unary_components:
        filename = '{}{}.csv'.format(csv_prefix, uc.name)
        np_out = np.hstack((uc.boxes, uc.scores[:, np.newaxis]))
        np.savetxt(filename, np_out, fmt='%d, %d, %d, %d, %0.6f')    
    
    # get the binary factor scores
    bin_scores = get_binary_scores(query, query_to_model_map, rc)
    return bin_scores
    # TODO: save overall detail
    # TODO: save best box & score for each unary, best score for each relationship



"""
    This is for saving all relationship prob scores to a set of files
"""
def get_binary_scores(query, qry_to_model_map, model_components):
    import itertools
    
    use_scaling = True
    do_binary_xform = True
    binary_models_dict = model_components.binary_components
    unary_obj_descriptors = model_components.unary_components
    bin_relations = query.binary_triples
    relationships = []
    
    if isinstance(bin_relations, np.ndarray):
        for rel in bin_relations:
            relationships.append(rel)
    else:
        relationships.append(bin_relations)
    
    bin_fn_list = []
    for rel in relationships:
        # get object boxes and generate box pairs
        subject_name = query.objects[rel.subject].names
        object_name = query.objects[rel.object].names
        
        # specific: <subject_<relationship>_<object>
        specific_rel = subject_name + '_'
        specific_rel += rel.predicate.replace(' ', '_')
        specific_rel += '_' + object_name
        
        # wildcard: *_<relationship>_*
        wildcard_rel = rel.predicate.replace(' ', '_')
        
        # get the model string
        relationship_key = ''
        if specific_rel in binary_models_dict:
            relationship_key = specific_rel
        elif wildcard_rel in binary_models_dict:
            relationship_key = wildcard_rel
        else:
            continue
        
        # generate box pairs
        sub_boxes = get_boxes(subject_name, unary_obj_descriptors)
        n_sub_boxes = len(sub_boxes)
        obj_boxes = get_boxes(object_name, unary_obj_descriptors)
        n_obj_boxes = len(obj_boxes)
        
        box_pairs = np.array([x for x in itertools.product(sub_boxes, obj_boxes)])
        gmm_features = iutl.get_gmm_features(box_pairs, in_format='xywh')
        params = binary_models_dict[relationship_key]
        
        # run the features through the relationship model
        scores = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        if do_binary_xform:
            scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
            if use_scaling and params.platt_a is not None and params.platt_b is not None:
                #scores = np.log(scores)
                scores = 1. / (1. + np.exp(-(params.platt_a * scores + params.platt_b)))
            #scores = -np.log(scores)
        
        bin_fns = np.reshape(scores, (n_sub_boxes, n_obj_boxes))
        
        #TODO: fix from here on
        sub_var_ix = qry_to_model_map[rel.subject]
        obj_var_ix = qry_to_model_map[rel.object]
        var_ixs = [sub_var_ix, obj_var_ix]
        
        if obj_var_ix < sub_var_ix:
            bin_fns = bin_fns.T
            var_ixs = [obj_var_ix, sub_var_ix]
        
        bin_fn_list.append((sub_var_ix, subject_name, obj_var_ix, object_name, relationship_key, bin_fns))
    
    bf = bin_fn_list[0][5]
    box_ixs = np.unravel_index(np.argmax(bf), bf.shape)
    import pdb; pdb.set_trace()
    bin_results = get_rel_data(model_components, (2,2), bf)
    
    import pdb; pdb.set_trace()
    return bin_results



def get_rel_data(model_components, cls_ixs, bin_results, min_unary=0.00):
    raveled = np.argsort(bin_results.ravel())
    unraveled = np.unravel_index(raveled, bin_results.shape)
    sort_ixs = np.dstack(unraveled)[0][::-1]
    
    rel_results = []
    for rank, ix in enumerate(sort_ixs):
        u1 = model_components.unary_components[cls_ixs[0]].scores[ix[0]]
        if u1 < min_unary:
            u1 = min_unary
        u2 = model_components.unary_components[cls_ixs[1]].scores[ix[1]]
        if u2 < min_unary:
            u2 = min_unary
        b12 = bin_results[ix[0]][ix[1]]
        results = (ix[0], u1, ix[1], u2, b12, u1 * u2 * b12)
        rel_results.append(results)
    return np.array(rel_results)



#===============================================================================
def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Scene graph situations')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--cfg', dest='cfg_file', help='configuration param file (.yml)')
    parser.add_argument('--b', dest='batch', help='configuration batch to run')
    
    args = parser.parse_args()
    return args



def check_cfg(cfg):
    query_string = cfg.get('query_string')
    query_function = cfg.get('query_function')
    
    gmm_filename = cfg.get('gmm_filename')
    data_dir = cfg.get('data_dir')
    
    precalc_dir = cfg.get('precalc_dir')
    image_dir = cfg.get('image_dir')
    image_set_file = cfg.get('image_set_file')
    model_file = cfg.get('model_file')
    weights = cfg.get('weights')
    class_file = cfg.get('class_file')
    
    energy_output_dir = cfg.get('energy_output_dir')
    energy_filename = cfg.get('energy_filename')
    rel_filefmt = cfg.get('top_rel_filefmt')
    
    viz_output_dir = cfg.get('viz_output_dir')
    viz_file_format = cfg.get('viz_file_format')
    
    misconfig = False
    if query_string is None:
        print('No query_string specified')
        misconfig = True
    
    if query_function is None:
        print('No query_function specified')
        misconfig = True
    elif not iqg.querygen_fns.has_key(query_function):
        print('Unknown query function: "{}"'.format(query_function))
        misconfig = True
    
    if data_dir is None:
        print('No data_dir specified')
        misconfig = True
    elif not os.path.exists(data_dir):
        print('Could not find data directory: "{}"'.format(data_dir))
        misconfig = True
    
    if gmm_filename is None:
        print('No gmm_filename specified')
        misconfig = True
    if gmm_filename is None or not os.path.isfile(os.path.join(data_dir, gmm_filename)):
        print('Could not find GMM file: "{}"'.format(gmm_filename))
        misconfig = True
    
    caffe_good = model_file is not None and weights is not None and class_file is not None
    if caffe_good:
        return False
    
    precalc_bad = precalc_dir is None
    imagedir_bad = image_dir is not None and not caffe_good
    
    if precalc_bad and imagedir_bad:
        print('Input data misconfiguration.  precalc_dir, image_dir/model/weights/classes, and image_list/model/weights/classes are all missing')
        misconfig = True
    
    if energy_output_dir is None and viz_output_dir is None:
        print('Output dir misconfiguration, both energy and viz dirs are missing')
        misconfig = True
    
    return misconfig



"""
python irsg_situation.py --cpu --cfg irsg.yml --b DW_PREGEN
python irsg_situation.py --cpu --cfg irsg.yml --b DW_TEST_IMAGES
"""
if __name__ == '__main__':
    import os.path
    import sys
    import yaml
    import cPickle
    import irsg_querygen as iqg
    
    #save_top_rel_scores = True # expermiental feature, needs to be parameterized
    save_pgm_data = True # expermiental feature, needs to be parameterized
    
    args = parse_args()
    
    # read and check the config file
    f = open(args.cfg_file, 'rb')
    full_cfg = yaml.load(f)
    cfg = full_cfg[args.batch]
    misconfig = check_cfg(cfg)
    
    # bail out if cfg doesn't make sense
    if misconfig:
        sys.exit(0)
    
    # pull the cfg vars
    cpu_flag = args.cpu_mode
    query_string = cfg.get('query_string')
    query_fn = cfg.get('query_function')
    gmm_filename = cfg.get('gmm_filename')
    
    # pull data dir
    data_dir = cfg.get('data_dir')
    
    # pull input dir
    precalc_dir = cfg.get('precalc_dir')
    image_dir = cfg.get('image_dir')
    imageset_file = cfg.get('imageset_file')
    model_file = cfg.get('model_file')
    weights = cfg.get('weights')
    class_file = cfg.get('class_file')
    
    mode = ''
    class_list = []
    caffe_ready = model_file is not None and weights is not None and class_file is not None
    if precalc_dir is not None:
        mode = 'precalc'
        class_list = os.listdir(precalc_dir)
    elif image_dir is not None and caffe_ready:
        mode = 'image_dir'
        
        # figure out the classes form the class list file
        f = open(os.path.join(data_dir, class_file), 'rb')
        class_list = f.readlines()
        class_list = [cls.rstrip('\n') for cls in class_list]
    else:
        sys.exit(0)
    
    # see if there's a fileset
    imageset = []
    imageset_file = cfg.get('imageset_file')
    if imageset_file is not None:
        imageset_filepath = os.path.join(data_dir, imageset_file)
        if os.path.exists(imageset_filepath):
            f = open(imageset_filepath)
            imageset = f.readlines()
            imageset = [fname.rstrip('\n') for fname in imageset]
            imageset = [fname.rstrip('\r') for fname in imageset]
            imageset = [fname.split('.')[0] for fname in imageset]
    
    # unpickle the gmms
    gmms = None
    if gmm_filename is not None:
        gmm_path = os.path.join(data_dir, gmm_filename)
        f = open(gmm_path, 'rb')
        gmms = cPickle.load(f)
        f.close()
    
    # generate the query
    query = None
    if query_string is not None:
        query = iqg.querygen_fns[query_fn](query_string)
    
    # get energy calculation method (default is pgm)
    energy_method = cfg.get('energy_method', 'pgm')
    if energy_method not in ENERGY_METHODS:
        print('unknown energy calculation method "{}", expected {}'.format(energy_method, ENERGY_METHODS))
    
    # determine which outputs to generate
    energy_output_dir = cfg.get('energy_output_dir')
    energy_filename = cfg.get('energy_filename')
    do_energy = False
    if energy_output_dir is not None and energy_filename is not None:
        do_energy = True
    
    rel_filefmt = cfg.get('top_rel_filefmt')
    save_top_rel_scores = rel_filefmt is not None
    
    viz_output_dir = cfg.get('viz_output_dir')
    viz_file_format = cfg.get('viz_file_format')
    do_viz = False
    if viz_output_dir is not None and viz_file_format is not None:
        do_viz = True
        if not os.path.exists(viz_output_dir):
            os.makedirs(viz_output_dir)
    
    box_and_score_savedir = cfg.get('box_and_score_output_dir')
    
    # generate the bbox and scores for the images
    rc_list = []
    if mode == 'precalc':
        filenames = os.listdir(os.path.join(precalc_dir, class_list[0]))
        for filename in filenames:
            if len(imageset) > 0 and filename.split('.')[0] not in imageset:
                continue
            pgc = get_pregen_components(filename, precalc_dir, gmms, class_list)
            rc_list.append(pgc)
    elif mode == 'image_dir':
        rc_list = get_boxes_and_scores(image_dir, model_file, weights, class_list, gmms, cpu_flag, nms_threshold=0.5, imageset=imageset)
    elif mode == 'image_list':
        print('Image List not yet implented, try image_dir')
        sys.exit(1)
    
    # process all items
    if box_and_score_savedir is not None:
        save_box_and_score_data(rc_list, box_and_score_savedir)
    
    energy_results = []
    energy_file_handle = None
    if do_energy:
        energy_pathname = os.path.join(energy_output_dir, energy_filename)
        if not os.path.exists(energy_output_dir):
            os.makedirs(energy_output_dir)
        energy_file_handle = open(energy_pathname, 'wb')
        energy_file_handle.write('file, energy\n')
    
    if query is None:
        sys.exit(1)
    
    viz_results = []
    top_rel_score_results = {}
    for rc in rc_list:
        print('calculating similarity for {}'.format(rc.image_filename))
        energy = None
        best_box_ixs = None
        query_to_model_map = None
        if energy_method == 'pgm':
        # generate factor graph and run inference
            pgm, query_to_model_map = gen_factor_graph(query, rc)
            energy, best_box_ixs, marginals = do_inference(pgm)
            if save_top_rel_scores:
                rel_dict = get_top_rel_scores(best_box_ixs, rc, gmms)
                for key in rel_dict.keys():
                    if key not in top_rel_score_results:
                        top_rel_score_results[key] = []
                    for score_str in rel_dict[key]:
                        top_rel_score_results[key].append((rc.image_filename, score_str))
            if save_pgm_data:
                csv_prefix = energy_output_dir + rc.image_filename.split('.')[0] + '_'
                bin_results = save_all_binary_probs(query, query_to_model_map, best_box_ixs, rc, csv_prefix)
        elif energy_method == 'geo_mean':
            objects_per_class = get_objects_per_class(query, rc)
            query_to_model_map = get_query_to_detection_map(query, rc)
            energy, best_box_ixs = get_geo_mean_energy(rc, objects_per_class)
        energy_results.append((rc.image_filename, energy))
        if do_energy:
            energy_file_handle.write('{}, {}\n'.format(rc.image_filename, energy))
        
        # generate viz plot, if so configured
        if do_viz:
            viz_data = []
            for query_ix, unary_ix in enumerate(query_to_model_map):
                cls_unaries = rc.unary_components[unary_ix]
                best_ix = best_box_ixs[query_ix]
                best_box = cls_unaries.boxes[best_ix]
                best_score = cls_unaries.scores[best_ix]
                cls_name = cls_unaries.name
                bbox_viz = (cls_name, best_box, best_score)
                viz_data.append(bbox_viz)
            
            image_prefix = rc.image_filename.split('.')[0]
            viz_output_filename = viz_file_format.format(image_prefix)
            viz_output_filename = '{:06.3f}_{}'.format(energy, viz_output_filename)
            viz_image_filepath = os.path.join(viz_output_dir, viz_output_filename)
            image_filepath = os.path.join(image_dir, rc.image_filename)
            if image_filepath.endswith('.csv'):
                #TODO: parameterize the extension
                image_filepath = image_filepath.split('.')[0] + '.jpg'
            viz_detections(image_filepath, viz_data, viz_image_filepath)
    
    # generate the relationship score file
    if save_top_rel_scores:
        for key in top_rel_score_results.keys():
            rel_filename = rel_filefmt.format('rel_'+key)
            fq_outfile = os.path.join(energy_output_dir, rel_filename)
            f = open(fq_outfile, 'wb')
            f.write('filename, probability, pdf, sub_x, sub_y, sub_w, sub_h, obj_x, obj_y, obj_w, obj_h, rel_x, rel_y, rel_w, rel_h\n')
            for line in top_rel_score_results[key]:
                f.write('{}, {}\n'.format(line[0], line[1]))
            f.close()
    # generate the energy file, if so configured
    if do_energy:
        energy_file_handle.close()



