class ModelConfig(object):
    def __init__(self, base_path, definitions, weights):
        self.base_path = base_path
        if base_path[-1] != '/':
            base_path += '/'
        self.definitions = base_path + definitions
        self.weights = base_path + weights



class RelationshipParameters (object):
    def __init__(self, platt_a, platt_b, gmm_weights, gmm_mu, gmm_sigma, gmm=None):
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.gmm_weights = gmm_weights
        self.gmm_mu = gmm_mu
        self.gmm_sigma = gmm_sigma
        self.model = gmm



class IRSGComponents (object):
    def __init__(self, image_filename, boxes, score_matrix, class_names, class_dict, bin_model_dict):
        self.image_filename = image_filename
        self.boxes = boxes
        self.score_matrix = score_matrix
        self.class_names = class_names
        self.class_dict = class_dict
        self.bin_model_dict = bin_model_dict



class RelationComponents (object):
    def __init__(self, image_filename, unary, binary):
        self.image_filename = image_filename
        self.unary_components = unary # list of UnaryComponents
        self.binary_components = binary # dict of 'relation_name' -> RelationshipParameters



class UnaryComponents (object):
    def __init__(self, name, boxes, scores, ious):
        self.name = name
        self.boxes = boxes
        self.scores = scores
        self.ious = ious



"""
    format xyxy is for x0, y0, x1, y1
    format xywh is for x, y, w, h
"""
def get_gmm_features(box_pairs, in_format='xyxy'):
    import numpy as np
    
    subj_axis = 0
    obj_axis = 1
    
    subj_widths = box_pairs[:, subj_axis, 2]
    if in_format == 'xyxy':
        subj_widths = box_pairs[:, subj_axis, 2] - box_pairs[:, subj_axis, 0]
    subj_widths = subj_widths.astype(np.float)
    subj_x_centers = box_pairs[:, subj_axis, 0] + 0.5 * subj_widths
    
    obj_widths = box_pairs[:, obj_axis, 2]
    if in_format == 'xyxy':
        obj_widths = box_pairs[:, obj_axis, 2] - box_pairs[:, obj_axis, 0]
    obj_x_centers  = box_pairs[:, obj_axis, 0] + 0.5 * obj_widths
    
    subj_heights = box_pairs[:, subj_axis, 3]
    if in_format == 'xyxy':
        subj_heights = box_pairs[:, subj_axis, 3] - box_pairs[:, subj_axis, 1]
    subj_heights = subj_heights.astype(np.float)
    subj_y_centers = box_pairs[:, subj_axis, 1] + 0.5 * subj_heights

    obj_heights = box_pairs[:, obj_axis, 3]
    if in_format == 'xyxy':
        obj_heights = box_pairs[:, obj_axis, 3] - box_pairs[:, obj_axis, 1]
    obj_y_centers  = box_pairs[:, obj_axis, 1] + 0.5 * obj_heights
    
    if min(subj_widths) < 0.001:
        import pdb; pdb.set_trace()
    
    relative_x_centers = (subj_x_centers - obj_x_centers) / subj_widths
    relative_y_centers = (subj_y_centers - obj_y_centers) / subj_heights
    relative_heights = obj_heights / subj_heights
    relative_widths = obj_widths / subj_widths
    
    features = np.vstack((relative_x_centers, relative_y_centers, relative_widths, relative_heights))
    features = features.T
    
    return features



"""
    An annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...
    we assume prototypical dog walking: 1 walker, 1 dog, 1 leash, n other objects
    returns: dict of object names and their GT bboxes
"""
def get_dw_boxes(annotation):
    import numpy as np
    
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    objects = tokens[-n_boxes:]
    n_coords = len(tokens) - n_boxes
    boxes = tokens[3 : n_coords]
    
    obj_dict = {}
    
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('dog-walker'):
            obj_dict['walker'] = coords
        elif obj_name.startswith('dog'):
            obj_dict['dog'] = coords
        elif obj_name.startswith('leash'):
            obj_dict['leash'] = coords
        else:
            obj_dict[obj_name.split(' ')[0]] = coords
    
    return obj_dict['walker'], obj_dict['dog'], obj_dict['leash']



"""Convert the mat file binary model storage to a more convienent structure for python
    Input:
        binary model mat file name
    Output:
        none
"""
def get_relationship_models(binary_model_mat_filename, out_filename):
    import scipy.io as sio
    import cPickle
    
    binary_model_mat = sio.loadmat(binary_model_mat_filename, struct_as_record=False, squeeze_me=True)
    
    # create a map from trip_string -> index (e.g. 'shirt_on_man' -> 23)
    trip_ix_root = binary_model_mat['binary_models_struct'].s_triple_str_to_idx.serialization
    trip_to_index_keys = trip_ix_root.keys
    trip_to_index_vals = trip_ix_root.values
    trip_str_dict = dict(zip(trip_to_index_keys, trip_to_index_vals))
    
    # for each trip_str key, pull params from the mat and generate a RelationshipParameters object
    param_list = []
    for trip_str in trip_to_index_keys:
        ix = trip_str_dict[trip_str]
        ix -= 1 # MATLAB uses 1-based indexing here
        platt_params = binary_model_mat['binary_models_struct'].platt_models[ix]
        gmm_params = binary_model_mat['binary_models_struct'].models[ix].gmm_params
        rel_params = RelationshipParameters(platt_params[0], platt_params[1], gmm_params.ComponentProportion, gmm_params.mu, gmm_params.Sigma.T)
        param_list.append(rel_params)
    
    str_to_param_map = dict(zip(trip_to_index_keys, param_list))
    
    with open(out_filename, 'wb') as out_pkl:
        cPickle.dump(str_to_param_map, out_pkl, protocol = cPickle.HIGHEST_PROTOCOL)



"""
calc_iou
expects bbox coords in xywh format
"""
def calc_iou(bbA, bbB, format='xywh'):
    import numpy as np
    
    a, b = []
    if format == 'xywh':
        a = np.array([bbA[0], bbA[1], bbA[0] + bbA[2], bbA[1] + bbA[3]])
        b = np.array([bbB[0], bbB[1], bbB[0] + bbB[2], bbB[1] + bbB[3]])
    else:
        a = bbA
        b = bbB
    
    # generate the jitter angle and amount of IoU
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
""" Gaussian Mixture Model PDF
  
    Given n (number of observations), m (number of features), c (number of components)
  
    Args:
        X : feature vector (n x m)
        mixture : GMM component vector (1 x c)
        mu : mu vectors (c x m)
        sigma : covariance matrices (c x m x m)
  
    returns:
        (n x 1) numpy vector of pdf values
"""
def gmm_pdf(X, mixture, mu, sigma):
    import numpy as np
    from scipy.stats import multivariate_normal as mvn
  
    n_components = len(mixture)
    n_vals = len(X)
  
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
        mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
  
    return mixed_pdf



def get_pregen_components(image_filename, pregen_dir, gmms, classes):
    import numpy as np
    import irsg_utils as iu
    import os.path
    
    unary_components = []
    for obj_class in classes:
        obj_dir = pregen_dir + obj_class + '/'
        if not os.path.isdir(obj_dir):
            continue
        
        fq_filename = obj_dir + image_filename.split('.')[0] + '.csv'
        class_results = np.genfromtxt(fq_filename, delimiter=',')
        
        boxes = class_results[:, 0:4]
        scores = class_results[:, 4]
        ious = class_results[:, 5]
        ocs = iu.UnaryComponents(obj_class, boxes, scores, ious)
        
        unary_components.append(ocs)
    
    rc = iu.RelationComponents(image_filename, unary_components, gmms)
    return rc



def get_boxes(obj_name, descriptors):
    for obj_desc in descriptors:
        if obj_name == obj_desc.name:
            return obj_desc.boxes
    return None



def get_dw_gmms(model_filename='dw_gmms_platt.pkl'):
    import irsg_utils as iu
    import cPickle
    
    f = open('/home/econser/School/research/data/' + model_filename, 'rb')
    gmms = cPickle.load(f)
    return gmms
