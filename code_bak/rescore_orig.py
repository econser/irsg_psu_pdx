def get_data():
    import scipy.io as sio
    
    print 'loading vg_data data...'
    vgd_filename = base_path + 'vg_data.mat'
    vgd = sio.loadmat(vgd_path, struct_as_record=False, squeeze_me=True)
    
    print 'loading potentials data...'
    potentials_filename = base_path + 'vg_data.mat'
    potentials = sio.loadmat(potentials_path, struct_as_record=False, squeeze_me=True)
    
    print("loading platt model data...")
    platt_filename = base_path + "platt_models_struct.mat"
    platt_params = sio.loadmat(platt_filename, struct_as_record=False, squeeze_me=True)
    
    print("loading test queries...")
    query_filename = base_path + "simple_graphs.mat"
    sg_queries = sio.loadmat(query_filename, struct_as_record=False, squeeze_me=True)
    
    return vgd, potentials, platt_params, sg_queries



def score_all_images():
    return None



def score_image():
    return None



def calc_ratk():
    return None



#======================================================================
#
#
def get_object_scores(image_ix, potentials_mat, platt_mat):
"""
Get object detection data from an image
Input:
    image_ix: image number
    potentials_mat: potentials .mat file
    platt_mat: platt model .mat file
Output:
    dict: object name (str) -> boxes (numpy array of [x,y,w,h,p] entries), platt model applied to probabilites
"""
    object_mask = [name[:3] == 'obj' for name in potentials_mat['potentials_s'].classes]
    object_mask = np.array(object_mask)
    object_names = potentials_mat['potentials_s'].classes[object_mask]
    object_detections = get_class_detections(image_ix, potentials_mat, platt_mod, object_names)
    return object_detections    



def get_class_detections(image_ix, potential_data, platt_mod, object_names, verbose=False):
"""Generate box & score values for an image and set of object names
  
Args:
    image_ix (int): the image to generate detections from
    potential_data (.mat data): potential data (holds boxes, scores, and class to index map)
    platt_data (.mat data): holds platt model parameters
    object_names (numpy array of str): the names of the objects to detect
    verbose (bool): default 'False'
Returns:
    dict: object name (str) -> boxes (numpy array)
"""
    n_objects = object_names.shape[0]
    detections = np.empty(n_objects, dtype=np.ndarray)
    
    box_coords = np.copy(potential_data['potentials_s'].boxes[image_ix])
    box_coords[:,2] -= box_coords[:,0]
    box_coords[:,3] -= box_coords[:,1]
    
    class_to_index_keys = potential_data['potentials_s'].class_to_idx.serialization.keys
    class_to_index_vals = potential_data['potentials_s'].class_to_idx.serialization.values
    obj_id_dict = dict(zip(class_to_index_keys, class_to_index_vals))
    
    det_ix = 0
    for o in object_names:
        if o not in obj_id_dict:
            continue
    
    obj_ix = obj_id_dict[o]
    obj_ix -= 1 # matlab is 1-based
    
    a = 1.0
    b = 1.0
    platt_keys = platt_mod['platt_models'].s_models.serialization.keys
    platt_vals = platt_mod['platt_models'].s_models.serialization.values
    platt_dict = dict(zip(platt_keys, platt_vals))
    if o in platt_dict:
        platt_coeff = platt_dict[o]
        a = platt_coeff[0]
        b = platt_coeff[1]
    
    scores = potential_data['potentials_s'].scores[image_ix][:,obj_ix]
    scores = 1.0 / (1.0 + np.exp(a * scores + b))
    
    n_detections = scores.shape[0]
    scores = scores.reshape(n_detections, 1)
    
    class_det = np.concatenate((box_coords, scores), axis=1)
    detections[det_ix] = class_det
    if verbose: print "%d: %s" % (det_ix, o)
    det_ix += 1
    
    return dict(zip(object_names, detections))
