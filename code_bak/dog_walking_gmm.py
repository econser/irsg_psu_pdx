"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP
"""
def gen_gmms(base_dir = '/home/econser/School/research/', train_file = 'dogwalkingtest_fnames_train.txt', test_file = 'dogwalkingtest_fnames_test.txt'):
    import irsg_utils as iu
    
    data_dir = base_dir + 'data/'
    image_dir = data_dir + 'dog_walking/'
    
    tt_split_dict = get_train_test_splits(data_dir, train_file, test_file)
    relationship_dict = get_dw_annotations(image_dir, tt_split_dict)
    
    gmms = {}
    sk_gmm = train_gmm(relationship_dict['train']['holding'])
    gmms['holding'] = iu.RelationshipParameters(None, None, sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_)
    
    sk_gmm = train_gmm(relationship_dict['train']['attached_to'])
    gmms['attached_to'] = iu.RelationshipParameters(None, None, sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_)
    
    sk_gmm = train_gmm(relationship_dict['train']['is_walking'])
    gmms['is_walking'] = iu.RelationshipParameters(None, None, sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_)

    sk_gmm = train_gmm(relationship_dict['train']['walked_by'])
    gmms['walked_by'] = iu.RelationshipParameters(None, None, sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_)
    
    return gmms



def get_gmms(base_dir = '/home/econser/School/research/', train_file = 'dogwalkingtest_fnames_train.txt', test_file = 'dogwalkingtest_fnames_test.txt'):
    import irsg_utils as iu
    
    data_dir = base_dir + 'data/'
    image_dir = data_dir + 'dog_walking/'
    
    tt_split_dict = get_train_test_splits(data_dir, train_file, test_file)
    relationship_dict = get_dw_annotations(image_dir, tt_split_dict)
    
    gmms = {}
    sk_gmm = train_gmm(relationship_dict['train']['holding'])
    gmms['holding'] = sk_gmm
    
    sk_gmm = train_gmm(relationship_dict['train']['attached_to'])
    gmms['attached_to'] = sk_gmm
    
    sk_gmm = train_gmm(relationship_dict['train']['is_walking'])
    gmms['is_walking'] = sk_gmm
    
    sk_gmm = train_gmm(relationship_dict['train']['walked_by'])
    gmms['walked_by'] = sk_gmm
    
    return gmms, tt_split_dict



def train_gmm(relationship_boxes, n_components=3):
    import irsg_utils as iutl
    import sklearn.mixture as skl
    import sklearn.calibration as cal

    import pdb; pdb.set_trace()
    features = iutl.get_gmm_features(relationship_boxes, in_format='xywh')
    gmm = skl.GaussianMixture(n_components, 'full', verbose='true')
    gmm.fit(features)
    
    #TODO : platt scale
    return gmm



def get_train_test_splits(base_dir, train_filename, test_filename):
    f = open(base_dir + train_filename, 'rb')
    train_files = f.readlines()
    f.close()
    
    f = open(base_dir + test_filename, 'rb')
    test_files = f.readlines()
    f.close()
    
    file_dict = {}
    for filename in train_files:
        file_prefix = filename.split('.')[0]
        file_dict[file_prefix] = 'train'
    
    for filename in test_files:
        file_prefix = filename.split('.')[0]
        file_dict[file_prefix] = 'test'
    
    return file_dict



def get_dw_annotations(directory, test_train_dict):
    import os
    import numpy as np
    
    file_list = os.listdir(directory)
    n_annotations = 0
    for filename in file_list:
        if filename.endswith('.labl'):
            n_annotations += 1
    
    relationship_dict = {}
    
    test_rel_dict = {}
    relationship_dict['test'] = test_rel_dict
    test_rel_dict['holding'] = []
    test_rel_dict['attached_to'] = []
    test_rel_dict['is_walking'] = []
    test_rel_dict['walked_by'] = []
    
    train_rel_dict = {}
    relationship_dict['train'] = train_rel_dict
    train_rel_dict['holding'] = []
    train_rel_dict['attached_to'] = []
    train_rel_dict['is_walking'] = []
    train_rel_dict['walked_by'] = []
    
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        hFile = open(directory + filename, 'rb')
        lines = hFile.readlines()
        
        for line in lines:
            walker_bbox, leash_bbox, dog_bbox = get_dw_boxes(line)
            holding_rel = np.array((walker_bbox, leash_bbox))
            attached_to_rel = np.array((leash_bbox, dog_bbox))
            is_walking_rel = np.array((walker_bbox, dog_bbox))
            walked_by_rel = np.array((dog_bbox, walker_bbox))
            
            file_prefix = filename.split('.')[0]
            image_type = test_train_dict[file_prefix]
            relationship_dict[image_type]['holding'].append(holding_rel)
            relationship_dict[image_type]['attached_to'].append(attached_to_rel)
            relationship_dict[image_type]['is_walking'].append(is_walking_rel)
            relationship_dict[image_type]['walked_by'].append(walked_by_rel)
        
        hFile.close()
    
    relationship_dict['test']['holding'] = np.array(relationship_dict['test']['holding'])
    relationship_dict['test']['attached_to'] = np.array(relationship_dict['test']['attached_to'])
    relationship_dict['test']['is_walking'] = np.array(relationship_dict['test']['is_walking'])
    relationship_dict['test']['walked_by'] = np.array(relationship_dict['test']['walked_by'])
    
    relationship_dict['train']['holding'] = np.array(relationship_dict['train']['holding'])
    relationship_dict['train']['attached_to'] = np.array(relationship_dict['train']['attached_to'])
    relationship_dict['train']['is_walking'] = np.array(relationship_dict['train']['is_walking'])
    relationship_dict['train']['walked_by'] = np.array(relationship_dict['train']['walked_by'])
    
    return relationship_dict



"""
    An annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...
    we assume prototypical dog walking: 1 walker, 1 dog, 1 leash, n other objects
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



def run_test(gmms, data_split='test', relationship='holding'):
    import irsg_utils as iutl
    features = iutl.get_gmm_features(relationship_dict[data_split][relationship])
    features = np.array(features)
    scores = gmms[relationship].score_samples(features)
    return scores    
