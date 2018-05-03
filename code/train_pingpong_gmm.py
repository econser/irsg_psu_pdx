"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP

import cPickle
import train_pingpong_gmm as ppg
g = ppg.gen_gmms()
out_filename = '/home/econser/research/data/pingpong_gmms_l1.pkl'
f = open(out_filename, 'wb')
cPickle.dump(g, f, cPickle.HIGHEST_PROTOCOL)
f.close()
"""
rel_str = ['on', 'at', 'playing_pingpong_with']
cls_name = ['player__1', 'player__2', 'net', 'table']

def gen_gmms(base_dir = '/home/econser/research/', train_file = 'pingpong_fnames_train.txt', test_file = 'pingpong_fnames_test.txt', n_components=3):
    import irsg_utils as iu
    
    data_dir = base_dir + 'data/'
    image_dir = data_dir + 'PingPong/'
    
    tt_split_dict = get_train_test_splits(data_dir, train_file, test_file)
    relationship_dict = get_annotations(image_dir, tt_split_dict)
    
    train_rel = []
    train_rel.append(relationship_dict['train'][rel_str[0]])
    train_rel.append(relationship_dict['train'][rel_str[1]])
    train_rel.append(relationship_dict['train'][rel_str[2]])

    neg_rel = []
    neg_rel.append(relationship_dict['neg_train'][rel_str[0]])
    neg_rel.append(relationship_dict['neg_train'][rel_str[1]])
    neg_rel.append(relationship_dict['neg_train'][rel_str[2]])
    
    gmms = {}
    sk_gmm, platt = train_gmm(train_rel[0], neg_rel[0], n_components)
    gmms[rel_str[0]] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(train_rel[1], neg_rel[1], n_components)
    gmms[rel_str[1]] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(train_rel[2], neg_rel[2], n_components)
    gmms[rel_str[2]] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    return gmms



def train_gmm(pos_boxes, neg_boxes=None, n_components=3):
    import numpy as np
    import irsg_utils as iutl
    import sklearn.mixture as skl
    import sklearn.linear_model as lm
    
    pos_features = iutl.get_gmm_features(pos_boxes, in_format='xywh')
    n_pos = len(pos_features)
    pos_labels = np.ones(n_pos) #* ((n_pos + 1.)/(n_pos + 2.))
    
    neg_features = iutl.get_gmm_features(neg_boxes, in_format='xywh')
    n_neg = len(neg_features)
    neg_labels = np.zeros(n_neg) #* (1. / (n_neg + 2.))
    
    all_features = np.concatenate((pos_features, neg_features))
    all_labels = np.concatenate((pos_labels, neg_labels))
    
    gmm = skl.GaussianMixture(n_components, 'full', verbose='true', max_iter=500, n_init=50, tol=1e-6, init_params='random')
    gmm.fit(pos_features)
    
    # test X and Y fit for GMM
    #gmm_ = skl.GaussianMixture(n_components, 'full', verbose='true', n_init=25, tol=1e-6)
    #gmm_.fit(all_features, all_labels)
    
    # use GMM scoring
    #pos_scores = gmm.score_samples(pos_features)
    #neg_scores = gmm.score_samples(neg_features)
    
    pos_scores = iutl.gmm_pdf(pos_features, gmm.weights_, gmm.means_, gmm.covariances_)
    neg_scores = iutl.gmm_pdf(neg_features, gmm.weights_, gmm.means_, gmm.covariances_)
    all_scores = np.concatenate((pos_scores, neg_scores))
    all_log_scores = all_scores#np.log(all_scores + np.finfo(np.float).eps)
    
    from sklearn.utils import shuffle
    
    shuff_scores, shuff_labels = shuffle(all_log_scores, all_labels)
    platt_cal = lm.LogisticRegression(penalty='l1')
    platt_cal.fit(shuff_scores.reshape(-1,1), shuff_labels)
    platt_params = (platt_cal.coef_[0][0], platt_cal.intercept_[0])
    
    return gmm, platt_params



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



def get_annotations(directory, test_train_dict):
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
    test_rel_dict[rel_str[0]] = []
    test_rel_dict[rel_str[1]] = []
    test_rel_dict[rel_str[2]] = []
    
    train_rel_dict = {}
    relationship_dict['train'] = train_rel_dict
    train_rel_dict[rel_str[0]] = []
    train_rel_dict[rel_str[1]] = []
    train_rel_dict[rel_str[2]] = []

    neg_rel_dict = {}
    relationship_dict['neg_train'] = neg_rel_dict
    neg_rel_dict[rel_str[0]] = []
    neg_rel_dict[rel_str[1]] = []
    neg_rel_dict[rel_str[2]] = []
    
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        hFile = open(directory + filename, 'rb')
        lines = hFile.readlines()
        
        for line in lines:
            file_prefix = filename.split('.')[0]
            image_type = test_train_dict[file_prefix]
            
            bboxes = get_bboxes(line)
            
            # -on- relation
            b = (bboxes[cls_name[2]], bboxes[cls_name[3]])
            b = np.array(b)
            relationship_dict[image_type][rel_str[0]].append(b)

            if image_type == 'train':
                b = (bboxes[cls_name[3]], bboxes[cls_name[2]])
                b = np.array(b)
                relationship_dict['neg_train'][rel_str[0]].append(b)
            
            # -at- relation (for p1 & p2)
            b = (bboxes[cls_name[0]], bboxes[cls_name[3]])
            b = np.array(b)
            relationship_dict[image_type][rel_str[1]].append(b)
            
            b = (bboxes[cls_name[1]], bboxes[cls_name[3]])
            b = np.array(b)
            relationship_dict[image_type][rel_str[1]].append(b)

            if image_type == 'train':
                b = (bboxes[cls_name[3]], bboxes[cls_name[0]])
                b = np.array(b)
                relationship_dict['neg_train'][rel_str[1]].append(b)

                b = (bboxes[cls_name[3]], bboxes[cls_name[1]])
                b = np.array(b)
                relationship_dict['neg_train'][rel_str[1]].append(b)            
            
            # -playing_pingpong_with- relation (for p1 and p2)
            b = (bboxes[cls_name[0]], bboxes[cls_name[1]])
            b = np.array(b)
            relationship_dict[image_type][rel_str[2]].append(b)
            
            b = (bboxes[cls_name[1]], bboxes[cls_name[0]])
            b = np.array(b)
            relationship_dict[image_type][rel_str[2]].append(b)

            if image_type == 'train':
                b = (bboxes[cls_name[0]], bboxes[cls_name[0]])
                b = np.array(b)
                relationship_dict['neg_train'][rel_str[2]].append(b)

                b = (bboxes[cls_name[1]], bboxes[cls_name[1]])
                b = np.array(b)
                relationship_dict['neg_train'][rel_str[2]].append(b)
        hFile.close()
    
    for batch_type in ['test', 'train', 'neg_train']:
        for rel in rel_str:
            relationship_dict[batch_type][rel] = np.array(relationship_dict[batch_type][rel])
    
    return relationship_dict



def gen_box_set(pos_boxes, neg_boxes):
    import numpy as np
    neg = np.copy(neg_boxes)
    neg[:,1] = 0.0
    box_set = np.vstack((pos_boxes, neg))
    return box_set




"""
    An annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...
"""
def get_bboxes(annotation):
    import numpy as np
    
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    objects = tokens[-n_boxes:]
    n_coords = len(tokens) - n_boxes
    boxes = tokens[3 : n_coords]
    
    obj_dict = {}
    
    player_instance = 1
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('player-'):
            obj_dict['player__{}'.format(player_instance)] = coords.copy()
            player_instance += 1
        elif obj_name.startswith('net'):
            obj_dict['net'] = coords.copy()
        elif obj_name.startswith('table'):
            obj_dict['table'] = coords.copy()
        else:
            obj_dict[obj_name.split(' ')[0]] = coords.copy()
    
    return obj_dict



def run_test(gmms, data_split='test', relationship='holding'):
    import irsg_utils as iutl
    features = iutl.get_gmm_features(relationship_dict[data_split][relationship])
    features = np.array(features)
    scores = gmms[relationship].score_samples(features)
    return scores    
