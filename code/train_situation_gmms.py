"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP

import train_situation_gmms as g
gmms = g.gen_gmms('situation_config.json')
g.save(gmms, '/home/econser/research/data/situation_gmms.pkl')


"""
import json as j

rel_str = ['extending', 'handshaking']
cls_name = ['person__1', 'person__2', 'handshake']

def gen_gmms(situation_cfg, annotation_dir = '/home/econser/research/data/Handshake', train_file = '/home/econser/research/data/handshake_fnames_train.txt', n_components=3):
    import irsg_utils as iu

    # pull important info from the situation cfg
    f_cfg = load(situation_cfg)
    j_cfg = j.load(f_cfg)
    f_cfg.close()
    situation_name = j['situation_name']
    rel_names = []
    for n in j['relationships']:
        rel_names.append(n['name'])
    object_names = j['object_names']

    # format the training fnames
    training_fnames = get_training_data(train_file)
    relationship_dict = get_annotations(situation_name, annotation_dir, training_fnames, relation_names)

    train_rel = []
    train_rel.append(relationship_dict['train'][rel_str[0]])
    train_rel.append(relationship_dict['train'][rel_str[1]])
    
    gmms = {}
    sk_gmm, platt = train_gmm(train_rel[0], train_rel[1], n_components)
    gmms[rel_str[0]] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(train_rel[1], train_rel[0], n_components)
    gmms[rel_str[1]] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    return gmms



def get_annotations_(situation_name, annotation_dir, training_fnames, relationship_names):
    import os
    import numpy as np
    
    file_list = os.listdir(directory)
    n_annotations = 0
    for filename in file_list:
        if filename.endswith('.labl'):
            n_annotations += 1
    
    relationship_dict = {}
    for rel_name in relationship_names:
        relationship_dict[rel_name] = []
    
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        # parse the annotation
        # append np.array(subj_bbox, obj_bbox) to the appropriate relationship
    # convert each relation bbox list to array

    return relationship_dict
        

        
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
    all_log_scores = np.log(all_scores + np.finfo(np.float).eps)
    
    from sklearn.utils import shuffle
    
    shuff_scores, shuff_labels = shuffle(all_log_scores, all_labels)
    platt_cal = lm.LogisticRegression(penalty='l1')
    platt_cal.fit(shuff_scores.reshape(-1,1), shuff_labels)
    platt_params = (platt_cal.coef_[0][0], platt_cal.intercept_[0])
    
    return gmm, platt_params



def get_training_data(train_filename):
    f = open(train_filename, 'rb')
    train_files = f.readlines()
    f.close()
    
    file_dict = {}
    for filename in train_files:
        file_prefix = filename.split('.')[0]
        file_dict[file_prefix] = 'train'
    
    return file_dict



def get_annotations_(annotation_dir, training_dict, relationships):
    import os
    import numpy as np
    
    file_list = os.listdir(directory)
    n_annotations = 0
    for filename in file_list:
        if filename.endswith('.labl'):
            n_annotations += 1
    
    relationship_dict = {}
    for rel_name in relationship_names:
        relationship_dict[rel_name] = []
    
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        fname = os.path.join(annotation_dir, filename)
        f = open(fname, 'rb')
        lines = f.readlines()
        anno = lines[0]
        
        file_prefix = filename.split('.')[0]
        image_type = test_train_dict[file_prefix]
        
        bboxes = get_bboxes(line)
        
        # -extending- relation (for p1 & p2)
        b = (bboxes[cls_name[0]], bboxes[cls_name[2]])
        b = np.array(b)
        relationship_dict[image_type][rel_str[0]].append(b)
        
        b = (bboxes[cls_name[1]], bboxes[cls_name[2]])
        b = np.array(b)
        relationship_dict[image_type][rel_str[0]].append(b)
        
        # -handshaking- relation (for p1 and p2)
        b = (bboxes[cls_name[0]], bboxes[cls_name[1]])
        b = np.array(b)
        relationship_dict[image_type][rel_str[1]].append(b)
        
        b = (bboxes[cls_name[1]], bboxes[cls_name[0]])
        b = np.array(b)
        relationship_dict[image_type][rel_str[1]].append(b)
    
    for batch_type in ['test', 'train']:
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
    
    person_instance = 1
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('person-'):
            obj_dict['person__{}'.format(person_instance)] = coords.copy()
            person_instance += 1
        elif obj_name.startswith('handshake'):
            obj_dict['handshake'] = coords.copy()
        else:
            obj_dict[obj_name.split(' ')[0]] = coords.copy()
    
    return obj_dict



def run_test(gmms, data_split='test', relationship='extending'):
    import irsg_utils as iutl
    features = iutl.get_gmm_features(relationship_dict[data_split][relationship])
    features = np.array(features)
    scores = gmms[relationship].score_samples(features)
    return scores    



def save_gmms(gmms, fname):
    import cPickle
    f = open(out_filename, 'wb')
    cPickle.dump(gmms, f, cPickle.HIGHEST_PROTOCOL)
    f.close()



#===============================================================================
def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Binary Relation generation')
    parser.add_argument('--cfg', dest='json_cfg')
    args = parser.parse_args()
    
    json_fname = args.json_cfg
    f = open(json_fname, 'rb')
    j = json.load(f)
    f.close()

    object_names = j['object_names']
    relationships = j['relationships']

    # training images (training_dir or training_files)
    training_fnames = []
    if 'training_filenames' in j:
        with open(j['training_filenames'], 'rb') as f:
            training_fnames = f.readlines()
            training_fnames = [fname.rstrip('\n') for fname in training_fnames]
            training_fnames = [fname.rstrip('\r') for fname in training_fnames]
            training_fnames = [fname.replace('.jpg', '.labl') for fname in training_fnames]
    else:
        training_filenames = os.path.listdir(j['annotation_dir'])
        training_filenames = filter(lambda x: '.labl' in x, training_filenames)

    return object_names, relationships, training_fnames



if __name__ == '__main__':
    cfg = get_cfg()
    object_names = cfg[0]
    relationships = cfg[1]
    training_fnames = cfg[2]

    import pdb; pdb.set_trace()
    print('done')
