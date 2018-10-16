"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP

import train_situation_gmms as g
gmms = g.gen_gmms('handshake_situation.json')
g.save(gmms, '/home/econser/research/data/situation_gmms.pkl')


"""
from __future__ import print_function
import os
import sys
import json as j
import numpy as np
import irsg_utils as iu
import sklearn.mixture as skl
import sklearn.linear_model as lm
from sklearn.utils import shuffle



def gen_gmms(situation_cfg, annotation_dir, training_fnames, cal_method, n_components=3):
    # pull important info from the situation cfg
    f_cfg = open(situation_cfg, 'rb')
    j_cfg = j.load(f_cfg)
    f_cfg.close()
    situation_name = j_cfg['situation_name']
    
    relationships = j_cfg['relationships']
    relationship_dict = get_annotations(situation_name, annotation_dir, training_fnames, relationships)
    
    gmms = {}
    for rel_data in relationships:
        rel_name = rel_data['name']
        pos_bboxes = relationship_dict[rel_name]
        
        print('Creating model for "{}" relationship:'.format(rel_name))
        gmm = train_gmm(pos_bboxes, n_components)
        
        if cal_method == 'sample':
            neg_bboxes = gen_neg_bboxes(gmm, pos_bboxes, neg_size=len(pos_bboxes))
        else:
            neg_bboxes = get_neg_bboxes(relationship_dict, rel_name)
        cal_params = calibrate(gmm, pos_bboxes, neg_bboxes, title=rel_name)
        
        gmms[rel_name] = iu.RelationshipParameters(cal_params[0], cal_params[1], gmm.weights_, gmm.means_, gmm.covariances_, gmm)
    
    return gmms



def get_annotations(situation_name, annotation_dir, training_fnames, relationship_cfg):
    import gen_bboxes as g
    
    #file_list = os.listdir(annotation_dir)
    #n_annotations = 0
    #for filename in file_list:
    #    if filename.endswith('.labl'):
    #        n_annotations += 1
    
    relationship_dict = {}
    for rel in relationship_cfg:
        rel_name = rel['name']
        relationship_dict[rel_name] = []
    
    bbox_fn = g.get_bbox_fn(situation_name)
    for fname in training_fnames:
        if not fname.endswith('.labl'):
            continue
        
        # parse the annotation
        f = open(fname, 'rb')
        line = f.readlines()[0]
        bboxes = bbox_fn(line)
        
        # append np.array(subj_bbox, obj_bbox) to the appropriate relationship
        for rel_data in relationship_cfg:
            rel_name = rel_data['name']
            for rel in rel_data['components']:
                sub_class = rel['subject']
                obj_class = rel['object']
                box_tup = (bboxes[sub_class], bboxes[obj_class])
                relationship_dict[rel_name].append(np.array(box_tup))
    
    # convert each relation bbox list to array
    for rel in relationship_cfg:
        rel_name = rel['name']
        relationship_dict[rel_name] = np.array(relationship_dict[rel_name])
    
    return relationship_dict



def train_gmm(bboxes, n_components):
    # just train the gmm and return it
    features = iu.get_gmm_features(bboxes, in_format='xywh')
    print('  training relationship GMM...', end=''); sys.stdout.flush()
    gmm = skl.GaussianMixture(n_components, 'full', verbose=0, max_iter=500, n_init=50, tol=1e-6, init_params='random')
    gmm.fit(features)
    print('done!')
    
    return gmm



def calibrate(gmm, pos_bboxes, neg_bboxes, title=''):
    pos_features = iu.get_gmm_features(pos_bboxes, in_format='xywh')
    pos_scores = iu.gmm_pdf(pos_features, gmm.weights_, gmm.means_, gmm.covariances_)
    n_pos = len(pos_features)
    pos_labels = np.ones(n_pos) #* ((n_pos + 1.)/(n_pos + 2.))    
    
    neg_features = iu.get_gmm_features(neg_bboxes, in_format='xywh')
    neg_scores = iu.gmm_pdf(neg_features, gmm.weights_, gmm.means_, gmm.covariances_)
    n_neg = len(neg_features)
    neg_labels = np.zeros(n_neg) #* (1. / (n_neg + 2.))
    
    all_features = np.concatenate((pos_features, neg_features))
    all_labels = np.concatenate((pos_labels, neg_labels))
    all_scores = np.concatenate((pos_scores, neg_scores))
    all_log_scores = np.log(all_scores + np.finfo(np.float).eps)
    
    shuff_scores, shuff_labels = shuffle(all_scores, all_labels)
    platt_cal = lm.LogisticRegression(penalty='l2')
    platt_cal.fit(shuff_scores.reshape(-1,1), shuff_labels)
    platt_params = (platt_cal.coef_[0][0], platt_cal.intercept_[0])
    
    import matplotlib.pyplot as plt
    plt.scatter(all_scores, all_labels)
    x_vals = np.linspace(0.0, np.max(all_scores), num=100)
    y_vals = 1. / (1. + np.exp(-(platt_params[0] * x_vals + platt_params[1])))
    plt.plot(x_vals, y_vals)
    plt.title(title)
    plt.show()
    
    return platt_params

def get_neg_bboxes(bbox_dict, pos_rel_name):
    neg_bboxes = None
    rel_names = bbox_dict.keys()
    if len(rel_names) > 1:
        # use a different relationship as the negative set
        pos_rel_ix = rel_names.index(pos_rel_name)
        neg_rel_ix = (pos_rel_ix + 1) % len(rel_names)
        neg_rel_name = rel_names[neg_rel_ix]
        neg_bboxes  = bbox_dict[neg_rel_name]
    else:
        # swap the subject and object bboxes for the pos relation
        pos_bboxes = bbox_dict[pos_rel_name]
        neg_bboxes = np.zeros_like(pos_bboxes, dtype=np.int)
        neg_bboxes[:,0] = pos_bboxes[:,1]
        neg_bboxes[:,1] = pos_bboxes[:,0]
    return neg_bboxes

def gen_neg_bboxes(gmm, pos_bboxes, n_samples=1000000, neg_size=100):
    # take many samples from the gmm and use the lowest N as low probability configs
    samples = gmm.sample(n_samples)[0]
    scores = iu.gmm_pdf(samples, gmm.weights_, gmm.means_, gmm.covariances_)
    sorted_ixs = np.argsort(scores)
    neg_samples = samples[sorted_ixs[:neg_size]]
    
    # generate gmms for sampling subject bboxes
    subject_bboxes = pos_bboxes[:,0]
    print('  generating subject bbox GMM...', end=''); sys.stdout.flush()
    subject_bbox_gmm = skl.GaussianMixture(3, 'full', verbose=0, max_iter=500, n_init=50, tol=1e-6, init_params='random')
    subject_bbox_gmm.fit(subject_bboxes)
    print('done!')
    
    # generate a subject bbox for each neg sample
    subject_samples = subject_bbox_gmm.sample(neg_size)[0]
    obj_w = neg_samples[:,2] * subject_samples[:,2]
    obj_x = subject_samples[:,0] + 0.5 * subject_samples[:,2] - subject_samples[:,2] * neg_samples[:,0] - 0.5 * obj_w
    obj_h = neg_samples[:,3] * subject_samples[:,3]
    obj_y = subject_samples[:,1] + 0.5 * subject_samples[:,3] - subject_samples[:,3] * neg_samples[:,1] - 0.5 * obj_h
    object_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
    
    # clamp subject and object samples to [0, +inf] and convert to int
    
    subject_samples = np.array(subject_samples, dtype=np.int)
    subject_samples[:,0] = np.clip(subject_samples[:,0], 0, None)
    subject_samples[:,1] = np.clip(subject_samples[:,1], 0, None)
    subject_samples[:,2] = np.clip(subject_samples[:,2], 1, None)
    subject_samples[:,3] = np.clip(subject_samples[:,3], 1, None)
    object_samples = np.array(object_samples, dtype=np.int)
    object_samples[:,0] = np.clip(object_samples[:,0], 0, None)
    object_samples[:,1] = np.clip(object_samples[:,1], 0, None)
    object_samples[:,2] = np.clip(object_samples[:,2], 1, None)
    object_samples[:,3] = np.clip(object_samples[:,3], 1, None)
    
    neg_bboxes = np.hstack((subject_samples, object_samples))
    neg_bboxes = np.reshape(neg_bboxes, (neg_size, 2, 4))
    return neg_bboxes



def save_gmms(gmms, fname):
    import cPickle
    f = open(fname, 'wb')
    cPickle.dump(gmms, f, cPickle.HIGHEST_PROTOCOL)
    f.close()



#===============================================================================



def train_gmm_(pos_boxes, neg_boxes=None, n_components=3):
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



#===============================================================================
def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Binary Relation generation')
    parser.add_argument('--cfg', dest='json_cfg')
    parser.add_argument('--method', dest='neg_method', default='sample')
    parser.add_argument('--n', dest='n_components', default=3)
    parser.add_argument('--out_fname', dest='out_fname')
    args = parser.parse_args()
    
    json_fname = args.json_cfg
    f = open(json_fname, 'rb')
    j_cfg = j.load(f)
    f.close()
    
    # training images (training_dir or training_files)
    annotation_dir = j_cfg['annotation_dir']
    training_fnames = []
    if 'training_filenames' in j_cfg:
        with open(j_cfg['training_filenames'], 'rb') as f:
            training_fnames = f.readlines()
            training_fnames = [fname.rstrip('\n') for fname in training_fnames]
            training_fnames = [fname.rstrip('\r') for fname in training_fnames]
            training_fnames = [fname.replace('.jpg', '.labl') for fname in training_fnames]
    else:
        training_fnames = os.listdir(annotation_dir)
        training_fnames = filter(lambda x: '.labl' in x, training_fnames)
    
    fq_training_fnames = []
    for fname in training_fnames:
        fq_training_fnames.append(os.path.join(annotation_dir, fname))
    
    return json_fname, annotation_dir, fq_training_fnames, int(args.n_components), args.out_fname, args.neg_method



if __name__ == '__main__':
    cfg = get_cfg()
    json_cfg = cfg[0]
    annotation_dir = cfg[1]
    training_fnames = cfg[2]
    n_components = cfg[3]
    out_fname = cfg[4]
    neg_method = cfg[5]
    
    gmm = gen_gmms(json_cfg, annotation_dir, training_fnames, neg_method, n_components)
    print('GMMs trained and calibrated')
    if out_fname is not None:
        save_gmms(gmm, out_fname)
        print('GMMs saved as {}'.format(out_fname))
