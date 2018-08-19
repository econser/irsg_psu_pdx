"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP

import cPickle
import leading_horse_gmm as lhg
g = lhg.gen_gmms()
out_filename = '/home/econser/research/irsg_psu_pdx/data/lh_gmms.pkl'
f = open(out_filename, 'wb')
cPickle.dump(g, f, cPickle.HIGHEST_PROTOCOL)
f.close()
"""
def gen_gmms(base_dir = '/home/econser/research/irsg_psu_pdx/', train_file = 'leadinghorse_fnames_train.txt', test_file = 'leadinghorse_fnames_test.txt', n_components=3):
    import irsg_utils as iu
    
    data_dir = base_dir + 'data/'
    image_dir = data_dir + 'LeadingHorse/'
    
    tt_split_dict = get_train_test_splits(data_dir, train_file, test_file)
    relationship_dict = get_lh_annotations(image_dir, tt_split_dict)
    
    holding = relationship_dict['train']['holding']
    attached_to = relationship_dict['train']['attached_to']
    is_leading = relationship_dict['train']['is_leading']
    lead_by = relationship_dict['train']['lead_by']
    
    gmms = {}
    sk_gmm, platt = train_gmm(holding, attached_to, n_components)
    gmms['holding'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(attached_to, holding, n_components)
    gmms['attached_to'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(is_leading, lead_by, n_components)
    gmms['is_leading'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)

    sk_gmm, platt = train_gmm(lead_by, is_leading, n_components)
    gmms['lead_by'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    return gmms



"""
"""
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
    platt_cal = lm.LogisticRegression(penalty='l1', fit_intercept=True)
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



def get_lh_annotations(directory, test_train_dict):
    import os
    import numpy as np
    import irsg_utils as iu
    
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
    test_rel_dict['is_leading'] = []
    test_rel_dict['lead_by'] = []
    
    train_rel_dict = {}
    relationship_dict['train'] = train_rel_dict
    train_rel_dict['holding'] = []
    train_rel_dict['attached_to'] = []
    train_rel_dict['is_leading'] = []
    train_rel_dict['lead_by'] = []
    
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        hFile = open(directory + filename, 'rb')
        lines = hFile.readlines()
        
        for line in lines:
            obj_dict = iu.get_lh_boxes(line)
            
            leader_bbox = obj_dict['horse-leader']
            horse_bbox= obj_dict['horse']
            lead_bbox = obj_dict['lead']
            
            holding_rel = np.array((leader_bbox, lead_bbox))
            attached_to_rel = np.array((lead_bbox, horse_bbox))
            is_leading_rel = np.array((leader_bbox, horse_bbox))
            lead_by_rel = np.array((horse_bbox, leader_bbox))
            
            file_prefix = filename.split('.')[0]
            image_type = test_train_dict[file_prefix]
            relationship_dict[image_type]['holding'].append(holding_rel)
            relationship_dict[image_type]['attached_to'].append(attached_to_rel)
            relationship_dict[image_type]['is_leading'].append(is_leading_rel)
            relationship_dict[image_type]['lead_by'].append(lead_by_rel)
        
        hFile.close()
    
    relationship_dict['test']['holding'] = np.array(relationship_dict['test']['holding'])
    relationship_dict['test']['attached_to'] = np.array(relationship_dict['test']['attached_to'])
    relationship_dict['test']['is_leading'] = np.array(relationship_dict['test']['is_leading'])
    relationship_dict['test']['lead_by'] = np.array(relationship_dict['test']['lead_by'])
    
    holding = np.array(relationship_dict['train']['holding'])
    attached_to = np.array(relationship_dict['train']['attached_to'])
    is_leading = np.array(relationship_dict['train']['is_leading'])
    lead_by = np.array(relationship_dict['train']['lead_by'])
    
    relationship_dict['train']['holding'] = holding #gen_box_set(holding, attached_to)
    relationship_dict['train']['attached_to'] = attached_to #gen_box_set(attached_to, holding)
    relationship_dict['train']['is_leading'] = is_leading #gen_box_set(is_walking, walked_by)
    relationship_dict['train']['lead_by'] = lead_by #gen_box_set(walked_by, is_walking)
    
    return relationship_dict



def gen_box_set(pos_boxes, neg_boxes):
    import numpy as np
    neg = np.copy(neg_boxes)
    neg[:,1] = 0.0
    box_set = np.vstack((pos_boxes, neg))
    return box_set



def augment_box(box, max_adjustment=0.2, n_augments=5):
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



def run_test(gmms, data_split='test', relationship='holding'):
    import irsg_utils as iutl
    features = iutl.get_gmm_features(relationship_dict[data_split][relationship])
    features = np.array(features)
    scores = gmms[relationship].score_samples(features)
    return scores    
