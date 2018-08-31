"""
    return a dict of relation_name -> RelationshipParameters
    uses scikit learn GaussianMixture to train the GMM, an pulls the parameters into the RP

import cPickle
import person_wearing_glasses_gmm as mdl
g = mdl.gen_gmms()
out_filename = '/home/econser/research/irsg_psu_pdx/data/pwg_gmms.pkl'
f = open(out_filename, 'wb')
cPickle.dump(g, f, cPickle.HIGHEST_PROTOCOL)
f.close()
"""
def gen_gmms(base_dir = '/home/econser/research/irsg_psu_pdx/', train_file = 'personwearingglasses_fnames_train.txt', n_components=3):
    import irsg_utils as iu
    
    data_dir = base_dir + 'data/'
    image_dir = data_dir + 'PersonWearingGlasses/PersonWearingGlassesTrain/'
    
    tt_split_dict = get_training_data(data_dir, train_file)
    relationship_dict = get_pwg_annotations(image_dir, tt_split_dict)
    
    wearing = relationship_dict['train']['wearing']
    worn_by = relationship_dict['train']['worn_by']
    
    gmms = {}
    sk_gmm, platt = train_gmm(wearing, worn_by, n_components)
    gmms['wearing'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
    sk_gmm, platt = train_gmm(worn_by, wearing, n_components)
    gmms['worn_by'] = iu.RelationshipParameters(platt[0], platt[1], sk_gmm.weights_, sk_gmm.means_, sk_gmm.covariances_, sk_gmm)
    
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



def get_training_data(base_dir, train_filename):
    f = open(base_dir + train_filename, 'rb')
    train_files = f.readlines()
    f.close()
    
    file_dict = {}
    for filename in train_files:
        file_prefix = filename.split('.')[0]
        file_dict[file_prefix] = 'train'
    
    return file_dict



def get_pwg_annotations(directory, test_train_dict):
    import os
    import numpy as np
    import gen_bboxes as bb
    
    file_list = os.listdir(directory)
    n_annotations = 0
    for filename in file_list:
        if filename.endswith('.labl'):
            n_annotations += 1
    
    relationship_dict = {}
    
    test_rel_dict = {}
    relationship_dict['test'] = test_rel_dict
    test_rel_dict['wearing'] = []
    test_rel_dict['worn_by'] = []
    
    train_rel_dict = {}
    relationship_dict['train'] = train_rel_dict
    train_rel_dict['wearing'] = []
    train_rel_dict['worn_by'] = []

    bbox_fn = bb.get_bbox_fn_('person_wearing_glasses')
    for filename in file_list:
        if not filename.endswith('.labl'):
            continue
        
        hFile = open(directory + filename, 'rb')
        lines = hFile.readlines()
        
        for line in lines:
            obj_dict = bbox_fn(line)
            
            person_bbox = obj_dict['person']
            glasses_bbox= obj_dict['glasses']
            
            wearing_rel = np.array((person_bbox, glasses_bbox))
            worn_by_rel = np.array((glasses_bbox, person_bbox))
            
            file_prefix = filename.split('.')[0]
            if file_prefix not in test_train_dict:
                continue
            image_type = test_train_dict[file_prefix]
            relationship_dict[image_type]['wearing'].append(wearing_rel)
            relationship_dict[image_type]['worn_by'].append(worn_by_rel)
        
        hFile.close()
    
    wearing = np.array(relationship_dict['train']['wearing'])
    worn_by = np.array(relationship_dict['train']['worn_by'])
    
    relationship_dict['train']['wearing'] = wearing
    relationship_dict['train']['worn_by'] = worn_by
    
    return relationship_dict



def run_test(gmms, data_split='test', relationship='holding'):
    import irsg_utils as iutl
    features = iutl.get_gmm_features(relationship_dict[data_split][relationship])
    features = np.array(features)
    scores = gmms[relationship].score_samples(features)
    return scores    
