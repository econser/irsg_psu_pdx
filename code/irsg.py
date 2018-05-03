"""
COMMON OBJECTS:
      airplane,      backpack,         bench,        bicycle,           bowl,
           bus,           car,          cart,          chair,            dog,
      elephant,        helmet,         horse,           lamp,         laptop,
     microwave,    motorcycle,        person,          pizza,          purse,
        racket,  refrigerator,         stove,     sunglasses,          table,
           tie, traffic_light,         train

person on bench (4, 9, 132)
person wearing helmet (49, 93, 127, 147)
person on horse (59)
cup on table (95)

--------------------------------------------------------------------------------
EXAMPLES:

=== querying a pickled config ===
import cPickle
h_mc_dog = open('/home/econser/School/research/images/dog_wearing_sunglasses.pkl', 'rb')
mc_dog = cPickle.load(h_mc_dog)
irsg.plot_box_matches('dog next_to person', mc_dog)

=== query every (pickled) image in a dir for energy ===
energies = irsg.qdir_energy('person on bench', '/home/econser/School/research/images/')

=== show every best-bbox image for a query
irsg.query_directory('person on bench', '/home/econser/School/research/images/')

===
import caffe as c
mc = i.ModelConfig('/home/econser/School/research/models/', 'model_definitions/dog_walking/faster_rcnn_end2end/test.prototxt', 'model_weights/dog_walking_faster_rcnn_final.caffemodel')
t_img_filename = '/home/econser/School/research/images/dog-walking1-resize.jpg'
cls = c.Classifier(mc.definitions, mc.weights, image_dims=(1000,1000))
t_input = [c.io.load_image(t_img_filename)]
pred = cls.predict(t_input, True)
--------------------------------------------------------------------------------
"""
# scene graph -> factor graph
# run GOP
# score each box (objects)
# score each box (attrs)
# score each box pair (relationships)
# infer energy

import sys
sys.path.append('/home/econser/School/Thesis/external/gop_1.3/src')
sys.path.append('/home/econser/School/Thesis/external/gop_1.3/build/lib')
sys.path.append('/home/econser/School/Thesis/external/py-faster-rcnn/lib')



def show_pypath():
    import sys
    for path in sys.path:
        print path

from gop import *
from util import *
import numpy as np
#import opengm as ogm
from irsg_utils import *



def plot_box_matches(query_str, model_components):
    import irsg_querygen as iqg
    import plot_utils as pu
    
    qry = iqg.gen_sro(query_str)
    img = model_components.image_filename
    best_matches = get_box_matches(qry, img, model_components)
    pu.draw_best_objects(img, best_matches)



def get_box_matches(query, image_filename, model_components=None):
    if model_components == None:
        model_components = get_components(image_filename)
    pgm, pgm_to_sg = gen_factor_graph(query, model_components, verbose=True)
    infr_result = do_inference(pgm)
    infr_result.infer()
    
    best_matches = []
    for pgm_var_ix, box_ix in enumerate(infr_result.arg()):
        box_coords = model_components.boxes[box_ix]
        sg_obj_ix = pgm_to_sg[pgm_var_ix]
        obj_name = query.objects[sg_obj_ix].names
        best_matches.append((obj_name, box_coords))
    return best_matches



def get_energy(query, image_filename, model_components=None):
    if model_components is None:
        model_components = get_components(image_filename)
    pgm, pgm_to_sg = gen_factor_graph(query, model_components, verbose=True)
    infr_result = do_inference(pgm)
    infr_result.infer()
    return infr_result.value()



def do_inference(pgm):
    ogm_params = ogm.InfParam(steps=120, damping=0., convergenceBound=0.001)
    infr_output = ogm.inference.BeliefPropagation(pgm, parameter=ogm_params)
    return infr_output



#===============================================================================
# Model parameter loading
#===============================================================================
def get_components(image_filename, class_names=None, class_dict=None, bin_model_dict=None):
    if class_names is None or class_dict is None:
        # get an array and dict for classifier class and index lookup
        class_filepath ='/home/econser/School/Thesis/external/caffe/data/ilsvrc12/'
        class_filename = 'det_synset_words.txt'
        class_names, class_dict = get_detected_classes(class_filepath, class_filename)
    
    if bin_model_dict is None:
        # get the relationship model dict
        relation_filepath = '/home/econser/School/research/data/'
        relation_filename = 'binary_models.pickle'
        bin_model_dict = get_binary_models(relation_filepath, relation_filename)
    
    # generate boxes and classify the patches
    boxes = get_boxes(image_filename, verbose=True)
    # TODO: SVM here
    classification = classify_boxes(image_filename, boxes, verbose=True)
    #classification = run_svm(svm_weights, classification)
    
    # convert the caffe classification to a 2d matrix (boxes, classes)
    score_matrix = []
    for row in classification:
        score_matrix.append(row['prediction'])
    score_matrix = np.array(score_matrix)
    
    # bundle up the model data
    model_components = IRSGComponents(image_filename, boxes, score_matrix, class_names, class_dict, bin_model_dict)
    return model_components



# parameterize:
#   n_seeds
#   segmentations_per_seed
#   max_iou
#   detector filename
#   geodesic K-means arg 3
# output:
#   boxes as [x_min, y_min, x_max, y_max]
def get_boxes(image, verbose=False):
    import time
    
    n_seeds = 140
    segmentations_per_seed = 6#4
    max_iou = 0.8
    
    sl = setupLearned(n_seeds, segmentations_per_seed, max_iou)
    prop = proposals.Proposal(sl)
    
    detector = contour.MultiScaleStructuredForest()
    detector.load("/home/econser/School/Thesis/external/gop_1.3/data/sf.dat")
    
    img = imgproc.imread(image)
    start = time.time()
    s = segmentation.geodesicKMeans(img, detector, 2000)
    b = prop.propose(s)
    if verbose: print 'GOP: {} proposals generated in {:.1f} seconds'.format(len(b), time.time() - start)
    
    boxes = s.maskToBox(b)
    return boxes



# parameterize:
#   model path
#   model def file
#   model weights file
def classify_boxes(image_filename, box_list, model_cfg=None, verbose=False):
    import os
    os.environ['GLOG_minloglevel'] = '2' # reduce caffe output
    import caffe
    import time
    
    model_path, model_def, model_weights = ''
    if model_cfg is None:
        model_path = '/home/econser/School/Thesis/external/caffe/models/'
        model_def = model_path + 'bvlc_reference_rcnn_ilsvrc13/deploy.prototxt'
        model_weights = model_path + 'bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel'
    else:
        model_path = model_cfg.base_path
        model_def = model_cfg.definitions
        model_weights = model_cfg.weights
    
    #model_weights = model_path + '/rcnn/caffe_nets/ilsvrc_2012_train_iter_310k'
    detector = caffe.Detector(model_def, model_weights, raw_scale=255)
    
    caffe_boxes = box_list.copy()
    caffe_boxes[:,0], caffe_boxes[:,1] = caffe_boxes[:,1], caffe_boxes[:,0].copy()
    caffe_boxes[:,2], caffe_boxes[:,3] = caffe_boxes[:,3], caffe_boxes[:,2].copy()
    boxes = [caffe_boxes]
    images = [image_filename]
    images_and_windows = zip(images, boxes)
    
    t_start = time.time()
    detections = detector.detect_windows(images_and_windows)
    if verbose: print 'RCNN: processed {} boxes in {:.1f} sec'.format(len(box_list), time.time() - t_start)
    
    return detections



def get_detected_classes(file_path, file_name):
    from operator import itemgetter
    
    class_list = []
    with open(file_path+file_name) as f:
        for line in f:
            synset_id = line.split()[0]
            class_name = '_'.join(line.split()[1:])
            class_list.append((synset_id, class_name))
    class_list.sort(key = itemgetter(1))
    
    class_dict = {}
    for ix, tup in enumerate(class_list):
        class_dict[tup[1]] = ix
    
    return np.array(class_list)[:,1].copy(), class_dict



def get_binary_models(relation_filepath, relation_filename):
    import cPickle
    with open(relation_filepath + relation_filename, 'rb') as in_pkl:
        bin_model_dict = cPickle.load(in_pkl)
    return bin_model_dict



def pickle_image_components(directory):
    import os
    import cPickle
    
    class_filepath ='/home/econser/School/Thesis/external/caffe/data/ilsvrc12/'
    class_filename = 'det_synset_words.txt'
    class_names, class_dict = get_detected_classes(class_filepath, class_filename)
    
    relation_filepath = '/home/econser/School/research/data/'
    relation_filename = 'binary_models.pickle'
    bin_model_dict = get_binary_models(relation_filepath, relation_filename)
    
    files = os.listdir(directory)
    for ix, img_file in enumerate(files):
        if img_file.lower().endswith(('.png', '.jpg')):
            print 'pickling image "{}": {} of {} ======================='.format(img_file, ix, len(files))
            mc = get_components(directory + img_file, class_names, class_dict, bin_model_dict)
            pickle_filename = '.'.join(img_file.split('.')[:-1])
            f = open(directory + pickle_filename + '.pkl', 'wb')
            cPickle.dump(mc, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()



#===============================================================================
# DIRECTORY QUERIES
#===============================================================================
def query_directory_showimages(query_str, directory):
    import os
    import cPickle
    
    files = os.listdir(directory)
    
    n_pkl_files = 0
    for file in files:
        if file.lower().endswith('.pkl'):
            n_pkl_files += 1
            
    for ix, pkl_file in enumerate(files):
        filename = '.'.join(pkl_file.split('.')[:-1])
        if pkl_file.lower().endswith('.pkl'):
            print 'processing image "{}": {} of {} ======================='.format(filename, ix, n_pkl_files)
            hFile = open(directory + pkl_file, 'rb')
            model_components = cPickle.load(hFile)
            plot_box_matches(query_str, model_components)



def qdir_energy(query_str, directory):
    import os
    import cPickle
    import irsg_querygen as iqg
    
    query_obj = iqg.gen_sro(query_str)
    
    files = os.listdir(directory)
    
    n_pkl_files = 0
    for file in files:
        if file.lower().endswith('.pkl'):
            n_pkl_files += 1
    
    energy_list = []
    for ix, pkl_file in enumerate(files):
        filename = '.'.join(pkl_file.split('.')[:-1])
        if pkl_file.lower().endswith('.pkl'):
            print 'processing image "{}": {} of {} ======================='.format(filename, ix, n_pkl_files)
            hFile = open(directory + pkl_file, 'rb')
            model_components = cPickle.load(hFile)
            energy = get_energy(query_obj, model_components.image_filename, model_components)
            energy_list.append((model_components.image_filename, energy))
    
    return energy_list



#===============================================================================
# FACTOR GRAPH GENERATION
#===============================================================================

"""
    gen_factor_graph
    in:
        query: query structure
         boxes: list of bounding box coords
        scores: (box, classes) numpy array
        class_list: list of class names
        class_dict: mapping of class names -> index
    out:
        opengm factor graph
"""
def gen_factor_graph(query, model_components, verbose=False):
    import irsg_utils as iutl
    import itertools
    
    verbose_tab = '  '
    
    boxes = model_components.boxes
    scores = model_components.score_matrix
    class_names = model_components.class_names
    class_dict = model_components.class_dict
    relationship_dict = model_components.bin_model_dict
    
    n_labels = len(boxes)
    n_vars = []
    fg_to_sg = []
    fg_functions = []
    
    unary_scores = []
    sg_objects = query.objects
    
    #---------------------------------------------------------------------------
    # GENERATE UNARY FUNCTIONS
    
    for obj_ix, sg_object in enumerate(sg_objects):
        obj_name = sg_object.names
        if obj_name in class_dict:
            print '{}using model for object "{}"'.format(verbose_tab, obj_name)
            n_vars.append(n_labels)
            obj_class_ix = np.where(class_names == obj_name)[0][0]
            class_scores = scores[:,obj_class_ix]
            unary_scores.append(class_scores)
            fg_to_sg.append(obj_ix)
        elif verbose:
            print '{}skipping object {}, not in scored classes'.format(verbose_tab, obj_name)
    gm = ogm.gm(n_vars, operator='adder')
    
    # add unary functions to gm
    for fg_ix, class_scores in enumerate(unary_scores):
        scores = np.log(-class_scores)
        fn_id = gm.addFunction(scores)
        fg_functions.append((1, fn_id, [fg_ix]))
    
    #---------------------------------------------------------------------------
    # GENERATE BINARY FUNCTIONS
    
    # generate box pairs and convert to GMM features
    box_pairs = np.array([x for x in itertools.product(boxes, boxes)])
    gmm_features = iutl.get_gmm_features(box_pairs)
    
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
        subject_name = query.objects[rel.subject].names
        object_name = query.objects[rel.object].names
        
        # specific: <subject_<relationship>_<object>
        specific_rel = subject_name + '_'
        specific_rel += rel.predicate.replace(' ', '_')
        specific_rel += '_' + object_name
        
        # check bail-out conditions
        if rel.subject == rel.object:
            if verbose: print '{}skipping self-relation: {}'.format(verbose_tab, specific_rel)
            continue
        if subject_name not in class_dict and object_name not in class_dict:
            if verbose: print '{}skipping relationship "{}", both objects unrecognized'.format(verbose_tab, specific_rel)
            continue
        if subject_name not in class_dict:
            if verbose: print '{}skipping relationship "{}", object "{}" unrecognized'.format(verbose_tab, specific_rel, subject_name)
            continue
        if object_name not in class_dict:
            if verbose: print '{}skipping relationship "{}", object "{}" unrecognized'.format(verbose_tab, specific_rel, object_name)
            continue
        
        # wildcard: *_<relationship>_*
        wildcard_rel = '*_'
        wildcard_rel += rel.predicate.replace(' ', '_')
        wildcard_rel += '_*'
        
        # get the model params from the GMM parameter dictionary
        relationship_key = ''
        if specific_rel in relationship_dict:
            if verbose: print '{}using relationship model for "{}"'.format(verbose_tab, specific_rel)
            relationship_key = specific_rel
        elif wildcard_rel in relationship_dict:
            if verbose: print '{}no relationship model for "{}", using "{}"'.format(verbose_tab, specific_rel, wildcard_rel)
            relationship_key = wildcard_rel
        else:
            if verbose: print '{}no relationship models for "{}" or "{}", skipping relationship'.format(verbose_tab, specific_rel, wildcard_rel)
            continue
        
        params = relationship_dict[relationship_key]
        
        # run the features through the relationship model
        scores = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
        scores = np.log(scores)
        platt_scores = scores
        if params.platt_a is not None and params.platt_b is not None:
            platt_scores = 1. / (1. + np.exp(params.platt_a * scores + params.platt_b))
        log_likelihoods = -np.log(platt_scores)
        
        bin_fns = np.reshape(log_likelihoods, (n_labels, n_labels))
        
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
    
    return gm, fg_to_sg



"""
/home/econser/School/Thesis/external/caffe/python
/home/econser/School/Thesis/code/external/opengm/build/src/interfaces/python/opengm
/home/econser/School/Thesis/external/gop_1.3/build/lib
/home/econser/School/research/code
/home/econser/anaconda3/envs/py27/lib/python27.zip
/home/econser/anaconda3/envs/py27/lib/python2.7
/home/econser/anaconda3/envs/py27/lib/python2.7/plat-linux2
/home/econser/anaconda3/envs/py27/lib/python2.7/lib-tk
/home/econser/anaconda3/envs/py27/lib/python2.7/lib-old
/home/econser/anaconda3/envs/py27/lib/python2.7/lib-dynload
/home/econser/anaconda3/envs/py27/lib/python2.7/site-packages
/home/econser/anaconda3/envs/py27/lib/python2.7/site-packages/PIL
/home/econser/anaconda3/envs/py27/lib/python2.7/site-packages/setuptools-3.6-py2.7.egg
"""
