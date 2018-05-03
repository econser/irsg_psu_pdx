"""
import prob_check as p

fd_fp = p.get_image_nums('neg_fd_fp')
td_tp = p.get_image_nums('neg_td_tp')
td_fp = p.get_image_nums('neg_td_fp')
td_tp = p.get_image_nums('neg_td_tp')

p.avg_hist(td_tp, 'neg_td_tp')
p.avg_hist(td_fp, 'neg_td_fp')
p.avg_hist(fd_fp, 'neg_fd_fp')
p.avg_hist(fd_tp, 'neg_fd_tp')
p.avg_hist(np.arange(100)+301, 'pos_test')

p.avg_hist([301, 302, 303], 'pos_test')
"""
class InstanceScore (object):
    def __init__(self, name, probabilities, densities=None, nlls=None):
        self.name = name
        self.probs = probabilities
        self.densities = densities
        self.nlls = nlls



def q_hist(images, gmms=None, query_obj=None):
    if gmms is None:
        import irsg_utils as iu
        gmms = iu.get_dw_gmms('dw_gmms_n3_log.pkl')
    
    if query_obj is None:
        import irsg_querygen as qgen
        query_obj = qgen.gen_three_obj_loop('dog_walker holding leash attached_to dog walked_by dog_walker')
    
    do_hist_(images, query_obj, gmms)



def avg_hist(image_nums, image_type, gmms=None, query_obj=None):
    import numpy as np
    import irsg_utils as iu
    import matplotlib.pyplot as plt
    
    if gmms is None:
        import irsg_utils as iu
        gmms = iu.get_dw_gmms('dw_gmms_n3_log.pkl')
    
    if query_obj is None:
        import irsg_querygen as qgen
        query_obj = qgen.gen_three_obj_loop('dog_walker holding leash attached_to dog walked_by dog_walker')
    
    classes = [obj.names for obj in query_obj.objects]
    
    n_rows = 1
    n_cols = len(query_obj.objects) + len(query_obj.binary_triples)
    plot_num = 1
    
    sum_of_bins = np.zeros((n_cols,50))
    bin_names = []
    bin_ranges = None
    
    for image_num in image_nums:
        filename, file_dir = get_file_loc(image_num, image_type)
        
        if len(filename) == 0 or len(file_dir) == 0:
            print 'no image #{} for image type "{}"'.format(image_num, image_type)
            import pdb; pdb.set_trace()
            continue
        
        pg_comps = iu.get_pregen_components(filename, file_dir, gmms, classes)
        scores = gen_scores(query_obj, pg_comps)
        
        for i, o in enumerate(scores):
            histo = np.histogram(o.probs, bins=50, range=(0.0, 1.0))
            sum_of_bins[i] += histo[0]
            
            if bin_ranges is None:
                bin_ranges = histo[1]
            
            # store bin names (assuming objects are processed in the same order)
            if len(bin_names) != n_cols:
                bin_names.append(o.name)
    
    sum_of_bins /= (len(image_nums) * 1.)
    for i, hist in enumerate(sum_of_bins):
        object_name = bin_names[i]
        title = '{}'.format(object_name)
        
        plt.subplot(n_rows, n_cols, plot_num)
        
        plt.bar(bin_ranges[:-1], hist, width=0.02, log=True)
        axes = plt.gca()
        y_max = int(np.max(hist))
        y_max = np.log10(y_max).round()+1
        if y_max < 3:
            y_max = 3
        
        axes.set_ylim([0.1, pow(10,y_max)])
        #axes.set_xlim([0.0, 1.0])
        
        if len(title) != 0:
            plt.title(title)
        
        plt.grid(True)
        plot_num += 1
    
    plt.subplots_adjust(wspace=0.20)
    plt.show()



def do_hist_(image_nums, query, gmms):
    import numpy as np
    import irsg_utils as iu
    import matplotlib.pyplot as plt
    
    classes = [obj.names for obj in query.objects]
    
    n_rows = len(image_nums)
    n_cols = len(query.objects) + len(query.binary_triples)
    plot_num = 1
    
    for image_tup in image_nums:
        image_num = image_tup[0]
        image_type = image_tup[1]
        
        filename, file_dir = get_file_loc(image_num, image_type)
        
        if len(filename) == 0 or len(file_dir) == 0:
            print 'no image #{} for image type "{}"'.format(image_num, image_type)
            import pdb; pdb.set_trace()
            continue
        
        pg_comps = iu.get_pregen_components(filename, file_dir, gmms, classes)
        scores = gen_scores(query, pg_comps)
        
        for i, o in enumerate(scores):
            object_name = o.name
            title = '{} - n={}'.format(object_name, len(scores))
            plt.subplot(n_rows, n_cols, plot_num)
            hist_sub(o.probs, title=object_name, bins=50)
            plot_num += 1
    plt.subplots_adjust(wspace=0.20)
    plt.show()



def get_image_nums(image_type):
    import os.path
    import numpy as np
    from os import listdir
    from os.path import isfile, join
    
    base_path = '/home/econser/School/research/run_results/'
    sub_dirs = {
        'pos_test'  : ('dogwalking, positive, portland test/' ,'dog-walking'),
        'pos_train' : ('dogwalking, positive, portland train/','dog_walking'),
        'neg_td_tp' : ('dogwalking, negative, dogandperson/'  ,''),
        'neg_td_fp' : ('dogwalking, negative, dognoperson/'   ,''),
        'neg_fd_fp' : ('dogwalking, negative, nodognoperson/' ,''),
        'neg_fd_tp' : ('dogwalking, negative, personnodog/'   ,'')}
    
    if not sub_dirs.has_key(image_type):
        return '', ''
    
    image_subdir = sub_dirs[image_type][0]
    prefix = sub_dirs[image_type][1]
    search_path = base_path + image_subdir + 'dog/'
    
    onlyfiles = [f for f in listdir(search_path) if isfile(join(search_path, f))]
    image_nums = np.array([int(f[:-4]) for f in onlyfiles])
    return image_nums



def get_file_loc(image_num, image_type):
    import os.path
    
    base_path = '/home/econser/School/research/run_results/'
    sub_dirs = {
        'pos_test'  : ('dogwalking, positive, portland test/' ,'dog-walking'),
        'pos_train' : ('dogwalking, positive, portland train/','dog_walking'),
        'neg_td_tp' : ('dogwalking, negative, dogandperson/'  ,''),
        'neg_td_fp' : ('dogwalking, negative, dognoperson/'   ,''),
        'neg_fd_fp' : ('dogwalking, negative, nodognoperson/' ,''),
        'neg_fd_tp' : ('dogwalking, negative, personnodog/'   ,'')}
    
    if not sub_dirs.has_key(image_type):
        return '', ''
    
    image_subdir = sub_dirs[image_type][0]
    prefix = sub_dirs[image_type][1]
    
    filename = '{}{}.csv'.format(prefix, image_num)
    if os.path.isfile(base_path + image_subdir + 'dog/' + filename):
        return filename, base_path + image_subdir
    return '', ''



def hist_sub(scores, title='', bins=100):
    import numpy as np
    import matplotlib.pyplot as plt
    
    bins_ = np.linspace(start=0., stop=1., num=bins)
    n, bins, patches = plt.hist(scores, bins=bins_, log=True)
    
    axes = plt.gca()
    y_max = np.max(n)
    y_max = np.log10(y_max).round()+1
    if y_max < 3:
        y_max = 3
        
    
    axes.set_ylim([0.1, pow(10,y_max)])
    axes.set_xlim([0.0, 1.0])
    
    if len(title) != 0:
        plt.title(title)
    
    plt.grid(True)



def gen_scores(query, model_components, verbose=False, use_scaling=True):
    import itertools
    import numpy as np
    import opengm as ogm
    import irsg_utils as iutl
    
    verbose_tab = '  '
    do_unary_xform = True
    do_binary_xform = True
    
    unary_obj_descriptors = model_components.unary_components
    binary_models_dict = model_components.binary_components
    
    n_vars = []
    fg_to_sg = []
    fg_functions = []
    
    score_list = []
    
    #---------------------------------------------------------------------------
    # GENERATE UNARY FUNCTIONS
    
    for obj_ix, sg_object in enumerate(unary_obj_descriptors):
        if verbose: print('{}using model for object "{}"'.format(verbose_tab, sg_object.name))
        n_labels = len(sg_object.boxes)
        n_vars.append(len(sg_object.boxes))
        fg_to_sg.append(obj_ix)
    gm = ogm.gm(n_vars, operator='adder')
    
    # add unary functions to gm
    for fg_ix, sg_object in enumerate(unary_obj_descriptors):
        instance = InstanceScore(sg_object.name, sg_object.scores)
        scores = np.copy(sg_object.scores)
        instance.nlls = -np.log(scores + np.finfo(np.float).eps)
        score_list.append(instance)
    
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
        
        # get the model string
        relationship_key = ''
        if specific_rel in binary_models_dict:
            relationship_key = specific_rel
        elif wildcard_rel in binary_models_dict:
            relationship_key = wildcard_rel
        else:
            continue
        
        # generate box pairs
        sub_boxes = iutl.get_boxes(subject_name, model_components.unary_components)
        n_sub_boxes = len(sub_boxes)
        obj_boxes = iutl.get_boxes(object_name, model_components.unary_components)
        n_obj_boxes = len(obj_boxes)
        
        box_pairs = np.array([x for x in itertools.product(sub_boxes, obj_boxes)])
        gmm_features = iutl.get_gmm_features(box_pairs, in_format='xywh')
        
        params = model_components.binary_components[relationship_key]
        
        # run the features through the relationship model
        densities = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        if use_scaling and params.platt_a is not None and params.platt_b is not None:
            log_densities = np.log(scores + np.finfo(np.float).eps)
            probs = 1. / (1. + np.exp(params.platt_a * log_densities + params.platt_b))
        nlls = -np.log(probs)
        instance = InstanceScore(relationship_key, probs, densities=densities, nlls=nlls)
        score_list.append(instance)
    return score_list



def hist(scores, title='', bins=100):
    import numpy as np
    import matplotlib.pyplot as plt
    
    bins_ = np.linspace(start=0., stop=1., num=bins)
    n, bins, patches = plt.hist(scores, bins=bins_, log=True)
    
    axes = plt.gca()
    y_max = np.max(n)
    y_max = np.log10(y_max).round()+1
    if y_max < 3:
        y_max = 3
        
    
    axes.set_ylim([0.1, pow(10,y_max)])
    axes.set_xlim([0.0, 1.0])
    
    if len(title) != 0:
        plt.title(title)
    
    plt.grid(True)
    plt.show()



def gmm_pdf(X, mixture, mu, sigma):
    import numpy as np
    from scipy.stats import multivariate_normal as mvn
    
    n_components = len(mixture)
    n_vals = len(X)
    
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
        mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
    
    return mixed_pdf
