from __future__ import print_function
import numpy as np
import opengm as ogm

dir_map = {
    'stanford': '/home/econser/School/research/run_results/dogwalking, positive, stanford/', # stanford_walking_the_dog_001.csv
    'pos_test': '/home/econser/School/research/run_results/dogwalking, positive, portland test/',   # dog-walking301.csv
    'pos_train': '/home/econser/School/research/run_results/dogwalking, positive, portland train/', # dog-walking1.csv
    'neg_np_nd': '/home/econser/School/research/run_results/dogwalking, negative, nodognoperson/',  # <number>.csv
    'neg_np_pd': '/home/econser/School/research/run_results/dogwalking, negative, dognoperson/',    # <number>.csv
    'neg_pp_nd': '/home/econser/School/research/run_results/dogwalking, negative, personnodog/',    # <number>.csv
    'neg_pp_pd': '/home/econser/School/research/run_results/dogwalking, negative, dogandperson/'    # <number>.csv
}



"""
fb_query_str = 'dog_walker holding leash attached_to dog'
fb_query_fn = qgen.gen_two_rel_chain
fb_base_dir = '/home/econser/School/research/output/'
fb_energy_subdir = ''
fb_viz_plot_subdir = 'viz_psu_full-chain/'
fb_energy_filename = 'energy_psu_full-chain.csv'
fb_r_at_k_data_filename = 'r_at_k_psu_full_chain.csv'
fb_r_at_k_plot_filename = 'r_at_k_psu_full_chain.png'
fb_batch_split = 'pos'
dw.full_batch(fb_query_str, fb_query_fn, fb_base_dir, fb_viz_plot_subdir, fb_energy_subdir, fb_energy_filename, fb_r_at_k_data_filename, fb_r_at_k_plot_filename, fb_batch_split, gmms)

import plot_utils as p
ranks = p.gen_ranks(nrg_dir + nrg_filename)
r_at_k = p.get_r_at_k(ranks, 400)
p.r_at_k_plot(r_at_k, filename=nrg_dir+nrg_plotname)
"""
def multi_batch(gmms=None, use_scaling=True, do_viz_plots=True):
    import irsg_querygen as qgen
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    chain_psu = ('dog_walker holding leash attached_to dog', qgen.gen_two_rel_chain, '/home/econser/School/research/output/', 'viz_psu_full-chain/', '', 'energy_psu_full-chain.csv', 'r_at_k_psu_full_chain.csv', 'r_at_k_psu_full_chain.png', 'pos')
    chain_stanford = ('dog_walker holding leash attached_to dog', qgen.gen_two_rel_chain, '/home/econser/School/research/output/', 'viz_stanford_full-chain/', '', 'energy_stanford_full-chain.csv', 'r_at_k_stanford_full_chain.csv', 'r_at_k_stanford_full_chain.png', 'stanford')
    
    cycle_psu = ('dog_walker holding leash attached_to dog walked_by dog_walker', qgen.gen_three_obj_loop, '/home/econser/School/research/output/', 'viz_psu_full-cycle/', '', 'energy_psu_full-cycle.csv', 'r_at_k_psu_full_cycle.csv', 'r_at_k_psu_full_cycle.png', 'pos')
    cycle_stanford = ('dog_walker holding leash attached_to dog walked_by dog_walker', qgen.gen_three_obj_loop, '/home/econser/School/research/output/', 'viz_stanford_full-cycle/', '', 'energy_stanford_full-cycle.csv', 'r_at_k_stanford_full_cycle.csv', 'r_at_k_stanford_full_cycle.png', 'stanford')
    
    dwhl_psu = ('dog_walker holding leash', qgen.gen_sro, '/home/econser/School/research/output/', 'viz_psu_dwhl/', '', 'energy_psu_dwhl.csv', 'r_at_k_psu_dwhl.csv', 'r_at_k_psu_dwhl.png', 'pos')
    dwhl_stanford = ('dog_walker holding leash', qgen.gen_sro, '/home/econser/School/research/output/', 'viz_stanford_dwhl/', '', 'energy_stanford_dwhl.csv', 'r_at_k_stanford_dwhl.csv', 'r_at_k_stanford_dwhl.png', 'stanford')
    
    latd_psu = ('leash attached_to dog', qgen.gen_sro, '/home/econser/School/research/output/', 'viz_psu_latd/', '', 'energy_psu_latd.csv', 'r_at_k_psu_latd.csv', 'r_at_k_psu_latd.png', 'pos')
    latd_stanford = ('leash attached_to dog', qgen.gen_sro, '/home/econser/School/research/output/', 'viz_stanford_latd/', '', 'energy_stanford_latd.csv', 'r_at_k_stanford_latd.csv', 'r_at_k_stanford_latd.png', 'stanford')

    #configs = [chain_psu, chain_stanford, cycle_psu, cycle_stanford, dwhl_psu, dwhl_stanford, latd_psu, latd_stanford]
    configs = [cycle_psu]
    for config in configs:
        full_batch(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], config[8], gmms, use_scaling, do_viz_plots)



"""

gmms = dw.get_dw_gmms('dw_gmms_platt.pkl')
dw.full_batch('dog_walker holding leash attached_to dog', qgen.gen_two_rel_chain, '/home/econser/School/research/output/', 'viz_psu_full-chain/', '', 'energy_psu_full-chain.csv', 'r_at_k_psu_full_chain.csv', 'r_at_k_psu_full_chain.png', 'pos', gmms)
"""
def full_batch(query_str, query_fn, base_dir, viz_plot_subdir, energy_subdir, energy_filename, r_at_k_data_filename, r_at_k_plot_filename, batch_split, gmms=None, use_scaling=True, do_viz_plots=True):
    import os.path
    import plot_utils as p; reload(p)
    import numpy as np
    
    if batch_split == 'neg':
        print('full_batch does not work for negative batch type')
        print('use: stanford or pos')
        return
    
    query_tup = gen_query_tup(query_str, query_fn)
    if gmms is None:
        gmms = get_dw_gmms()
    
    energy_dir = base_dir + energy_subdir
    if not os.path.exists(energy_dir):
        os.makedirs(energy_dir)
    
    viz_plot_dir = base_dir + viz_plot_subdir
    if not os.path.exists(viz_plot_dir):
        os.makedirs(viz_plot_dir)
    
    # run the energy batch with neg set
    print('===== Running batch type "{}" ===================='.format(batch_split))
    print('Calculating energy values')
    full_energy_batch(energy_dir, energy_filename, batch_split, gmms, query_tup, include_neg=True, use_scaling=use_scaling)
    
    # run the viz image batch
    if do_viz_plots:
        print('\nGenerating viz plots')
        full_viz_plot_batch(viz_plot_dir, batch_split, gmms, query_tup, include_neg=True, use_scaling=use_scaling)
    
    # gen the r_at_k plot and csv
    print('\nCalculating r@k values and plot')
    ranks = p.gen_ranks(energy_dir + energy_filename)
    r_at_k = p.get_r_at_k(ranks, 400)
    
    t = np.hstack(((np.arange(len(r_at_k))+1).reshape((400,1)),(r_at_k.reshape(400,1))))
    np.savetxt(energy_dir + r_at_k_data_filename, t, fmt='%d, %03f')
    
    p.r_at_k_plot(r_at_k, filename=energy_dir+r_at_k_plot_filename, x_limit=100)



#-------------------------------------------------------------------------------
# VIZ DATA CALLS
#

""" 
import irsg_dw as dw
import irsg_querygen as qgen

gmms = dw.get_dw_gmms()

latd_qry_str = 'leash attached_to dog'
latd_qry_obj = qgen.gen_sro(latd_qry_str)
latd_qry_tup = (latd_qry_obj, latd_qry_str)

dwhl_qry_str = 'dog_walker holding leash'
dwhl_qry_obj = qgen.gen_sro(dwhl_qry_str)
dwhl_qry_tup = (dwhl_qry_obj, dwhl_qry_str)

cyc_qry_str = 'dog_walker holding leash attached_to dog walked_by dog_walker'
cyc_qry_obj = qgen.gen_three_obj_loop(cyc_qry_str)
cyc_qry_tup = (cyc_qry_obj, cyc_qry_str)

full_viz_batch: generates viz data for the viz tool
full_viz_plot_batch: generates viz plots
full_batch: generates energy data
"""
def full_viz_batch(out_dir, out_filename, batch_split, gmms=None, query_tup=None):
    import irsg_querygen as qgen
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    if query_tup is None:
        query_str = 'dog_walker holding leash attached_to dog'
        query_obj = qgen.gen_two_rel_chain(query_str)
        query_tup = (query_obj, query_str)
    lines = []
    
    if batch_split == 'pos':
        print('  Running PSU images')
        pos_test = viz_batch('pos_test', query_tup, gmms, verbose=True)
        lines.extend(pos_test)
    elif batch_split == 'stanford':
        print('  Running Stanford images')
        res = viz_batch(batch_split, query_tup, gmms, verbose=True)
    else:
        print('  Running negative images (No person, No dog)')
        neg_np_nd = viz_batch('neg_np_nd', query_tup, gmms, verbose=True)
        lines.extend(neg_np_nd)
        
        print('  Running negative images (No person, dog)')
        neg_np_pd = viz_batch('neg_np_pd', query_tup, gmms, verbose=True)
        lines.extend(neg_np_pd)
        
        print('  Running negative images (person, No dog)')
        neg_pp_nd = viz_batch('neg_pp_nd', query_tup, gmms, verbose=True)
        lines.extend(neg_pp_nd)
        
        print('  Running negative images (person, dog)')
        neg_pp_pd = viz_batch('neg_pp_pd', query_tup, gmms, verbose=True)
        lines.extend(neg_pp_pd)
    
    # store all the viz data
    fq_filename = out_dir
    if fq_filename[-1] != '/':
        fq_filename += '/'
    fq_filename += out_filename
    
    f = open(fq_filename, 'wb')
    for line in lines:
        f.write(line)
    f.close()



def viz_batch(image_type, query_tup, gmms, verbose=False):
    import os
    import opengm as ogm
    import sys
    
    file_dir = dir_map[image_type]
    filenames = os.listdir(file_dir+'dog/')
    results = []
    for file_ix, filename in enumerate(filenames):
        if verbose: print('{}{:03d}/{:03d} - {}          '.format('    ', file_ix, len(filenames), filename), end='\r'); sys.stdout.flush()
        viz_data = viz_csv(filename, image_type, gmms, query_tup)
        results.append(viz_data)
    if verbose: print('')
    return results



def viz_csv(filename='dog-walking301.csv', image_type='pos_test', gmms=None, query_tup=None):
    import irsg_querygen as qgen
    import opengm as ogm
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    image_num = filename_to_id(filename)
    
    if query_tup is None:
        query_str = 'dog_walker holding leash attached_to dog'
        query_obj = qgen.gen_two_rel_chain(query_str)
        query_tup = (query_obj, query_str)
    
    pregen_dir = dir_map[image_type]
    classes = [obj.names for obj in query_tup[0].objects]
    pg_comps = get_pregen_components(filename, pregen_dir, gmms, classes)
    
    pgm, pgm_to_sg = gen_factor_graph(query_tup[0], pg_comps)
    energy, best_box_ixs, marginals = do_inference(pgm)
    
    # generate the csv data
    # 0 : image ID
    viz_data = '{:03d}, 0, "{}"\n'.format(image_num, query_tup[1])
    
    # 1 : object ix, obj name, bbox coords
    for pgm_obj_ix, bbox_ix in enumerate(best_box_ixs):
        sg_obj_ix = pgm_to_sg[pgm_obj_ix]
        sg_obj = pg_comps.unary_components[sg_obj_ix]
        
        obj_name = sg_obj.name
        b = sg_obj.boxes[bbox_ix]
        
        viz_data += '{:03d}, 1, {}, "{}", {}, {}, {}, {}\n'.format(image_num, sg_obj_ix, obj_name, int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    
    # 2 : energy
    viz_data += '{:03d}, 2, {:0.4f}\n'.format(image_num, energy)
    
    # 3 : match/no match
    # TODO: paramaterize the TP IDs
    if image_num <= 400:
        viz_data += '{:03d}, 3, match\n'.format(image_num)
    else:
        viz_data += '{:03d}, 3, no match\n'.format(image_num)
    
    # 4 : image subdirectory
    image_subdir = 'DogWalkingSituationNegativeExamples'
    if image_type == 'neg_np_nd':
        image_subdir += '/NoDogNoPerson'
    elif image_type == 'neg_np_pd':
        image_subdir += '/DogNoPerson'
    elif image_type == 'neg_pp_nd':
        image_subdir += '/PersonNoDog'
    elif image_type == 'neg_pp_pd':
        image_subdir += '/DogAndPerson'
    elif image_type == 'pos_test' or image_type == 'pos_train':
        image_subdir = 'PortlandSimpleDogWalking'
    viz_data += '{:03d}, 4, {}\n'.format(image_num, image_subdir)
    
    # 5 : image filename
    image_filename = id_to_filename(image_num, image_type)
    viz_data += '{:03d}, 5, {}\n'.format(image_num, image_filename)
    
    return viz_data



def vt(image_id=301, image_type='pos_test', gmms=None):
    filename = '{}.csv'.format(image_id)
    if image_type == 'pos_test' or image_type == 'pos_train':
        filename = 'dog-walking'+filename
    return viz_csv(filename, image_type, gmms)



#-------------------------------------------------------------------------------
# VIZ PLOT CALLS
#

"""
import irsg_dw as dw
import irsg_querygen as qgen
gmms = dw.get_dw_gmms()
dwhl_qry_str = 'dog_walker holding leash'
dwhl_qry_obj = qgen.gen_sro(dwhl_qry_str)
dwhl_qry_tup = (dwhl_qry_obj, dwhl_qry_str)
dw.viz_plot(out_dir='/home/econser/School/research/output/', gmms=gmms, query_tup=dwhl_qry_tup)
"""
def full_viz_plot_batch(out_dir, batch_split, gmms=None, query_tup=None, include_neg=False, use_scaling=True):
    import irsg_querygen as qgen
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    if query_tup is None:
        query_str = 'dog_walker holding leash attached_to dog'
        query_obj = qgen.gen_two_rel_chain(query_str)
        query_tup = (query_obj, query_str)
    
    if batch_split == 'pos':
        print('  Running PSU images')
        viz_plot_batch('pos_test', query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)
    elif batch_split == 'stanford':
        print('  Running Stanford images')
        viz_plot_batch(batch_split, query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)
    else:
        return
    
    if include_neg:
        print('  Running negative images (No person, No dog)')
        viz_plot_batch('neg_np_nd', query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)
        print('  Running negative images (No person, dog)')
        viz_plot_batch('neg_np_pd', query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)
        print('  Running negative images (person, No dog)')
        viz_plot_batch('neg_pp_nd', query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)
        print('  Running negative images (person, dog)')
        viz_plot_batch('neg_pp_pd', query_tup, gmms, out_dir, verbose=True, use_scaling=use_scaling)



def viz_plot_batch(image_type, query_tup, gmms, out_dir, verbose=False, use_scaling=True):
    import os
    import sys
    import opengm as ogm
    
    file_dir = dir_map[image_type]
    filenames = os.listdir(file_dir+'dog/')
    results = []
    for file_ix, filename in enumerate(filenames):
        if verbose: print('{}{:03d}/{:03d} - {}          '.format('    ', file_ix, len(filenames), filename), end='\r'); sys.stdout.flush()
        viz_plot(out_dir, filename, image_type, gmms, query_tup, use_scaling=use_scaling)
    if verbose: print('')



def viz_plot(outdir='/home/econser/School/research/output/', filename='dog-walking301.csv', image_type='pos_test', gmms=None, query_tup=None, use_scaling=True):
    import opengm as ogm
    import plot_utils as pt; reload(pt)
    import irsg_querygen as qgen
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    image_num = filename_to_id(filename)
    
    if query_tup is None:
        query_str = 'dog_walker holding leash attached_to dog'
        query_obj = qgen.gen_two_rel_chain(query_str)
        query_tup = (query_obj, query_str)
    
    pregen_dir = dir_map[image_type]
    classes = [obj.names for obj in query_tup[0].objects]
    pg_comps = get_pregen_components(filename, pregen_dir, gmms, classes)
    
    pgm, pgm_to_sg = gen_factor_graph(query_tup[0], pg_comps, use_scaling=use_scaling)
    energy, best_box_ixs, marginals = do_inference(pgm)
    
    image_subdir = 'psu_dw/DogWalkingSituationNegativeExamples'
    if image_type == 'neg_np_nd':
        image_subdir += '/NoDogNoPerson'
    elif image_type == 'neg_np_pd':
        image_subdir += '/DogNoPerson'
    elif image_type == 'neg_pp_nd':
        image_subdir += '/PersonNoDog'
    elif image_type == 'neg_pp_pd':
        image_subdir += '/DogAndPerson'
    elif image_type == 'pos_test' or image_type == 'pos_train':
        image_subdir = 'psu_dw/PortlandSimpleDogWalking'
    elif image_type == 'stanford':
        image_subdir = 'StanfordSimpleDogWalking'
    
    image_filename = id_to_filename(image_num, image_type)
    
    pt.draw_best_objects('/home/econser/School/research/images/'+image_subdir, pg_comps, best_box_ixs, energy, out_dir=outdir, out_filename='viz_'+image_filename.split('.')[0]+'.png')



#-------------------------------------------------------------------------------
# ENERGY FUNCTIONS
#
"""
    Get energy values for all images
"""
def full_energy_batch(out_dir, out_filename, batch_split, gmm1=None, query_tup=None, include_neg=True, use_scaling=True):
    import dog_walking_gmm as dwg
    import irsg_querygen as qgen
    
    if gmm1 is None:
        gmm1 = get_dw_gmms()
    
    if query_tup is None:
        query_str = 'dog_walker holding leash attached_to dog'
        query_obj = qgen.gen_two_rel_chain(query_str)
        query_tup = (query_obj, query_str)
    
    query = qgen.gen_two_rel_chain('dog_walker holding leash attached_to dog')
    batches = []
    
    if batch_split == 'pos':
        print('  Running PSU images')
        pos_test = energy_batch('pos_test', query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('pos', 'True', 'True', pos_test))
    elif batch_split == 'stanford':
        print('  Running Stanford images')
        res = energy_batch(batch_split, query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('pos', 'True', 'True', res))
    else:
        print('unknown batch split type "{}"'.format(batch_split))
    
    if batch_split == 'neg' or include_neg:
        print('  Running negative images (No person, No dog)')
        neg_np_nd = energy_batch('neg_np_nd', query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('neg', 'False', 'False', neg_np_nd))
        
        print('  Running negative images (No person, dog)')
        neg_np_pd = energy_batch('neg_np_pd', query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('neg', 'False', 'True', neg_np_pd))
        
        print('  Running negative images (person, No dog)')
        neg_pp_nd = energy_batch('neg_pp_nd', query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('neg', 'True', 'False', neg_pp_nd))
        
        print('  Running negative images (person, dog)')
        neg_pp_pd = energy_batch('neg_pp_pd', query_tup[0], gmm1, verbose=True, use_scaling=use_scaling)
        batches.append(('neg', 'True', 'True', neg_pp_pd))
    
    # store all the scores
    f = open(out_dir + out_filename, 'wb')
    for batch in batches:
        pos_neg_flag = batch[0]
        person_flag = batch[1]
        dog_flag = batch[2]
        results = batch[3]
        prefix = '{}, {}, {}, '.format(pos_neg_flag, person_flag, dog_flag)
        for result in results:
            line = prefix + '{}, {:4f}\n'.format(result[0], result[1])
            f.write(line)
    f.close()



"""
    Get energy values for all csvs in a directory
"""
def energy_batch(image_type, query, gmms, gt_bbox_dir=None, verbose=False, use_scaling=True):
    import os
    import opengm as ogm
    import sys
    
    classes = [obj.names for obj in query.objects]
    file_dir = dir_map[image_type]
    filenames = os.listdir(file_dir+'dog/')
    results = []
    for file_num, filename in enumerate(filenames):
        pg_comps = get_pregen_components(filename, file_dir, gmms, classes)
        pgm, pgm_to_sg = gen_factor_graph(query, pg_comps, use_scaling=use_scaling)
        energy, best_box_ixs, marginals = do_inference(pgm)
        results.append((filename, energy))
        iou = get_ious(pg_comps, best_box_ixs)
        if verbose: print('{}{:03d}/{:03d} - {}: {:.3f}          '.format('    ', file_num+1, len(filenames), filename, energy), end='\r'); sys.stdout.flush()
    if verbose: print('')
    return results



"""
Generate energy values for a particular directory
--------------------------------------------------------------------------------
import irsg_dw as dw
gmms = dw.get_dw_gmms('dw_gmms_n3_log.pkl')
import irsg_querygen as qgen
qry = qgen.gen_three_obj_loop('dog_walker holding leash attached_to dog walked_by dog_walker')
r = dw.energy_batch_('/home/econser/School/research/frcn_test/', qry, gmms)
"""
def energy_batch_(image_dir, query, gmms, gt_bbox_dir=None, verbose=False, use_scaling=True):
    import os
    import opengm as ogm
    import sys
    
    classes = [obj.names for obj in query.objects]
    filenames = os.listdir(image_dir + 'dog/')
    results = []
    for file_num, filename in enumerate(filenames):
        pg_comps = get_pregen_components(filename, image_dir, gmms, classes)
        pgm, pgm_to_sg = gen_factor_graph(query, pg_comps, use_scaling=use_scaling)
        energy, best_box_ixs, marginals = do_inference(pgm)
        results.append((filename, energy))
        iou = get_ious(pg_comps, best_box_ixs)
        if verbose: print('{}{:03d}/{:03d} - {}: {:.3f}          '.format('    ', file_num+1, len(filenames), filename, energy), end='\r'); sys.stdout.flush()
    if verbose: print('')
    return results



#-------------------------------------------------------------------------------
# QUICK TEST FUNCTIONS
#
"""
    Run one image from each image type
"""
def tt():
    gmm1 = get_dw_gmms()
    print('           pos test - 301  : {}'.format(t(301, 'pos_test', gmm1)))
    print(' no person,  no dog - 4    : {}'.format(t(4, 'neg_np_nd', gmm1)))
    print(' no person, yes dog - 561  : {}'.format(t(561, 'neg_np_pd', gmm1)))
    print('yes person,  no dog - 21   : {}'.format(t(21, 'neg_pp_nd', gmm1)))
    print('yes person, yes dog - 61605: {}'.format(t(61605, 'neg_pp_pd', gmm1)))



"""
    Run a single image, returns energy and box IDs
"""
def t(image_num=301, image_type='pos_test', gmms=None):
    import irsg_querygen as qgen
    import opengm as ogm
    
    if gmms is None:
        gmms = get_dw_gmms()
    
    filename = '{}.jpg'.format(image_num)
    if image_type == 'pos_test' or image_type == 'pos_train':
        filename = 'dog-walking'+filename
    pregen_dir = dir_map[image_type]
    query = qgen.gen_three_obj_loop('dog_walker holding leash attached_to dog walked_by dog_walker')
    classes = [obj.names for obj in query.objects]
    pg_comps = get_pregen_components(filename, pregen_dir, gmms, classes)
    
    pgm, pgm_to_sg = gen_factor_graph(query, pg_comps)
    energy, best_box_ixs, marginals = do_inference(pgm)
    return energy, best_box_ixs



#-------------------------------------------------------------------------------
"""
    Get the classes, boxes, scores for an images
"""
def get_pregen_components(image_filename, pregen_dir, gmms, classes):
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



def get_ious(pg_comps, best_box_ixs):
    return None

#-------------------------------------------------------------------------------
# FACTOR GRAPH GENERATION AND INFERENCE CALLS
#
"""
    Generate a factor graph from a query structure the model components
"""
def gen_factor_graph(query, model_components, verbose=False, use_scaling=True):
    
    import irsg_utils as iutl
    import itertools
    
    verbose_tab = '  '
    do_unary_xform = True
    do_binary_xform = True
    
    unary_obj_descriptors = model_components.unary_components
    binary_models_dict = model_components.binary_components
    
    n_vars = []
    fg_to_sg = []
    fg_functions = []
    
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
        scores = np.copy(sg_object.scores)
        if do_unary_xform:
            scores += np.finfo(np.float).eps
            scores = -np.log(scores)
        fn_id = gm.addFunction(scores)
        fg_functions.append((1, fn_id, [fg_ix]))
    
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
        #wildcard_rel = '*_'
        #wildcard_rel += rel.predicate.replace(' ', '_')
        #wildcard_rel += '_*'
        
        # get the model string
        relationship_key = ''
        if specific_rel in binary_models_dict:
            if verbose: print('{}using relationship model for "{}"'.format(verbose_tab, specific_rel))
            relationship_key = specific_rel
        elif wildcard_rel in binary_models_dict:
            if verbose: print('{}no relationship model for "{}", using "{}"'.format(verbose_tab, specific_rel, wildcard_rel))
            relationship_key = wildcard_rel
        else:
            if verbose: print('{}no relationship models for "{}" or "{}", skipping relationship'.format(verbose_tab, specific_rel, wildcard_rel))
            continue
        
        # generate box pairs
        sub_boxes = get_boxes(subject_name, unary_obj_descriptors)
        n_sub_boxes = len(sub_boxes)
        obj_boxes = get_boxes(object_name, unary_obj_descriptors)
        n_obj_boxes = len(obj_boxes)
        
        box_pairs = np.array([x for x in itertools.product(sub_boxes, obj_boxes)])
        gmm_features = iutl.get_gmm_features(box_pairs, in_format='xywh')
        
        params = binary_models_dict[relationship_key]
        
        # run the features through the relationship model
        
        scores = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        if do_binary_xform:
            scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
            if use_scaling and params.platt_a is not None and params.platt_b is not None:
                scores = np.log(scores)
                scores = 1. / (1. + np.exp(params.platt_a * scores + params.platt_b))
            scores = -np.log(scores)
        
        bin_fns = np.reshape(scores, (n_sub_boxes, n_obj_boxes))
        
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
    Run BP inference on a facor graph
    Returns energy, best match box indices
        marginals disabled - doesn't work when bbox counts are not the same for all objects
"""
def do_inference(gm, n_steps=120, damping=0., convergence_bound=0.001, verbose=False):
  """ Run belief propagation on the provided graphical model
  returns:
    energy (float): the energy of the GM
    var_indices (numpy array): indices for the best label for each variable
  """
  ogm_params = ogm.InfParam(steps=n_steps, damping=damping, convergenceBound=convergence_bound)
  infr_output = ogm.inference.BeliefPropagation(gm, parameter=ogm_params)
  
  if verbose:
    infr_output.infer(infr_output.verboseVisitor())
  else:
    infr_output.infer()
  
  detected_vars = []
  for i in range(0, gm.numberOfVariables):
    if gm.numberOfLabels(i) > 1:
      detected_vars.append(i)
  
  #infr_marginals = infr_output.marginals(detected_vars)
  #infr_marginals = np.exp(-infr_marginals)
  
  infr_best_match = infr_output.arg()
  infr_energy = infr_output.value()
  
  return infr_energy, infr_best_match, None



def get_all_scores(query, model_components, verbose=False):
    
    import irsg_utils as iutl
    import itertools
    
    verbose_tab = '  '
    do_unary_xform = True
    do_binary_xform = True
    
    unary_obj_descriptors = model_components.unary_components
    binary_models_dict = model_components.binary_components
    
    #n_vars = []
    #fg_to_sg = []
    #fg_functions = []
    
    score_dict = {}
    
    # Unary scores
    for fg_ix, sg_object in enumerate(unary_obj_descriptors):
        scores = np.copy(sg_object.scores)
        score_dict[sg_object.name] = scores
    
    # binary scores
    bin_relations = query.binary_triples
    relationships = []
    if isinstance(bin_relations, np.ndarray):
        for rel in bin_relations:
            relationships.append(rel)
    else:
        relationships.append(bin_relations)
    
    for rel in relationships:
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
            if verbose: print('{}using relationship model for "{}"'.format(verbose_tab, specific_rel))
            relationship_key = specific_rel
        elif wildcard_rel in binary_models_dict:
            if verbose: print('{}no relationship model for "{}", using "{}"'.format(verbose_tab, specific_rel, wildcard_rel))
            relationship_key = wildcard_rel
        else:
            if verbose: print('{}no relationship models for "{}" or "{}", skipping relationship'.format(verbose_tab, specific_rel, wildcard_rel))
            continue
        
        # generate box pairs
        sub_boxes = get_boxes(subject_name, unary_obj_descriptors)
        n_sub_boxes = len(sub_boxes)
        obj_boxes = get_boxes(object_name, unary_obj_descriptors)
        n_obj_boxes = len(obj_boxes)
        
        box_pairs = np.array([x for x in itertools.product(sub_boxes, obj_boxes)])
        gmm_features = iutl.get_gmm_features(box_pairs, in_format='xywh')
        
        params = binary_models_dict[relationship_key]
        
        # run the features through the relationship model
        
        scores = gmm_pdf(gmm_features, params.gmm_weights, params.gmm_mu, params.gmm_sigma)
        if do_binary_xform:
            scores += np.finfo(np.float).eps # float epsilon so that we don't try ln(0)
            if params.platt_a is not None and params.platt_b is not None:
                scores = 1. / (1. + np.exp(params.platt_a * scores + params.platt_b))
            
            score_dict[relationship_key] = scores
            #score_mtx = np.reshape(scores, (n_sub_boxes, n_obj_boxes))
            #score_dict[relationship_key] = score_mtx
    
    return score_dict



"""
    simple wrapper for pulling boxes from a list of UnaryComponents
"""
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



def gen_query_tup(query_string, qgen_fn):
    qry_obj = qgen_fn(query_string)
    return(qry_obj, query_string)



def filename_to_id(filename):
    t = filename.split('.')
    if t[0].startswith('dog-walking'):
        return int(t[0][11:])
    elif t[0].startswith('stanford_walking_the_dog_'):
        return int(t[0][25:])
    else:
        return int(t[0])



def id_to_filename(image_id, image_type):
    if image_type == 'pos_test' or image_type == 'pos_train':
        return 'dog-walking{}.jpg'.format(image_id)
    elif image_type == 'stanford':
        return 'stanford_walking_the_dog_{:03d}.jpg'.format(image_id)
    else:
        return '{}.jpg'.format(image_id)



def gmm_pdf(X, mixture, mu, sigma):
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
    from scipy.stats import multivariate_normal as mvn
    n_components = len(mixture)
    n_vals = len(X)
    
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
        mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
    
    return mixed_pdf
