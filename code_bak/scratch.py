import irsg_dw as dw

def chain_qry():
    import irsg_querygen as qgen
    chain_str = 'dog_walker holding leash attached_to dog'
    chain_obj = qgen.gen_two_rel_chain(chain_str)
    chain_tup = (chain_obj, chain_str)
    return chain_tup

def cycle_qry():
    import irsg_querygen as qgen
    cycle_str = 'dog_walker holding leash attached_to dog walked_by dog_walker'
    cycle_obj = qgen.gen_three_obj_loop(cycle_str)
    cycle_tup = (cycle_obj, cycle_str)
    return cycle_tup



def do_neg():
    gmms = dw.get_dw_gmms()
    
    query_tup = cycle_qry()
    out_dir = '/home/econser/School/research/output/'
    
    print('  Running negative images (No person, No dog)')
    dw.viz_plot_batch('neg_np_nd', query_tup, gmms, out_dir, verbose=True)
    
    print('  Running negative images (No person, dog)')
    dw.viz_plot_batch('neg_np_pd', query_tup, gmms, out_dir, verbose=True)
    
    print('  Running negative images (person, No dog)')
    dw.viz_plot_batch('neg_pp_nd', query_tup, gmms, out_dir, verbose=True)
    
    print('  Running negative images (person, dog)')
    dw.viz_plot_batch('neg_pp_pd', query_tup, gmms, out_dir, verbose=True)



"""
import scratch as sk
import irsg_dw as dw
gmms = dw.get_dw_gmms()
s_platt = sk.run_image(dw.dir_map['pos_test'], 'dog-walking350.csv', sk.cycle_qry(), gmms)
sk.hist(s_platt['attached_to'], bins=100)
gmms_ = dw.get_dw_gmms('dw_gmms_new.pkl')
s_new = sk.run_image(dw.dir_map['pos_test'], 'dog-walking350.csv', sk.cycle_qry(), gmms_)
sk.hist(s_new['attached_to'], bins=100)
"""
def run_image(image_dir, image_name, query=None, gmms=None):
    import os
    import sys
    
    if query is None:
        query = cycle_qry()
    
    classes = [obj.names for obj in query[0].objects]
    pg_comps = dw.get_pregen_components(image_name, image_dir, gmms, classes)
    scores = dw.get_all_scores(query[0], pg_comps)
    
    return scores



"""
import numpy as np
import matplotlib.pyplot as plt
n_bins = 500
bins, edges = np.histogram(scores, n_bins, normed=1)
left, right = edges[:-1], edges[1:]
X = np.array([left, right]).T.flatten()
Y = np.array([bins, bins]).T.flatten()
plt.plot(X, Y)
plt.show()
"""
def hist(scores, bins=500):
    import numpy as np
    import matplotlib.pyplot as plt
    
    n, bins, patches = plt.hist(scores, bins=bins, log=True)
    
    #bins, edges = np.histogram(scores, bins, normed=1)
    #left, right = edges[:-1], edges[1:]
    #X = np.array([left, right]).T.flatten()
    #Y = np.array([bins, bins]).T.flatten()
    #plt.plot(X, Y)
    #plt.semilogy(X, Y)
    
    plt.grid(True)
    plt.show()
