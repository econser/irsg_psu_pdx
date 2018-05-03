import numpy as np
import scipy.io.matlab.mio5_params as siom

#===============================================================================
# Scenegraph generation functions
#
# query
#   objects: np.ndarray(n, dtype=object)
#     names
#   unary_triples:
#     subject: 1
#     predicate: 'is'
#     object: 'red'
#   binary_triples: np.ndarray(n, dytpe=object)
#     subject: 0
#     predicate: 'wearing'
#     object: 1
#===============================================================================
def gen_sro(query_str):
    words = query_str.split()
    if len(words) != 3: return None
    
    sub_struct = siom.mat_struct()
    sub_struct.__setattr__('names', words[0])
    
    obj_struct = siom.mat_struct()
    obj_struct.__setattr__('names', words[2])
    
    rel_struct = siom.mat_struct()
    rel_struct.__setattr__('subject', 0)
    rel_struct.__setattr__('predicate', words[1])
    rel_struct.__setattr__('object', 1)
    
    det_list = np.array([sub_struct, obj_struct], dtype=np.object)
    query_struct = siom.mat_struct()
    query_struct.__setattr__('objects', det_list)
    query_struct.__setattr__('unary_triples', np.array([]))
    query_struct.__setattr__('binary_triples', rel_struct)
    
    return query_struct



def gen_asro(query_str):
    words = query_str.split()
    if len(words) != 4: return None
    
    sub_attr_struct = siom.mat_struct()
    sub_attr_struct.__setattr__('subject', 0)
    sub_attr_struct.__setattr__('predicate', 'is')
    sub_attr_struct.__setattr__('object', words[0])
    
    query_struct = gen_sro(' '.join([words[1], words[2], words[3]]))
    query_struct.__setattr__('unary_triples', sub_attr_struct)
    
    return query_struct



def gen_srao(query_str):
    words = query_str.split()
    if len(words) != 4: return None
    
    obj_attr_struct = siom.mat_struct()
    obj_attr_struct.__setattr__('subject', 1)
    obj_attr_struct.__setattr__('predicate', 'is')
    obj_attr_struct.__setattr__('object', words[2])
    
    query_struct = gen_sro(' '.join([words[0], words[1], words[3]]))
    query_struct.__setattr__('unary_triples', obj_attr_struct)
    
    return query_struct



def gen_asrao(query_str):
    words = query_str.split()
    if len(words) != 5: return None
    
    obj_attr_struct = siom.mat_struct()
    obj_attr_struct.__setattr__('subject', 1)
    obj_attr_struct.__setattr__('predicate', 'is')
    obj_attr_struct.__setattr__('object', words[3])
    
    query_struct = gen_asro(' '.join([words[0], words[1], words[2], words[4]]))
    query_struct.__setattr__('unary_triples', obj_attr_struct)
    
    return query_struct



"""
    dog_walker holding leash attached_to dog
    ob1        rel1    ob2   rel2        ob3
"""
def gen_two_rel_chain(query_str):
    words = query_str.split()
    if len(words) != 5: return None
    
    # generate the object list
    ob1_struct = siom.mat_struct()
    ob1_struct.__setattr__('names', words[0])
    
    ob2_struct = siom.mat_struct()
    ob2_struct.__setattr__('names', words[2])
    
    ob3_struct = siom.mat_struct()
    ob3_struct.__setattr__('names', words[4])
    
    ob_list = np.array([ob1_struct, ob2_struct, ob3_struct])
    
    # generate the first relation
    rel1_struct = siom.mat_struct()
    rel1_struct.__setattr__('subject', 0)
    rel1_struct.__setattr__('predicate', words[1])
    rel1_struct.__setattr__('object', 1)
    
    #rel1_objs = np.array([ob1_struct, ob2_struct], dtype=np.object)
    
    # generate the second relation
    rel2_struct = siom.mat_struct()
    rel2_struct.__setattr__('subject', 1)
    rel2_struct.__setattr__('predicate', words[3])
    rel2_struct.__setattr__('object', 2)
    
    #rel2_objs = np.array([ob2_struct, ob3_struct], dtype=np.object)
    
    # store the objects and relations
    rel_list = np.array([rel1_struct, rel2_struct])
    
    query_struct = siom.mat_struct()
    query_struct.__setattr__('objects', ob_list)
    query_struct.__setattr__('unary_triples', np.array([]))
    query_struct.__setattr__('binary_triples', rel_list)
    
    return query_struct



"""
    0          1       2     3           4   5    6
    dog_walker holding leash attached_to dog near dog_walker
    ob1        rel1    ob2   rel2        ob3 rel3 ob1
    
    expects this exact order of objects and relationships
"""
def gen_three_obj_loop(query_str):
    words = query_str.split()
    
    if len(words) != 7: return None
    if words[0] != words[6]: return None
    
    # generate the object list
    ob1_struct = siom.mat_struct()
    ob1_struct.__setattr__('names', words[0])
    
    ob2_struct = siom.mat_struct()
    ob2_struct.__setattr__('names', words[2])
    
    ob3_struct = siom.mat_struct()
    ob3_struct.__setattr__('names', words[4])
    
    ob_list = np.array([ob1_struct, ob2_struct, ob3_struct])
    
    # generate the first relation
    rel1_struct = siom.mat_struct()
    rel1_struct.__setattr__('subject', 0)
    rel1_struct.__setattr__('predicate', words[1])
    rel1_struct.__setattr__('object', 1)
    
    # generate the second relation
    rel2_struct = siom.mat_struct()
    rel2_struct.__setattr__('subject', 1)
    rel2_struct.__setattr__('predicate', words[3])
    rel2_struct.__setattr__('object', 2)
    
    # generate the third relation
    rel3_struct = siom.mat_struct()
    rel3_struct.__setattr__('subject', 2)
    rel3_struct.__setattr__('predicate', words[5])
    rel3_struct.__setattr__('object', 0)

    # store the objects and relations
    rel_list = np.array([rel1_struct, rel2_struct, rel3_struct])
    
    query_struct = siom.mat_struct()
    query_struct.__setattr__('objects', ob_list)
    query_struct.__setattr__('unary_triples', np.array([]))
    query_struct.__setattr__('binary_triples', rel_list)
    
    return query_struct
