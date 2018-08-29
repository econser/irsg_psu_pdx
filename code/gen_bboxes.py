"""
    A labl annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...
    returns: dict of object names and their GT bboxes
"""
import numpy as np



#===============================================================================
def tokenize(annotation):
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    n_coords = len(tokens) - n_boxes
    coords = tokens[3 : n_coords]
    
    objects = tokens[-n_boxes:]
    bboxes = []
    
    for obj_ix in range(0, len(objects)):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        
        coord_slice = coords[start_ix : end_ix]
        bbox = np.array(coord_slice, dtype=np.int)
        bboxes.append(bbox)
    
    return objects, bboxes



#===============================================================================
def get_bbox_fn_(model_name):
    if model_name == 'dog_walking':
        return (lambda anno: generic_bbox_fn(dw_anno_map, anno))
    elif model_name == 'ping_pong':
        return(lambda anno:  generic_bbox_fn(pp_anno_map, anno))
    elif model_name == 'handshake':
        return (lambda anno: generic_bbox_fn(hs_anno_map, anno))
    elif model_name == 'leading_horse':
        return (lambda anno: generic_bbox_fn(lh_anno_map, anno))
    elif model_name == 'person_wearing_glasses':
        return (lambda anno: generic_bbox_fn(pwg_anno_map, anno))
    elif model_name == 'generic':
        return (lambda anno: generic_bbox_fn(generic_anno_map, anno))
    else:
        return None



def generic_bbox_fn(map_fn, annotation):
    objects, bboxes = tokenize(annotation)
    
    # convert annotations to class names
    for obj_ix, obj_name in enumerate(objects):
        objects[obj_ix] = map_fn(obj_name)
    
    # update class names for multiple objects
    ready = []
    for ix in range(0, len(objects)):
        ready.append(False)
    
    for i, i_name in enumerate(objects):
        if ready[i]:
            continue
        
        for j, j_name in enumerate(objects[i+1:]):
            if j_name == i_name:
                # there's a match, dedupe
                suffix_num = 1
                for k in range(i, len(objects)):
                    k_name = objects[k]
                    if k_name == i_name:
                        objects[k] = '{}__{}'.format(i_name, suffix_num)
                        ready[k] = True
                        suffix_num += 1
        ready[i] = True
    
    # zip class names and bboxes
    obj_dict = dict(zip(objects, bboxes))
    return obj_dict



def generic_anno_map(obj_name):
    return obj_name.split(' ')[0]



def dw_anno_map(obj_name):
    if obj_name.startswith('dog-walker'):
        return 'dog_walker'
    elif obj_name.startswith('dog'):
        return 'dog'
    elif obj_name.startswith('leash'):
        return 'leash'
    else:
        return obj_name.split(' ')[0]



def pp_anno_map(obj_name):
    if obj_name.startswith('player-'):
        return 'player'
    elif obj_name.startswith('net'):
        return 'net'
    elif obj_name.startswith('table'):
        return 'table'
    else:
        return obj_name.split(' ')[0]



def hs_anno_map(obj_name):
    if obj_name.startswith('person-'):
        return 'person'
    elif obj_name.startswith('handshake'):
        return 'handshake'
    else:
        return obj_name.split(' ')[0]



def lh_anno_map(obj_name):
    if obj_name.startswith('horse-leader'):
        return 'horse-leader'
    elif obj_name.startswith('horse'):
        return 'horse'
    elif obj_name.startswith('lead'):
        return 'lead'
    else:
        return obj_name.split(' ')[0]



def pwg_anno_map(obj_name):
    if obj_name.startswith('person'):
        return 'person'
    elif obj_name.startswith('glasses'):
        return 'glasses'
    else:
        return obj_name.split(' ')[0]
