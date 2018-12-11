"""
    A labl annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...

    A json annotation looks like:
{
    "im_w":<INT>,
    "im_h":<INT>,
    "objects":[
	{
	    "desc":<STR>,
	    "box_xywh":[<INT>, <INT>, <INT>, <INT>]
	},
        {
            <OBJ_2>
        },
        ...
        {
            <OBJ_n>
        }
    ]
}

    returns: dict of object names and their GT bboxes
             when multiple objects of the same desc are present, they are 
             auto-indexed with a '__<INT>' suffix on the object desc:
             person__1, person__2, etc.
"""
import numpy as np



"""
import json as j
import gen_bboxes as g

prs_j = g.get_bbox_fn('person_wearing_glasses', 'json')
j_file = open('/home/econser/research/irsg_psu_pdx/data/PersonWearingGlasses/PersonWearingGlassesTrain/person-wearing-glasses1.json', 'rb')
j_obj = j.load(j_file)
j_out = prs_j(j_obj)

prs_l = g.get_bbox_fn('person_wearing_glasses', 'labl')
l_file = open('/home/econser/research/irsg_psu_pdx/data/PersonWearingGlasses/PersonWearingGlassesTrain/person-wearing-glasses1.labl', 'rb')
l_obj = l_file.readlines()[0]
l_out = prs_l(l_obj)
"""


#===============================================================================
def get_bbox_fn(model_name, anno_type):
    parse_fn_dict = {
        'labl' : labl_parse_fn,
        'json' : json_parse_fn
        }

    anno_map_fn_dict = {
        'dog_walking' : dw_anno_map,
        'ping_pong' : pp_anno_map,
        'handshake' : hs_anno_map,
        'leading_horse' : lh_anno_map,
        'person_wearing_glasses' : pwg_anno_map,
        'full_minivg' : mvg_anno_map
        }

    parse_fn = parse_fn_dict.get(anno_type)
    anno_map_fn = anno_map_fn_dict.get(model_name)
    
    if model_name == 'full_minivg':
        return (lambda anno: minivg(anno_map_fn, parse_fn, anno))
    
    return (lambda anno: generic_bbox_fn(anno_map_fn, parse_fn, anno))



def minivg(map_fn, parse_fn, annotation):
    objects, bboxes = parse_fn(annotation)
    
    # check to see if we need to pair up objects
    do_pairing = False
    for obj_name in objects:
        if '_' in obj_name:
            do_pairing = True
            break
        
    if do_pairing:
        # we're assuming 2 types of objects with '<obj_name>__<index>' format
        num_pairs = len(objects) / 2
        # split the objects into obj_name, ix, bbox tuples
        obj_names = []
        sit_ixs = []
        bbox_list = []
        for anno_ix, obj in enumerate(objects):
            name, ix = obj.split('_')
            ix = int(ix)
            obj_names.append(name.lower())
            sit_ixs.append(ix)
            bbox_list.append(bboxes[anno_ix])
        sit_ixs = np.array(sit_ixs)
        
        # now pair up the rel pairs
        obj_dicts = []
        for pair_ix in range(num_pairs):
            ref_ix = np.where(sit_ixs-1 == pair_ix)[0]
            obj_dict = {}
            obj_dict[obj_names[ref_ix[0]]] = bbox_list[ref_ix[0]]
            obj_dict[obj_names[ref_ix[1]]] = bbox_list[ref_ix[1]]
            
            obj_dicts.append(obj_dict)
        
        return obj_dicts
    else:
        for obj_ix, obj_name in enumerate(objects):
            objects[obj_ix] = map_fn(obj_name)

        objects = deconflict(objects)
    
        # zip class names and bboxes
        obj_dict = dict(zip(objects, bboxes))
        return [obj_dict]



def generic_bbox_fn(map_fn, parse_fn, annotation):
    objects, bboxes = parse_fn(annotation)
    
    # convert annotations to class names
    for obj_ix, obj_name in enumerate(objects):
        objects[obj_ix] = map_fn(obj_name)

    objects = deconflict(objects)
    
    # zip class names and bboxes
    obj_dict = dict(zip(objects, bboxes))
    return obj_dict



def deconflict(name_list):
    cpy = list(name_list)
    
    ready = []
    for ix in range(0, len(name_list)):
        ready.append(False)
    
    for i, i_name in enumerate(cpy):
        if ready[i]:
            continue
        
        for j, j_name in enumerate(cpy[i+1:]):
            if j_name == i_name:
                # there's a match, dedupe
                suffix_num = 1
                for k in range(i, len(cpy)):
                    k_name = name_list[k]
                    if k_name == i_name:
                        cpy[k] = '{}__{}'.format(i_name, suffix_num)
                        ready[k] = True
                        suffix_num += 1
        ready[i] = True
    
    return cpy



#-------------------------------------------------------------------------------
def labl_parse_fn(annotation):
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



def json_parse_fn(annotation):
    names = []
    bboxes = []
    for o in annotation['objects']:
        names.append(o['desc'])
        bboxes.append(np.array(o['box_xywh'], dtype=np.int))

    return names, bboxes



#-------------------------------------------------------------------------------
def generic_anno_map(obj_name):
    return obj_name.split(' ')[0]



def mvg_anno_map(obj_name):
    obj = obj_name.split(' ')[0]
    obj =  obj.split('_')[0]
    # check for multi
    return obj.lower()



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
