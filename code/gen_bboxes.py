"""
    An annotation looks like:
    width | height | nBoxes | x0 | y0 | w0 | h0 | x1 | y1 | w1 | h1 |...| label0 | label1 ...
    returns: dict of object names and their GT bboxes
"""
import numpy as np



#===============================================================================
def get_bbox_anno_fn(model_name):
    if model_name == 'dog_walking':
        return get_dw_boxes
    elif model_name == 'ping_pong':
        return get_pp_bboxes
    elif model_name == 'handshake':
        return get_hs_bboxes
    elif model_name == 'leading_horse':
        return get_lh_bboxes
    elif model_name == 'person_wearing_glasses':
        return get_pwg_bboxes
    else:
        return None



def tokenize(annotation):
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    objects = tokens[-n_boxes:]
    n_coords = len(tokens) - n_boxes
    boxes = tokens[3 : n_coords]

    return boxes



#===============================================================================
def get_dw_bboxes(annotation):
    boxes = tokenize(annotation)
    
    obj_dict = {}
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('dog-walker'):
            obj_dict['dog_walker'] = coords
        elif obj_name.startswith('dog'):
            obj_dict['dog'] = coords
        elif obj_name.startswith('leash'):
            obj_dict['leash'] = coords
        else:
            obj_dict[obj_name.split(' ')[0]] = coords
    
    return obj_dict



def get_pp_bboxes(annotation):
    import numpy as np
    
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    objects = tokens[-n_boxes:]
    n_coords = len(tokens) - n_boxes
    boxes = tokens[3 : n_coords]
    
    obj_dict = {}
    
    player_instance = 1
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('player-'):
            obj_dict['player__{}'.format(player_instance)] = coords.copy()
            player_instance += 1
        elif obj_name.startswith('net'):
            obj_dict['net'] = coords.copy()
        elif obj_name.startswith('table'):
            obj_dict['table'] = coords.copy()
        else:
            obj_dict[obj_name.split(' ')[0]] = coords.copy()
    
    return obj_dict



def get_hs_bboxes(annotation):
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



def get_lh_bboxes(annotation):
    import numpy as np
    
    tokens = annotation.split('|')
    n_boxes = int(tokens[2])
    objects = tokens[-n_boxes:]
    n_coords = len(tokens) - n_boxes
    boxes = tokens[3 : n_coords]
    
    obj_dict = {}
    
    for obj_ix, obj_name in enumerate(objects):
        start_ix = obj_ix * 4
        end_ix = start_ix + 4
        coords = boxes[start_ix : end_ix]
        coords = np.array(coords, dtype=np.int)
        
        if obj_name.startswith('horse-leader'):
            obj_dict['horse-leader'] = coords.copy()
        elif obj_name.startswith('horse'):
            obj_dict['horse'] = coords.copy()
        elif obj_name.startswith('lead'):
            obj_dict['lead'] = coords.copy()
        else:
            obj_dict[obj_name.split(' ')[0]] = coords.copy()
    
    if not obj_dict.has_key('horse'):
        import pdb; pdb.set_trace()
    if not obj_dict.has_key('horse-leader'):
        import pdb; pdb.set_trace()
    if not obj_dict.has_key('lead'):
        import pdb; pdb.set_trace()
    return obj_dict
