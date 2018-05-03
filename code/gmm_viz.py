from __future__ import print_function
import sys
import numpy as np
from PIL import Image
import irsg_utils as iu
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.patches as patches

"""
import cPickle
f = open('/home/econser/School/research/data/pingpong_gmms.pkl', 'rb')
gmms = cPickle.load(f)
import irsg_utils as iu
import plot_utils as pu
pu.sample_rel_model_('pingpong92.jpg', gmms, 'playing_pingpong_with', 'player__1', 'player__2', iu.get_pp_bboxes, input_dir='/home/econser/School/research/data/PingPong/')
img_name = 'pingpong302.jpg'
samples = 1000
pu.sample_rel_model_(img_name, gmms, 'playing_pingpong_with', 'player__1', 'player__2', iu.get_pp_bboxes, n_samples=samples, input_dir='/home/econser/School/research/data/PingPong/')
pu.sample_rel_model_(img_name, gmms, 'playing_pingpong_with', 'player__2', 'player__1', iu.get_pp_bboxes, n_samples=samples, input_dir='/home/econser/School/research/data/PingPong/')
pu.sample_rel_model_(img_name, gmms, 'at', 'player__1', 'table', iu.get_pp_bboxes, n_samples=samples, input_dir='/home/econser/School/research/data/PingPong/')
pu.sample_rel_model_(img_name, gmms, 'at', 'player__2', 'table', iu.get_pp_bboxes, n_samples=samples, input_dir='/home/econser/School/research/data/PingPong/')
pu.sample_rel_model_(img_name, gmms, 'on', 'net', 'table', iu.get_pp_bboxes, n_samples=samples, input_dir='/home/econser/School/research/data/PingPong/')

import cPickle
f = open('/home/econser/School/research/data/handshake_gmms.pkl', 'rb')
gmms = cPickle.load(f)
import irsg_utils as iu
import plot_utils as pu
pu.sample_rel_model_('handshake90.jpg', gmms, 'extending', 'person__1', 'person__2', iu.get_hs_boxes, input_dir='/home/econser/School/research/data/Handshake/')
"""
def sample_rel_model_(image_filename, gmms, relation_name, sub_class_name, obj_class_name, box_gen_fn, n_samples=100, input_dir='/home/econser/School/research/data/dog_walking/', output_dir='/home/econser/School/research/', output_filename=''):
    from PIL import Image
    import irsg_utils as iu
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.patches as patches
    
    anno_file = image_filename.split('.')[0] + '.labl'
    h_anno = open(input_dir + anno_file, 'rb')
    line = h_anno.readlines()
    bboxes = box_gen_fn(line[0])
    
    samples = gmms[relation_name].model.sample(n_samples)
    sbox = samples[0]
    
    sub = bboxes[sub_class_name]
    obj = bboxes[obj_class_name]
    
    obj_w = sbox[:,2] * sub[2]
    obj_x = sub[0] + 0.5 * sub[2] - sub[2] * sbox[:,0] - 0.5 * obj_w
    obj_h = sbox[:,3] * sub[3]
    obj_y = sub[1] + 0.5 * sub[3] - sub[3] * sbox[:,1] - 0.5 * obj_h
    obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
    
    img = Image.open(input_dir + image_filename).convert("L")
    img_array = np.array(img)
  
    # generate the detections map
    img_width = (img_array.shape)[1]
    img_height = (img_array.shape)[0]
    box_map = np.zeros((img_width, img_height), dtype=np.float)
    
    for i in range(0, len(obj_samples)):
        box = obj_samples[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        
        # skip boxes completely outside the image bounds
        if w < 0 or h < 0:
            continue
        if x+w <= 0 or y+h <= 0:
            continue
        if x > img_width or y > img_height:
            continue
        
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > img_width - 1:
            w -= x + w - img_width
        if y + h > img_height - 1:
            h -= y + h - img_height
            
        p_box = np.ones(w*h)
        p_box = np.reshape(p_box, (w,h))
        #print('{}: box={}, clip=({},{},{},{}) p_box={} img_w={} img_h={}'.format(i,box,x,y,w,h,p_box.shape,img_width,img_height))
        box_map[x:x+w, y:y+h] = box_map[x:x+w, y:y+h] + p_box
    
    fig, ax = plt.subplots(1)
    plt.xticks([])
    plt.yticks([])
    
    sub_box = patches.Rectangle((sub[0],sub[1]),sub[2],sub[3], linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(sub_box)
    
    obj_box = patches.Rectangle((obj[0],obj[1]),obj[2],obj[3], linewidth=3, edgecolor='green', facecolor='none')
    ax.add_patch(obj_box)
    
    plt.imshow(img_array, cmap='gray')
    box_map_blur = gaussian_filter(box_map, sigma=7)
    plt.imshow(box_map_blur.T, alpha=0.4)
    plt.tight_layout()
    
    plt.title('{} {} {}'.format(sub_class_name, relation_name, obj_class_name))
    
    if len(output_filename) > 0:
        plt.savefig(filename+'{}/{}_{}_{}.png'.format(output_dir, object_name, relation_name, subject_name), dpi=175)
    else:
        plt.show()



#===============================================================================
# GLOBALS
g_sub = None
g_sub_box = None
g_sub_box_patch = None
g_box_map = None
g_box_map_plt = None

g_gmms = None
g_current_rel_ix = 0

box_move_amount = 5
box_scale_amount = 5



#===============================================================================
def up():
    global g_box_map
    g_sub[1] -= box_move_amount
    g_sub_box_patch.set_xy((g_sub[0], g_sub[1]))
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def left():
    global g_box_map
    g_sub[0] -= box_move_amount
    g_sub_box_patch.set_xy((g_sub[0], g_sub[1]))
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def down():
    global g_box_map
    g_sub[1] += box_move_amount
    g_sub_box_patch.set_xy((g_sub[0], g_sub[1]))
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def right():
    global g_box_map
    g_sub[0] += box_move_amount
    g_sub_box_patch.set_xy((g_sub[0], g_sub[1]))
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def ctrl_up():
    global g_box_map
    g_sub[3] -= box_scale_amount
    g_sub_box_patch.set_height(g_sub[3])
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def ctrl_left():
    global g_box_map
    g_sub[2] -= box_scale_amount
    g_sub_box_patch.set_width(g_sub[2])
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def ctrl_down():
    global g_box_map
    g_sub[3] += box_scale_amount
    g_sub_box_patch.set_height(g_sub[3])
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def ctrl_right():
    global g_box_map
    g_sub[2] += box_scale_amount
    g_sub_box_patch.set_width(g_sub[2])
    
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    g_box_map_plt.set_data(g_box_map.T)

def r():
    print('refresh samples')

def prev_rel():
    print('use previous relationship')

def next_rel():
    global g_gmms
    global g_current_rel_ix
    
    relation_names = list(g_gmms.keys())
    g_current_rel_ix += 1
    g_current_rel_ix = (g_current_rel_ix % len(relation_names))
    relation_name = relation_names[g_current_rel_ix]
    samples = g_gmms[relation_name].model.sample(1000)#n_samples)
    print('updating samples')
    g_samples = samples[0]

def space():
    print('space')

def noop():
    pass
    #print('noop')



def update_heatmap(samples, sub_bbox, box_map):
    obj_w = samples[:,2] * sub_bbox[2]
    obj_x = sub_bbox[0] + 0.5 * sub_bbox[2] - sub_bbox[2] * samples[:,0] - 0.5 * obj_w
    obj_h = samples[:,3] * sub_bbox[3]
    obj_y = sub_bbox[1] + 0.5 * sub_bbox[3] - sub_bbox[3] * samples[:,1] - 0.5 * obj_h
    obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
  
    # generate the detections map
    img_width = (box_map.shape)[0]
    img_height = (box_map.shape)[1]
    box_map *= 0. #np.zeros((img_width, img_height), dtype=np.float)
    
    for i in range(0, len(obj_samples)):
        box = obj_samples[i]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        
        # skip boxes completely outside the image bounds
        if w < 0 or h < 0:
            continue
        if x+w <= 0 or y+h <= 0:
            continue
        if x > img_width or y > img_height:
            continue
        
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > img_width - 1:
            w -= x + w - img_width
        if y + h > img_height - 1:
            h -= y + h - img_height
            
        p_box = np.ones(w*h)
        p_box = np.reshape(p_box, (w,h))
        box_map[x:x+w, y:y+h] = box_map[x:x+w, y:y+h] + 1.0 #p_box
    return box_map



#===============================================================================
g_key_fn_map = {
    'up' : up,
    'left' : left,
    'down' : down,
    'right' : right,
    'ctrl+up' : ctrl_up,
    'ctrl+left' : ctrl_left,
    'ctrl+down' : ctrl_down,
    'ctrl+right' : ctrl_right,
    'r' : r,
    '[' : prev_rel,
    ']' : next_rel,
    ' ' : space}



def press(event):
    fn = g_key_fn_map.get(event.key, noop)
    fn()
    #TODO: dont're redraw on every key?
    fig.canvas.draw()
    sys.stdout.flush()



def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='Scene graph situations')
    parser.add_argument('--img', dest='image_filename', help='The image file to use')
    parser.add_argument('--samples', dest='n_samples', help='samples per viz', default=1000)
    parser.add_argument('--cfg', dest='cfg_file', help='configuration param file (.yml)')
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    import cPickle
    
    args = parse_args()
    image_filename = args.image_filename
    n_samples = args.n_samples
    # TODO: read cfg_file
    image_dir = '/home/econser/School/research/data/PingPong/' #cfg['image_dir']
    gmm_filepath = '/home/econser/School/research/data/pingpong_gmms.pkl' #cfg['gmm_filepath']
    
    # unpickle the GMMs
    f = open(gmm_filepath, 'rb')
    g_gmms = cPickle.load(f)
    
    # pull relation keys from gmms
    relation_names = list(g_gmms.keys())
    g_current_rel_ix = 1
    relation_name = relation_names[g_current_rel_ix]
    samples = g_gmms[relation_name].model.sample(n_samples)
    g_samples = samples[0]
    
    # draw the base image
    img = Image.open(image_dir + image_filename).convert("L")
    img_array = np.array(img)
    img_width = (img_array.shape)[1]
    img_height = (img_array.shape)[0]
    
    #
    fig, ax = plt.subplots(1)
    fig.canvas.mpl_connect('key_press_event', press)
    plt.xticks([])
    plt.yticks([])
    
    g_sub = np.array((0., 0., 200., 200.))
    g_sub_box = patches.Rectangle((g_sub[0], g_sub[1]), g_sub[2], g_sub[3], linewidth=3, edgecolor='red', facecolor='none')
    g_sub_box_patch = ax.add_patch(g_sub_box)
    
    plt.imshow(img_array, cmap='gray')
    plt.tight_layout()
    
    plt.title('{}'.format(relation_name))
    
    # generate object boxes from subject box and samples
    g_box_map = np.zeros((img_width, img_height), dtype=np.float)
    #g_box_map = np.zeros((img_height, img_width), dtype=np.float)
    g_box_map = update_heatmap(g_samples, g_sub, g_box_map)
    
    # first heatmap setup here
    box_map_blur = gaussian_filter(g_box_map, sigma=7)
    g_box_map_plt = plt.imshow(box_map_blur.T, alpha=0.4)
    
    plt.show()
    
    #fig, ax = plt.subplots()
    #fig.canvas.mpl_connect('key_press_event', press)
    #ax.plot(np.random.rand(12), np.random.rand(12), 'go')
    #xl = ax.set_xlabel('easy come, easy go')
    #ax.set_title('Press a key')
    #plt.show()
