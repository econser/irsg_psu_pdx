import numpy as np



class ResultLine (object):
    def __init__(self, is_dog_walking, has_person, has_dog, filename, energy):
        if is_dog_walking == 'pos':
            self.is_dog_walking = True
        else:
            self.is_dog_walking = False
        
        if has_person == 'True':
            self.has_person = True
        else:
            self.has_person = False
        
        if has_dog == 'True':
            self.has_dog = True
        else:
            self.has_dog = False
        
        self.filename = filename
        self.energy = energy



"""
import plot_utils as p
import numpy as np
ranks = p.gen_ranks('/home/econser/School/research/output/energy_fullq.csv')
r_at_k = p.get_r_at_k(ranks, 400)
p.r_at_k_plot(r_at_k)
t = np.hstack(((np.arange(len(cyc_ratk))+1).reshape((400,1)),(cyc_ratk.reshape(400,1))))
np.savetxt('/home/econser/School/research/output/cyc_ratk.csv', t, fmt='%d, %03f')
"""
def gen_ranks(filename):
    import re
    
    pos_examples = []
    neg_examples = []
    
    f = open(filename)
    file_lines = f.readlines()
    for line in file_lines:
        t = re.split(', |,', line[:-1]) #line[:-1].split(',')
        
        if t[0] == 'pos':
            pos_examples.append([t[3], float(t[4])])
        else:
            neg_examples.append([t[3], float(t[4])])
    
    neg_examples = np.array(neg_examples, dtype=object)
    neg_examples = neg_examples[neg_examples[:,1].argsort()] # sort by energy
    
    # for each positive example
    pos_examples = np.array(pos_examples, dtype=object)
    ranks = np.searchsorted(neg_examples[:,1], pos_examples[:,1])
    ranks = ranks + 1 # searchsorted returns a 0-based value
    return ranks



def get_recalls(ranks, n_negatives):
    recalls = []
    for rank in ranks:
        not_found = np.zeros(rank - 1, dtype=np.float)
        found = np.ones(n_negatives - rank + 1, dtype=np.float)
        recall = np.hstack((not_found, found))
        recalls.append(recall)
    recalls = np.array(recalls)
    return recalls



def get_r_at_k(ranks, n_negatives):
    recalls = get_recalls(ranks, n_negatives)
    return np.average(recalls, axis=0)



def calc_median_ratk(in_file):
    ranks = gen_ranks(in_file)
    return np.median(ranks), np.average(ranks)



def r_at_k_plot(avg_r_at_k, filename=None, x_limit=None):
    import matplotlib.pyplot as plt
  
    plt.figure(1)
    plt.grid(True)
  
    plot_handle, = plt.plot(np.arange(len(avg_r_at_k)), avg_r_at_k)
  
    plt.xlabel("k")
    plt.ylabel("Recall at k")
  
    if x_limit is not None:
        plt.xlim([0, x_limit])
    plt.ylim([0, 1])
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    plt.close()



#-------------------------------------------------------------------------------
# VIZ PLOT SECTION
#
def draw_best_objects(image_dir, comps, best_box_ixs, energy, out_dir="", out_filename="", image_size=[]):
    import randomcolor
    
    class_names = [uc.name for uc in comps.unary_components]
    n_objects = len(class_names)
    
    rc = randomcolor.RandomColor()
    colorset = rc.generate(luminosity='bright', count=n_objects, format_='rgb')
    color_list = []
    for i in range(0, n_objects):
        color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
        color_array = color_array * (1. / 255.)
        color_list.append(color_array)
    
    legend_list = zip(class_names, color_list)
  
    box_list = []
    for i in range(0, n_objects):
        bbox_ix = best_box_ixs[i]
        box_coords = comps.unary_components[i].boxes[bbox_ix]
        box_and_color = np.hstack((box_coords, color_list[i]))
        box_list.append(box_and_color)
    
    title = ""
    #if energy != None:
    #    title = "Object Detections (energy={0:.3f})".format(energy)
    
    image_filename = '{}.jpg'.format(comps.image_filename.split('.')[0])
    out_filename = 'en:{:06.3f} -- {}.jpg'.format(energy, comps.image_filename.split('.')[0])
    draw_image_box(image_dir, image_filename, box_list, legend_list, title, out_dir + out_filename, size=image_size)



"""
base_dir = '/home/econser/School/research/'
p.draw_best_pregen_boxes(base_dir+'run_results/dogwalking, negative, nodognoperson/', base_dir+'images/psu_dw/DogWalkingSituationNegativeExamples/NoDogNoPerson/', base_dir+'output/viz_neg_rcnn/', 'no_dog_no_person_')
p.draw_best_pregen_boxes(base_dir+'run_results/dogwalking, negative, dognoperson/', base_dir+'images/psu_dw/DogWalkingSituationNegativeExamples/DogNoPerson/', base_dir+'output/viz_neg_rcnn/', 'dog_no_person_')
p.draw_best_pregen_boxes(base_dir+'run_results/dogwalking, negative, dogandperson/', base_dir+'images/psu_dw/DogWalkingSituationNegativeExamples/DogAndPerson/', base_dir+'output/viz_neg_rcnn/', 'dog_and_person_')
p.draw_best_pregen_boxes(base_dir+'run_results/dogwalking, negative, personnodog/', base_dir+'images/psu_dw/DogWalkingSituationNegativeExamples/PersonNoDog/', base_dir+'output/viz_neg_rcnn/', 'person_no_dog_')
"""
def draw_best_pregen_boxes(csv_dir, image_dir, out_dir, out_file_prefix):
    import randomcolor
    import os
    
    # pull the class dirs from the csv dir
    classes = [name for name in os.listdir(csv_dir)]
    n_objects = len(classes)
    
    rc = randomcolor.RandomColor()
    colorset = rc.generate(luminosity='bright', count=n_objects, format_='rgb')
    color_list = []
    for i in range(0, n_objects):
        color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
        color_array = color_array * (1. / 255.)
        color_list.append(color_array)
    
    legend_list = zip(classes, color_list)
    
    for filename in os.listdir(csv_dir + classes[0]):
        box_list = []
        for class_ix, class_name in enumerate(classes):
            boxes = np.genfromtxt(csv_dir + class_name + '/' + filename, delimiter=',')
            best_box_ix = boxes[:,4].argmax()
            box_and_color = np.hstack((boxes[best_box_ix,0:4], color_list[class_ix]))
            box_list.append(box_and_color)
        csv_as_image = filename.split('.')[0] + '.jpg'
        fq_out = out_dir + out_file_prefix + csv_as_image
        title = ""
        draw_image_box(image_dir, csv_as_image, box_list, legend_list, title, fq_out)



def draw_image_box(image_dir, image_filename, box_list, legend_list, title="", out_filename="", size=[]):
    import os.path
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects
    #plt.switch_backend('Qt4Agg')
    
    fq_image_filename = image_dir
    if image_dir[-1] != '/':
        fq_image_filename += '/'
    fq_image_filename += image_filename
    
    image_is_local = os.path.isfile(fq_image_filename)
    if not image_is_local:
        return
  
    img = Image.open(fq_image_filename)
    img_array = np.array(img, dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    for i in range(0, len(box_list)):
        box = box_list[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        c = box[4:7]
        box = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor=c, facecolor='none')
        ax.add_patch(box)
        txt = ax.text(x+5, y+5, legend_list[i][0], va='top', size=16, weight='bold', color='0.1')
        txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='w')])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    if len(title) > 0:
        plt.title(title)
    
    handle_list = []
    name_list = []
    for i in range(0, len(legend_list)):
        h = plt.Rectangle((0,0),0.125,0.125, fc=legend_list[i][1])
        handle_list.append(h)
        name_list.append(legend_list[i][0])
    #plt.legend(handle_list, name_list, bbox_to_anchor=(1.14, 1.01))#, loc='upper right')
    
    plt.tight_layout(pad=7.5)
    
    if len(out_filename) == 0:
        #plt.show(bbox_inches='tight')
        plt.show()
    else:
        plt.rcParams.update({'font.size': 6})
        plt.savefig(out_filename, dpi=175)
    
    plt.clf()
    plt.close()



"""
    Visualize samples drawn from a relationship GMM
    In:
    image_name: filename of image to sample against
    gmm: gmm dict

import irsg_dw as dw
g = dw.get_dw_gmms('dw_gmms_p5_mod.pkl')
import plot_utils as pu
reload(pu); pu.sample_rel_model('dog-walking1.jpg', g, n_samples = 10000)
"""
def sample_rel_model(image_filename, gmms, relation_name, n_samples=100, input_dir='/home/econser/School/research/data/dog_walking/', output_dir='/home/econser/School/research/', output_filename=''):
    from PIL import Image
    import irsg_utils as iu
    import matplotlib.pyplot as plt
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.patches as patches
    
    anno_file = image_filename.split('.')[0] + '.labl'
    h_anno = open(input_dir + anno_file, 'rb')
    line = h_anno.readlines()
    walker, dog, leash = iu.get_dw_boxes(line[0])
    
    samples = gmms[relation_name].model.sample(n_samples)
    sbox = samples[0]
    
    if relation_name == 'holding':
        sub = walker
        obj = leash
    elif relation_name == 'attached_to':
        sub = leash
        obj = dog
    elif relation_name == 'walked_by':
        sub = dog
        obj = walker
    
    obj_w = sbox[:,2] * sub[2]
    obj_x = sub[0] + 0.5 * sub[2] - sub[2] * sbox[:,0] - 0.5 * obj_w
    obj_h = sbox[:,3] * sub[3]
    obj_y = sub[1] + 0.5 * sub[3] - sub[3] * sbox[:,1] - 0.5 * obj_h
    obj_samples = np.vstack((obj_x, obj_y, obj_w, obj_h)).T
    """
    leash_w = sbox[:,2] * walker[2]
    leash_x = walker[0] + 0.5 * walker[2] - walker[2] * sbox[:,0] - 0.5 * leash_w
    leash_h = sbox[:,3] * walker[3]
    leash_y = walker[1] + 0.5 * walker[3] - walker[3] * sbox[:,1] - 0.5 * leash_h
    leash_samples = np.vstack((leash_x, leash_y, leash_w, leash_h)).T
    """
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
    
    if len(output_filename) > 0:
        plt.savefig(filename+'{}/{}_{}_{}.png'.format(output_dir, object_name, relation_name, subject_name), dpi=175)
    else:
        plt.show()
