from __future__ import print_function
import matplotlib; matplotlib.use('agg') #when running remotely
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
    This is a (terribly named) box-and-whisker plot generator
    The name will be fixed soon!
"""
def bnw():
    import csv
    import os.path
    import matplotlib.pyplot as plt
    
    base_dir = '/home/econser/research/irsg_psu_pdx/output/full_runs/dog_walking'
    pos_data_fname = 'dog_walking_postest_factors.csv'
    pos_color = 'lightgreen'
    neg_data_fname = 'dog_walking_hardneg_factors.csv'
    neg_color = 'tomato'
    
    # read in the header
    f = open(os.path.join(base_dir, pos_data_fname), 'rb')
    r = csv.reader(f)
    header = r.next()
    f.close()
    
    # read in the np arrays
    neg_data = np.genfromtxt(os.path.join(base_dir, neg_data_fname), delimiter=', ', dtype=np.object, skip_header=True)
    
    pos_data = np.genfromtxt(os.path.join(base_dir, pos_data_fname), delimiter=', ', dtype=np.object, skip_header=True)
    
    # interleave the pos and neg columns
    data = []
    labels = []
    colors = []
    for col_ix in range(0, len(pos_data[0])):
        if col_ix == 0:
            continue
        
        data.append(np.array(pos_data[:, col_ix], dtype=np.float))
        labels.append('pos_{}'.format(header[col_ix]))
        colors.append(pos_color)
        
        data.append(np.array(neg_data[:, col_ix], dtype=np.float))
        labels.append('neg_{}'.format(header[col_ix]))
        colors.append(neg_color)
    
    # start plot
    fix, ax = plt.subplots()
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    # plot and show/save
    boxplots = ax.boxplot(data, patch_artist=True)
    for patch, color in zip(boxplots['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(labels, rotation=45)
    
    plt.show()



"""
    run_ratk
    ---------------------------------------------------------------------------
    energy files must be in the format: '<model_name>_<dataset>_<method>_energy.csv'
    
    base_dir (str): base directory for the energy csv files
    model_name (str): which model (dogwalking, pingpong, etc.)
    dataset (str): which data set (postest, fullneg, hardneg, etc.)
    method (str): which energy generation method (pgm, geo,brute, etc.)
    test_fnames (str, optional): the subset of files to use in the pos dataset
    ratk_zoom (int, optional): if specified, ratk plots will be generated from [0, ratk_zoom] in addition to the full range
    
    OUTPUT:
    r@k csv file in base_dir (<model_name>_<dataset>_<method>_ratk.csv)
    r@k plots in base_dir/plots (<model_name>_<dataset>_<method>_ratk.png)
"""
def run_ratk(base_dir, model_name, energy_method_name, pos_set_name, neg_set_names, test_fnames='', ratk_zoom=None):
    import os
    
    pos_csv = '{}_{}_{}_energy.csv'.format(model_name, pos_set_name, energy_method_name)
    pos_energy_fname = os.path.join(base_dir, pos_csv)
    
    for neg_set_name in neg_set_names:
        neg_energy = '{}_{}_{}_energy.csv'.format(model_name, neg_set_name, energy_method_name)
        neg_energy_fname = os.path.join(base_dir, neg_energy)
        ratk = r_at_k(pos_energy_fname, neg_energy_fname, test_fnames)
        
        neg_ratk = '{}_{}_{}_ratk.csv'.format(model_name, neg_set_name, energy_method_name)
        neg_ratk_fname = os.path.join(base_dir, neg_ratk)
        np.savetxt(neg_ratk_fname, ratk, fmt='%d, %03f')
        
        plot_dir = os.path.join(base_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        if ratk_zoom is None or len(ratk) <= ratk_zoom:
            neg_plot = '{}_{}_{}_ratk.png'.format(model_name, neg_set_name, energy_method_name)
            neg_plot_fname = os.path.join(base_dir, 'plots', neg_plot)
            r_at_k_plot(ratk[:,1], filename=neg_plot_fname)
        else:
            neg_plot = '{}_{}_{}_ratk_full.png'.format(model_name, neg_set_name, energy_method_name)
            neg_plot_fname = os.path.join(base_dir, 'plots', neg_plot)
            r_at_k_plot(ratk[:,1], filename=neg_plot_fname)
            
            neg_plot = '{}_{}_{}_ratk_100.png'.format(model_name, neg_set_name, energy_method_name)
            neg_plot_fname = os.path.join(base_dir, 'plots', neg_plot)
            r_at_k_plot(ratk[:,1], filename=neg_plot_fname, x_limit=100)



def personwearingglasses_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/person_wearing_glasses', 'pwg', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/personwearingglasses_fnames_test.txt', ratk_zoom)

def leadinghorse_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/leadinghorse', 'lh', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/leadinghorse_fnames_test.txt', ratk_zoom)

def pingpong_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/pingpong', 'pingpong', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/pingpong_fnames_test.txt', ratk_zoom)

def handshake_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/handshake', 'handshake', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/handshake_fnames_test.txt', ratk_zoom)

def dogwalking_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/dog_walking', 'dw_cycle', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/dogwalkingtest_fnames_test.txt', ratk_zoom)

def stanford_dw_ratk(energy_method, ratk_zoom=None):
    run_ratk('/home/econser/research/irsg_psu_pdx/output/full_runs/stanford_dog_walking', 'stanford_dw_cycle', energy_method, 'postest', ['fullneg', 'hardneg'], '/home/econser/research/irsg_psu_pdx/data/stanford_fnames_test.txt', ratk_zoom)





def r_at_k(pos_energy_file, neg_energy_file, pos_dataset_file='', render_plot=False):
    
    
    # split the positive filename, energy into 2 lists
    f_pos_csv = open(pos_energy_file, 'rb')
    
    pos_image_list = []
    pos_energy_list = []
    
    for row in f_pos_csv.readlines():
        items = row.split(',')
        pos_image_list.append(items[0].strip().split('.')[0])
        pos_energy_list.append(items[1].strip())
    pos_image_list = pos_image_list[1:]
    pos_energy_list = pos_energy_list[1:]
    
    keep_ixs = np.arange(len(pos_image_list))
    if len(pos_dataset_file) != 0:
        f_dataset = open(pos_dataset_file, 'rb')
        dataset = f_dataset.readlines()
        dataset = [item.split('.')[0] for item in dataset]
        
        keep_ixs = []
        for ix, pos_img in enumerate(pos_image_list):
            if pos_img in dataset:
                keep_ixs.append(ix)
    
    pos_energies = np.array(pos_energy_list, dtype=np.float)[keep_ixs]
    pos_energies.sort()
    
    neg_energies = np.genfromtxt(neg_energy_file, skip_header=True)
    neg_energies = neg_energies[:,1]
    neg_energies.sort()
    n_negatives = len(neg_energies)
    
    #import pdb; pdb.set_trace()
    ranks = np.searchsorted(neg_energies, pos_energies)
    ranks = ranks + 1 # searchsorted returns a 0-based value
    
    recalls = get_recalls(ranks, n_negatives)
    avg_recall = np.average(recalls, axis=0)
    if render_plot:
        r_at_k_plot(avg_recall)
    
    k = np.arange(n_negatives)+1
    k = k[:, np.newaxis]
    ratk_out = np.hstack((k , avg_recall[:, np.newaxis]))
    return ratk_out



def r_at_k_single(energy_file, pos_dataset_fname, render_plot=False):
    f_pos_csv = open(pos_energy_file, 'rb')
    
    pos_image_list = []
    pos_energy_list = []
    
    for row in f_pos_csv.readlines():
        items = row.split(',')
        pos_image_list.append(items[0].strip().split('.')[0])
        pos_energy_list.append(items[1].strip())
    pos_image_list = pos_image_list[1:]
    pos_energy_list = pos_energy_list[1:]
    
    keep_ixs = np.arange(len(pos_image_list))
    if len(pos_dataset_file) != 0:
        f_dataset = open(pos_dataset_file, 'rb')
        dataset = f_dataset.readlines()
        dataset = [item.split('.')[0] for item in dataset]
        
        keep_ixs = []
        for ix, pos_img in enumerate(pos_image_list):
            if pos_img in dataset:
                keep_ixs.append(ix)
    
    pos_energies = np.array(pos_energy_list, dtype=np.float)[keep_ixs]
    pos_energies.sort()
    
    neg_energies = np.genfromtxt(neg_energy_file, skip_header=True)
    neg_energies = neg_energies[:,1]
    neg_energies.sort()
    n_negatives = len(neg_energies)
    
    #import pdb; pdb.set_trace()
    ranks = np.searchsorted(neg_energies, pos_energies)
    ranks = ranks + 1 # searchsorted returns a 0-based value
    
    recalls = get_recalls(ranks, n_negatives)
    avg_recall = np.average(recalls, axis=0)
    if render_plot:
        r_at_k_plot(avg_recall)
    
    k = np.arange(n_negatives)+1
    k = k[:, np.newaxis]
    ratk_out = np.hstack((k , avg_recall[:, np.newaxis]))
    return ratk_out



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
    
    plt.xlim(xmin=1)
    if x_limit is not None:
        plt.xlim(xmax=x_limit)
    plt.ylim([0, 1])
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    plt.close()



def r_at_k_plots(r_at_k_tups, filename=None, x_limit=None):
    import matplotlib.pyplot as plt
  
    plt.figure(1)
    plt.grid(True)
    
    for ratk in r_at_k_tups:
        name = ratk[0]
        data = ratk[1][:,1]
        plot_handle, = plt.plot(np.arange(len(data)), data, label=name)
  
    plt.xlabel("k")
    plt.ylabel("Recall at k")
  
    plt.xlim(xmin=1)
    if x_limit is not None:
        plt.xlim(xmax=x_limit)
    plt.ylim([0, 1])
    
    plt.legend(loc='lower right')
    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    
    plt.close()



def r_at_k_table(k_vals, row_tups):
    max_label_len = 0
    for row in row_tups:
        if len(row[0]) > max_label_len:
            max_label_len = len(row[0])
    
    print('batch', end='')
    for val in k_vals:
        print(', {}'.format(val), end='')
    print('')
    
    for row in row_tups:
        print('{}'.format(row[0]), end='')
        for val in k_vals:
            print(', {:0.2f}'.format(row[1][val-1, 1]), end='')
        print('')



#-------------------------------------------------------------------------------
# VIZ PLOT SECTION
#
def viz_top_boxes(energy_csv, image_dir, bbox_csv_fmt, object_class_file, output_fmt):
    import os
    import cv2
    import csv
    import matplotlib.pyplot as plt
    
    # get fname,energy pairs
    in_files = []
    with open(energy_csv, 'rb') as f:
        csv_reader = csv.reader(f)
        for row_ix, row in enumerate(csv_reader):
            # ignore header
            if row_ix == 0:
                continue
            
            # trim off extension
            fname = row[0].split('.')[0]
            energy = float(row[1])
            in_files.append((fname, energy))
    
    # get object classes
    class_list = []
    with open(object_class_file, 'rb') as f:
        class_list = f.readlines()
        class_list = [cls.rstrip('\n') for cls in class_list]
    
    # read in the bbox csv files
    bbox_data = {}
    for cls in class_list:
        box_fname = bbox_csv_fmt.format(cls)
        with open(box_fname, 'rb') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                fname = row[0]
                fname = fname.split('.')[0]
                p = float(row[1])
                x = int(row[2])
                y = int(row[3])
                w = int(row[4])
                h = int(row[5])
                
                if fname not in bbox_data:
                    bbox_data[fname] = {}
                if cls not in bbox_data[fname]:
                    bbox_data[fname][cls] = []
                
                bbox_tup = (x, y, w, h, p)
                bbox_data[fname][cls].append(bbox_tup)
    
    # create output dir, if necessary
    outpath_split = output_fmt.split('/')
    outpath = os.path.join('', *outpath_split[:-1])
    outpath += '/'
    outpath = '/' + outpath
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # viz each file
    for row in in_files:
        # gen out fname
        energy = row[1]
        img_name = row[0]
        out_fname = output_fmt.format(energy, img_name)
        
        # prep plot
        image_filename = os.path.join(image_dir, img_name + '.jpg')
        im = cv2.imread(image_filename)
        im = im[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im, aspect='equal')
        
        for cls in class_list:
            bbox_tups = bbox_data[img_name][cls]
            for tup in bbox_tups:
                x = tup[0]
                y = tup[1]
                w = tup[2]
                h = tup[3]
                score = tup[4]
                
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3.5)
                ax.add_patch(rect)
                
                ax.text(x, y-2, '{:s} {:.3f}'.format(cls, score), bbox=dict(facecolor='blue', alpha=0.5), fontsize=14, color='white')
            
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
        plt.savefig(out_fname)
        plt.close()

def viz_pingpong(dataset, energy_method):
    import os
    base_dir = '/home/econser/research/irsg_psu_pdx/'
    image_dirs = {
        'postest': os.path.join(base_dir, 'data/PingPong'),
        'fullneg': os.path.join(base_dir, 'images/Negatives'),
        'hardneg': os.path.join(base_dir, 'images/PingPongHardNeg')
    }
    energy_dir = os.path.join(base_dir, 'output/full_runs/pingpong/')
    energy_csv = energy_dir + 'pingpong_{}_{}_energy.csv'.format(dataset, energy_method)
    image_dir  = image_dirs[dataset]
    bbox_csv_format = energy_dir + 'pingpong_{}_{}_{{}}_bboxes.csv'.format(dataset, energy_method)
    object_class_file = os.path.join(base_dir, 'data/pingpong_classes.txt')
    output_format = energy_dir + 'viz_{}_{}/{{:06.3f}}_{{}}.jpg'.format(dataset, energy_method)
    
    viz_top_boxes(energy_csv, image_dir, bbox_csv_format, object_class_file, output_format)

def viz_handshake(dataset, energy_method):
    import os
    base_dir = '/home/econser/research/irsg_psu_pdx/'
    image_dirs = {
        'postest': os.path.join(base_dir, 'data/Handshake'),
        'fullneg': os.path.join(base_dir, 'images/Negatives'),
        'hardneg': os.path.join(base_dir, 'images/HandshakeHardNeg')
    }
    energy_dir = os.path.join(base_dir, 'output/full_runs/handshake/')
    energy_csv = energy_dir + 'handshake_{}_{}_energy.csv'.format(dataset, energy_method)
    image_dir  = image_dirs[dataset]
    bbox_csv_format = energy_dir + 'handshake_{}_{}_{{}}_bboxes.csv'.format(dataset, energy_method)
    object_class_file = os.path.join(base_dir, 'data/handshake_classes.txt')
    output_format = energy_dir + 'viz_{}_{}/{{:06.3f}}_{{}}.jpg'.format(dataset, energy_method)
    
    viz_top_boxes(energy_csv, image_dir, bbox_csv_format, object_class_file, output_format)

def viz_dogwalking(dataset, energy_method):
    import os
    base_dir = '/home/econser/research/irsg_psu_pdx/'
    image_dirs = {
        'postest': os.path.join(base_dir, 'data/dog_walking'),
        'fullneg': os.path.join(base_dir, 'images/Negatives'),
        'hardneg': os.path.join(base_dir, 'images/DogWalkingHardNeg')
    }
    energy_dir = os.path.join(base_dir, 'output/full_runs/dog_walking/')
    energy_csv = energy_dir + 'dw_cycle_{}_{}_energy.csv'.format(dataset, energy_method)
    image_dir  = image_dirs[dataset]
    bbox_csv_format = energy_dir + 'dw_cycle_{}_{}_{{}}_bboxes.csv'.format(dataset, energy_method)
    object_class_file = os.path.join(base_dir, 'data/dog_walking_classes.txt')
    output_format = energy_dir + 'viz_{}_{}/{{:06.3f}}_{{}}.jpg'.format(dataset, energy_method)
    
    viz_top_boxes(energy_csv, image_dir, bbox_csv_format, object_class_file, output_format)



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

p.draw_best_pregen_boxes('/home/econser/research/irsg_psu_pdx/run_results/lh_fullpos/', '/home/econser/research/irsg_psu_pdx/data/LeadingHorse/', '/home/econser/research/irsg_psu_pdx/output/viz_lh_fullpos/', 'viz_')
p.draw_best_pregen_boxes('/home/econser/research/irsg_psu_pdx/run_results/lh_hardneg/', '/home/econser/research/irsg_psu_pdx/data/LeadingHorseHardNegative/', '/home/econser/research/irsg_psu_pdx/output/viz_lh_hardneg/', 'viz_')
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
"""
def sample_rel_model_(image_filename, gmms, relation_name, sub_class_name, obj_class_name, box_gen_fn, n_samples=100, input_dir='/home/econser/School/research/data/dog_walking/', output_dir='/home/econser/School/research/', output_filename='', do_title=True):
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
    
    if do_title:
        plt.title('{} {} {}'.format(sub_class_name, relation_name, obj_class_name))
    
    if len(output_filename) > 0:
        plt.savefig(filename+'{}/{}_{}_{}.png'.format(output_dir, object_name, relation_name, subject_name), dpi=175)
    else:
        plt.show()



#===============================================================================
"""
basedir = '/home/econser/School/research/output/full_runs_l1/'
dw_fmt = 'dog_walking/dw_cycle_{}_pgm_rel_{}.csv'
dw_datasets = ['postest', 'fullneg', 'hardneg', 'origneg']

import plot_utils as pu
pu.rel_hist_plots('attached_to', dw_fmt, basedir, dw_datasets, dw_datasets)
"""
def rel_hist_plots(rel_str, file_fmt, basedir, datasets, labels, bins=20, normed=1, log=True):
    import os.path
    file_and_label = []
    for ix, ds in enumerate(datasets):
        filename = file_fmt.format(ds, rel_str)
        fq_filename = os.path.join(basedir, filename)
        file_and_label.append((fq_filename, labels[ix]))
    rel_hist(file_and_label, rel_str, bins, normed, log)
    
def rel_hist(file_and_id_pairs, title='', bins=20, normed=1, log=True):
    import matplotlib.pyplot as plt
    data = []
    labels = []
    for tup in file_and_id_pairs:
        a = np.genfromtxt(tup[0], delimiter=',')
        data.append(a[1:,1])
        labels.append(tup[1])
    x = plt.hist(data, bins=bins, normed=normed, log=log)
    l = plt.legend(labels)
    ax = plt.xlim([0.0, 1.0])
    if len(title) > 0:
        plt.title(title)
    plt.show()



#===============================================================================
# OLD R@K CALL - del after confident in new r@k calls

def run_all_ratk():
    csv_dir = '/home/econser/research/irsg_psu_pdx/output/full_runs/'
    data_dir = '/home/econser/research/irsg_psu_pdx/data/'
    od = '/home/econser/research/irsg_psu_pdx/output/full_runs/plots/'
    
    pp_set = {}
    subdir = 'pingpong_maxrel/'
    #pp_set['max_postest'] = csv_dir + subdir + 'pingpong_postest_maxrel_energy.csv'
    subdir = 'pingpong/'
    pp_set['pgm_postest'] = csv_dir + subdir + 'pingpong_postest_pgm_energy.csv'
    pp_set['pgm_fullneg'] = csv_dir + subdir + 'pingpong_fullneg_pgm_energy.csv'
    pp_set['pgm_hardneg'] = csv_dir + subdir + 'pingpong_hardneg_pgm_energy.csv'
    pp_set['geo_postest'] = csv_dir + subdir + 'pingpong_postest_geo_energy.csv'
    pp_set['geo_fullneg'] = csv_dir + subdir + 'pingpong_fullneg_geo_energy.csv'
    pp_set['geo_hardneg'] = csv_dir + subdir + 'pingpong_hardneg_geo_energy.csv'
    
    dw_set = {}
    subdir = 'dog_walking_maxrel/'
    #dw_set['max_postest'] = csv_dir + subdir + 'dw_cycle_postest_maxrel_energy.csv'
    subdir = 'dog_walking/'
    dw_set['pgm_postest'] = csv_dir + subdir + 'dw_cycle_postest_pgm_energy.csv'
    dw_set['pgm_fullneg'] = csv_dir + subdir + 'dw_cycle_fullneg_pgm_energy.csv'
    dw_set['pgm_hardneg'] = csv_dir + subdir + 'dw_cycle_hardneg_pgm_energy.csv'
    dw_set['geo_postest'] = csv_dir + subdir + 'dw_cycle_postest_geo_energy.csv'
    dw_set['geo_fullneg'] = csv_dir + subdir + 'dw_cycle_fullneg_geo_energy.csv'
    dw_set['geo_hardneg'] = csv_dir + subdir + 'dw_cycle_hardneg_geo_energy.csv'
    dw_set['geo_origneg'] = csv_dir + subdir + 'dw_cycle_origneg_geo_energy.csv'
    
    st_set = {}
    subdir = 'stanford_dog_walking/'
    st_set['pgm_postest'] = csv_dir + subdir + 'stanford_dw_cycle_postest_pgm_energy.csv'
    st_set['pgm_fullneg'] = csv_dir + subdir + 'stanford_dw_cycle_fullneg_pgm_energy.csv'
    st_set['pgm_hardneg'] = csv_dir + subdir + 'stanford_dw_cycle_hardneg_pgm_energy.csv'
    st_set['geo_postest'] = csv_dir + subdir + 'stanford_dw_cycle_postest_geo_energy.csv'
    st_set['geo_fullneg'] = csv_dir + subdir + 'stanford_dw_cycle_fullneg_geo_energy.csv'
    st_set['geo_hardneg'] = csv_dir + subdir + 'stanford_dw_cycle_hardneg_geo_energy.csv'
    st_set['geo_origneg'] = csv_dir + subdir + 'stanford_dw_cycle_origneg_geo_energy.csv'
    
    hs_set = {}
    subdir = 'handshake_maxrel/'
    #hs_set['max_postest'] = csv_dir + subdir + 'handshake_postest_maxrel_energy.csv'
    subdir = 'handshake/'
    hs_set['pgm_postest'] = csv_dir + subdir + 'handshake_postest_pgm_energy.csv'
    hs_set['pgm_fullneg'] = csv_dir + subdir + 'handshake_fullneg_pgm_energy.csv'
    hs_set['pgm_hardneg'] = csv_dir + subdir + 'handshake_hardneg_pgm_energy.csv'
    hs_set['geo_postest'] = csv_dir + subdir + 'handshake_postest_geo_energy.csv'
    hs_set['geo_fullneg'] = csv_dir + subdir + 'handshake_fullneg_geo_energy.csv'
    hs_set['geo_hardneg'] = csv_dir + subdir + 'handshake_hardneg_geo_energy.csv'
    #-------------------------------------------------------------------------------
    pp_testset = data_dir + 'pingpong_fnames_test.txt'
    dw_testset = data_dir + 'dogwalkingtest_fnames_test.txt'
    hs_testset = data_dir + 'handshake_fnames_test.txt'
    st_testset = data_dir + 'stanford_fnames_test.txt'
    ratk = {}
    ratk['pp_pgm_fullneg'] = r_at_k(pp_set['pgm_postest'], pp_set['pgm_fullneg'], pp_testset)
    ratk['pp_pgm_hardneg'] = r_at_k(pp_set['pgm_postest'], pp_set['pgm_hardneg'], pp_testset)
    ratk['dw_pgm_fullneg'] = r_at_k(dw_set['pgm_postest'], dw_set['pgm_fullneg'], dw_testset)
    ratk['dw_pgm_hardneg'] = r_at_k(dw_set['pgm_postest'], dw_set['pgm_hardneg'], dw_testset)
    ratk['st_pgm_fullneg'] = r_at_k(st_set['pgm_postest'], st_set['pgm_fullneg'], st_testset)
    ratk['st_pgm_hardneg'] = r_at_k(st_set['pgm_postest'], st_set['pgm_hardneg'], st_testset)
    ratk['hs_pgm_fullneg'] = r_at_k(hs_set['pgm_postest'], hs_set['pgm_fullneg'], hs_testset)
    ratk['hs_pgm_hardneg'] = r_at_k(hs_set['pgm_postest'], hs_set['pgm_hardneg'], hs_testset)
    
    ratk['pp_geo_fullneg'] = r_at_k(pp_set['geo_postest'], pp_set['geo_fullneg'], pp_testset)
    ratk['pp_geo_hardneg'] = r_at_k(pp_set['geo_postest'], pp_set['geo_hardneg'], pp_testset)
    ratk['dw_geo_fullneg'] = r_at_k(dw_set['geo_postest'], dw_set['geo_fullneg'], dw_testset)
    ratk['dw_geo_hardneg'] = r_at_k(dw_set['geo_postest'], dw_set['geo_hardneg'], dw_testset)
    ratk['st_geo_fullneg'] = r_at_k(st_set['geo_postest'], st_set['geo_fullneg'], st_testset)
    ratk['st_geo_hardneg'] = r_at_k(st_set['geo_postest'], st_set['geo_hardneg'], st_testset)
    ratk['hs_geo_fullneg'] = r_at_k(hs_set['geo_postest'], hs_set['geo_fullneg'], hs_testset)
    ratk['hs_geo_hardneg'] = r_at_k(hs_set['geo_postest'], hs_set['geo_hardneg'], hs_testset)
    
    #ratk['pp_max_fullneg'] = r_at_k(pp_set['max_postest'], pp_set['pgm_fullneg'], pp_testset)
    #ratk['pp_max_hardneg'] = r_at_k(pp_set['max_postest'], pp_set['pgm_hardneg'], pp_testset)
    #ratk['dw_max_fullneg'] = r_at_k(dw_set['max_postest'], dw_set['pgm_fullneg'], dw_testset)
    #ratk['dw_max_hardneg'] = r_at_k(dw_set['max_postest'], dw_set['pgm_hardneg'], dw_testset)
    #ratk['hs_max_fullneg'] = r_at_k(hs_set['max_postest'], hs_set['pgm_fullneg'], hs_testset)
    #ratk['hs_max_hardneg'] = r_at_k(hs_set['max_postest'], hs_set['pgm_hardneg'], hs_testset)
    #ratk['st_max_fullneg'] = r_at_k(st_set['max_postest'], st_set['pgm_fullneg'], st_testset)
    #ratk['st_max_hardneg'] = r_at_k(st_set['max_postest'], st_set['pgm_hardneg'], st_testset)
    #-------------------------------------------------------------------------------
    np.savetxt(csv_dir+'pingpong/'            +'pingpong_fullneg_pgm_ratk.csv', ratk['pp_pgm_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'pingpong/'            +'pingpong_hardneg_pgm_ratk.csv', ratk['pp_pgm_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'dog_walking/'         +'dw_cycle_fullneg_pgm_ratk.csv', ratk['dw_pgm_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'dog_walking/'         +'dw_cycle_hardneg_pgm_ratk.csv', ratk['dw_pgm_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'stanford_dog_walking/'+'stanford_dw_cycle_fullneg_pgm_ratk.csv', ratk['st_pgm_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'stanford_dog_walking/'+'stanford_dw_cycle_hardneg_pgm_ratk.csv', ratk['st_pgm_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'handshake/'           +'handshake_fullneg_pgm_ratk.csv', ratk['hs_pgm_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'handshake/'           +'handshake_hardneg_pgm_ratk.csv', ratk['hs_pgm_hardneg'], fmt='%d, %03f')
    
    np.savetxt(csv_dir+'pingpong/'            +'pingpong_fullneg_geo_ratk.csv', ratk['pp_geo_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'pingpong/'            +'pingpong_hardneg_geo_ratk.csv', ratk['pp_geo_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'dog_walking/'         +'dw_cycle_fullneg_geo_ratk.csv', ratk['dw_geo_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'dog_walking/'         +'dw_cycle_hardneg_geo_ratk.csv', ratk['dw_geo_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'stanford_dog_walking/'+'stanford_dw_cycle_fullneg_geo_ratk.csv', ratk['st_geo_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'stanford_dog_walking/'+'stanford_dw_cycle_hardneg_geo_ratk.csv', ratk['st_geo_hardneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'handshake/'           +'handshake_fullneg_geo_ratk.csv', ratk['hs_geo_fullneg'], fmt='%d, %03f')
    np.savetxt(csv_dir+'handshake/'           +'handshake_hardneg_geo_ratk.csv', ratk['hs_geo_hardneg'], fmt='%d, %03f')
    
    #np.savetxt(csv_dir+'pingpong_maxrel/'     +'pingpong_fullneg_max_ratk.csv', ratk['pp_max_fullneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'pingpong_maxrel/'     +'pingpong_hardneg_max_ratk.csv', ratk['pp_max_hardneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'dog_walking_maxrel/'  +'dw_cycle_fullneg_max_ratk.csv', ratk['dw_max_fullneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'dog_walking_maxrel/'  +'dw_cycle_hardneg_max_ratk.csv', ratk['dw_max_hardneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'handshake_maxrel/'    +'handshake_fullneg_max_ratk.csv', ratk['hs_max_fullneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'handshake_maxrel/'    +'handshake_hardneg_max_ratk.csv', ratk['hs_max_hardneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'stanford_dog_walking_maxrel/'+'stanford_dw_cycle_fullneg_max_ratk.csv', ratk['st_max_fullneg'], fmt='%d, %03f')
    #np.savetxt(csv_dir+'stanford_dog_walking_maxrel/'+'stanford_dw_cycle_hardneg_max_ratk.csv', ratk['st_max_hardneg'], fmt='%d, %03f')
    
    #-------------------------------------------------------------------------------
    tab = [
        ('pp_pgm_fullneg', ratk['pp_pgm_fullneg']),
        ('pp_geo_fullneg', ratk['pp_geo_fullneg']),#        ('pp_max_fullneg', ratk['pp_max_fullneg']),
        #
        ('pp_pgm_hardneg', ratk['pp_pgm_hardneg']),
        ('pp_geo_hardneg', ratk['pp_geo_hardneg']),#        ('pp_max_hardneg', ratk['pp_max_hardneg']),
        #
        ('hs_pgm_fullneg', ratk['hs_pgm_fullneg']),
        ('hs_geo_fullneg', ratk['hs_geo_fullneg']),#        ('hs_max_fullneg', ratk['hs_max_fullneg']),
        #
        ('hs_pgm_hardneg', ratk['hs_pgm_hardneg']),
        ('hs_geo_hardneg', ratk['hs_geo_hardneg']),#        ('hs_max_hardneg', ratk['hs_max_hardneg']),
        #
        ('dw_pgm_fullneg', ratk['dw_pgm_fullneg']),
        ('dw_geo_fullneg', ratk['dw_geo_fullneg']),#        ('dw_max_fullneg', ratk['dw_max_fullneg']),
        #
        ('dw_pgm_hardneg', ratk['dw_pgm_hardneg']),
        ('dw_geo_hardneg', ratk['dw_geo_hardneg']),#        ('dw_max_hardneg', ratk['dw_max_hardneg']),
        #
        ('st_pgm_fullneg', ratk['st_pgm_fullneg']),
        ('st_geo_fullneg', ratk['st_geo_fullneg']),
        #
        ('st_pgm_hardneg', ratk['st_pgm_hardneg']),
        ('st_geo_hardneg', ratk['st_geo_hardneg'])
    ]
    k = np.array((1,2,5,10,20,100))
    r_at_k_table(k, tab)
    #-------------------------------------------------------------------------------
    pp_pgm = [
        ('full negative', ratk['pp_pgm_fullneg']),
        ('hard negative', ratk['pp_pgm_hardneg'])
    ]
    dw_pgm = [
        ('full negative', ratk['dw_pgm_fullneg']),
        ('hard negative', ratk['dw_pgm_hardneg'])
    ]
    st_pgm = [
        ('full negative', ratk['st_pgm_fullneg']),
        ('hard negative', ratk['st_pgm_hardneg'])
    ]
    hs_pgm = [
        ('full negative', ratk['hs_pgm_fullneg']),
        ('hard negative', ratk['hs_pgm_hardneg'])
    ]
    
    pp_geo = [
        ('full negative', ratk['pp_geo_fullneg']),
        ('hard negative', ratk['pp_geo_hardneg'])
    ]
    dw_geo = [
        ('full negative', ratk['dw_geo_fullneg']),
        ('hard negative', ratk['dw_geo_hardneg'])
    ]
    st_geo = [
        ('full negative', ratk['st_geo_fullneg']),
        ('hard negative', ratk['st_geo_hardneg'])
    ]
    hs_geo = [
        ('full negative', ratk['hs_geo_fullneg']),
        ('hard negative', ratk['hs_geo_hardneg'])
    ]
    
    #pp_max = [
    #    ('full negative', ratk['pp_max_fullneg']),
    #    ('hard negative', ratk['pp_max_hardneg'])
    #]
    #dw_max = [
    #    ('full negative', ratk['dw_max_fullneg']),
    #    ('hard negative', ratk['dw_max_hardneg'])
    #]
    #st_max = [
    #    ('full negative', ratk['st_max_fullneg']),
    #    ('hard negative', ratk['st_max_hardneg'])
    #]
    #hs_max = [
    #    ('full negative', ratk['hs_max_fullneg']),
    #    ('hard negative', ratk['hs_max_hardneg'])
    #]
    
    #pp_hard_cmp = [
    #    ('geo mean hard negative', ratk['pp_geo_hardneg']),
    #    ('standard hard negative', ratk['pp_pgm_hardneg']),
    #    ('max hard negative', ratk['pp_max_hardneg'])
    #]
    #dw_hard_cmp = [
    #    ('geo mean hard negative', ratk['dw_geo_hardneg']),
    #    ('standard hard negative', ratk['dw_pgm_hardneg']),
    #    ('max hard negative', ratk['dw_max_hardneg'])
    #]
    #st_hard_cmp = [
    #    ('geo mean hard negative', ratk['st_geo_hardneg']),
    #    ('standard hard negative', ratk['st_pgm_hardneg']),
    #    ('max hard negative', ratk['st_max_hardneg'])
    #]
    #hs_hard_cmp = [
    #    ('geo mean hard negative', ratk['hs_geo_hardneg']),
    #    ('standard hard negative', ratk['hs_pgm_hardneg']),
    #    ('max hard negative', ratk['hs_max_hardneg'])
    #]
    
    #pp_full_cmp = [
    #    ('geo mean full negative', ratk['pp_geo_fullneg']),
    #    ('standard full negative', ratk['pp_pgm_fullneg']),
    #    ('max full negative', ratk['pp_max_fullneg'])
    #]
    #dw_full_cmp = [
    #    ('geo mean full negative', ratk['dw_geo_fullneg']),
    #    ('standard full negative', ratk['dw_pgm_fullneg']),
    #    ('max full negative', ratk['dw_max_fullneg'])
    #]
    #st_full_cmp = [
    #    ('geo mean full negative', ratk['st_geo_fullneg']),
    #    ('standard full negative', ratk['st_pgm_fullneg']),
    #    ('max full negative', ratk['st_max_fullneg'])
    #]
    #hs_full_cmp = [
    #    ('geo mean full negative', ratk['hs_geo_fullneg']),
    #    ('standard full negative', ratk['hs_pgm_fullneg']),
    #    ('max full negative', ratk['hs_max_fullneg'])
    #]
    
    r_at_k_plots(pp_pgm, filename=od+'pingpong_pgm_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['pp_pgm_fullneg'][:,1], filename=od+'pingpong_pgm_fullneg_5k.png')
    r_at_k_plot(ratk['pp_pgm_fullneg'][:,1], filename=od+'pingpong_pgm_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['pp_pgm_hardneg'][:,1], filename=od+'pingpong_facgraph_hardneg.png')
    
    r_at_k_plots(hs_pgm, filename=od+'handshake_pgm_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['hs_pgm_fullneg'][:,1], filename=od+'handshake_pgm_fullneg_5k.png')
    r_at_k_plot(ratk['hs_pgm_fullneg'][:,1], filename=od+'handshake_pgm_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['hs_pgm_hardneg'][:,1], filename=od+'handshake_facgraph_hardneg.png')
    
    r_at_k_plots(dw_pgm, filename=od+'dogwalking_pgm_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['dw_pgm_fullneg'][:,1], filename=od+'dogwalking_pgm_fullneg_5k.png')
    r_at_k_plot(ratk['dw_pgm_fullneg'][:,1], filename=od+'dogwalking_pgm_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['dw_pgm_hardneg'][:,1], filename=od+'dogwalking_pgm_hardneg.png')
    
    r_at_k_plots(st_pgm, filename=od+'stanford_dogwalking_pgm_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['st_pgm_fullneg'][:,1], filename=od+'stanford_dogwalking_pgm_fullneg_5k.png')
    r_at_k_plot(ratk['st_pgm_fullneg'][:,1], filename=od+'stanford_dogwalking_pgm_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['st_pgm_hardneg'][:,1], filename=od+'stanford_dogwalking_pgm_hardneg.png')
    #----------------------------------------------
    r_at_k_plots(pp_geo, filename=od+'pingpong_geo_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['pp_geo_fullneg'][:,1], filename=od+'pingpong_geo_fullneg_5k.png')
    r_at_k_plot(ratk['pp_geo_fullneg'][:,1], filename=od+'pingpong_geo_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['pp_geo_hardneg'][:,1], filename=od+'pingpong_geo_hardneg.png')
    
    r_at_k_plots(hs_geo, filename=od+'handshake_geo_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['pp_geo_fullneg'][:,1], filename=od+'pingpong_geo_fullneg_5k.png')
    r_at_k_plot(ratk['pp_geo_fullneg'][:,1], filename=od+'pingpong_geo_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['pp_geo_hardneg'][:,1], filename=od+'pingpong_geo_hardneg.png')
    
    r_at_k_plots(dw_geo, filename=od+'dogwalking_geo_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['dw_geo_fullneg'][:,1], filename=od+'dogwalking_geo_fullneg_5k.png')
    r_at_k_plot(ratk['dw_geo_fullneg'][:,1], filename=od+'dogwalking_geo_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['dw_geo_hardneg'][:,1], filename=od+'dogwalking_geo_hardneg.png')
    
    r_at_k_plots(st_geo, filename=od+'stanford_dogwalking_geo_bothneg.png', x_limit=100)
    r_at_k_plot(ratk['st_geo_fullneg'][:,1], filename=od+'stanford_dogwalking_geo_fullneg_5k.png')
    r_at_k_plot(ratk['st_geo_fullneg'][:,1], filename=od+'stanford_dogwalking_geo_fullneg_100.png', x_limit=100)
    r_at_k_plot(ratk['st_geo_hardneg'][:,1], filename=od+'stanford_dogwalking_geo_hardneg.png')
#-------------------------------------------------------------------------------
    #r_at_k_plots(pp_max, filename=od+'pingpong_max_bothneg.png', x_limit=100)
    #r_at_k_plot(ratk['pp_max_fullneg'][:,1], filename=od+'pingpong_max_fullneg_5k.png')
    #r_at_k_plot(ratk['pp_max_fullneg'][:,1], filename=od+'pingpong_max_fullneg_100.png', x_limit=100)
    #r_at_k_plot(ratk['pp_max_hardneg'][:,1], filename=od+'pingpong_max_hardneg.png')
    
    #r_at_k_plots(hs_max, filename=od+'handshake_max_bothneg.png', x_limit=100)
    #r_at_k_plot(ratk['pp_max_fullneg'][:,1], filename=od+'pingpong_max_fullneg_5k.png')
    #r_at_k_plot(ratk['pp_max_fullneg'][:,1], filename=od+'pingpong_max_fullneg_100.png', x_limit=100)
    #r_at_k_plot(ratk['pp_max_hardneg'][:,1], filename=od+'pingpong_max_hardneg.png')
    
    #r_at_k_plots(dw_max, filename=od+'dogwalking_max_bothneg.png', x_limit=100)
    #r_at_k_plot(ratk['dw_max_fullneg'][:,1], filename=od+'dogwalking_max_fullneg_5k.png')
    #r_at_k_plot(ratk['dw_max_fullneg'][:,1], filename=od+'dogwalking_max_fullneg_100.png', x_limit=100)
    #r_at_k_plot(ratk['dw_max_hardneg'][:,1], filename=od+'dogwalking_max_hardneg.png')
    
    #r_at_k_plots(st_max, filename=od+'stanford_dogwalking_max_bothneg.png', x_limit=100)
    #r_at_k_plot(ratk['st_max_fullneg'][:,1], filename=od+'stanford_dogwalking_max_fullneg_5k.png')
    #r_at_k_plot(ratk['st_max_fullneg'][:,1], filename=od+'stanford_dogwalking_max_fullneg_100.png', x_limit=100)
    #r_at_k_plot(ratk['st_max_hardneg'][:,1], filename=od+'stanford_dogwalking_max_hardneg.png')
#-------------------------------------------------------------------------------
    #r_at_k_plots(pp_hard_cmp, filename=od+'pingpong_cmp_hard.png', x_limit=100)
    #r_at_k_plots(hs_hard_cmp, filename=od+'handshake_cmp_hard.png', x_limit=100)
    #r_at_k_plots(dw_hard_cmp, filename=od+'dogwalking_cmp_hard.png', x_limit=100)
    #r_at_k_plots(st_hard_cmp, filename=od+'stanford_dogwalking_cmp_hard.png', x_limit=100)
    #r_at_k_plots(pp_full_cmp, filename=od+'pingpong_cmp_full.png', x_limit=100)
    #r_at_k_plots(hs_full_cmp, filename=od+'handshake_cmp_full.png', x_limit=100)
    #r_at_k_plots(dw_full_cmp, filename=od+'dogwalking_cmp_full.png', x_limit=100)
    #r_at_k_plots(st_full_cmp, filename=od+'stanford_dogwalking_cmp_full.png', x_limit=100)
