import os.path
import randomcolor
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects



def draw_best_objects(image_filename, obj_box_pairs, title="", output_filename=""):
    rc = randomcolor.RandomColor()
    colorset = rc.generate(luminosity='bright', count=len(obj_box_pairs), format_='rgb')
    color_list = []
    for i in range(0, len(obj_box_pairs)):
        color_array = np.fromstring(colorset[i][4:-1], sep=",", dtype=np.int)
        color_array = color_array * (1. / 255.)
        color_list.append(color_array)
    
    draw_image_boxes(image_filename, obj_box_pairs, color_list, title, output_filename)



def draw_image_boxes(image_filename, obj_box_pairs, color_list, title="", output_filename="", verbose=False):
      #plt.switch_backend('Qt4Agg')
    
    if not os.path.isfile(image_filename):
        return
    
    img = Image.open(image_filename)
    img_array = np.array(img, dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(img_array)
    
    for ix, obj_and_box in enumerate(obj_box_pairs):
        box = obj_and_box[1]
        x = box[0]
        y = box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        box = patches.Rectangle((x,y),w,h, linewidth=4, edgecolor=color_list[ix], facecolor='none')
        ax.add_patch(box)
        txt = ax.text(x+5, y+5, obj_and_box[0], va='top', size=16, weight='bold', color='0.1')
        txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='w')])
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.tight_layout(pad=7.5)
    
    if len(output_filename) == 0:
        plt.show(bbox_inches='tight')
        plt.show()
    else:
        plt.rcParams.update({'font.size': 10})
        plt.savefig(filename, dpi=175)
    
    plt.clf()
    plt.close()
