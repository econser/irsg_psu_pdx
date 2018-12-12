#import _init_paths
#import matplotlib; matplotlib.use('agg') #when running remotely
import sys
sys.path.append('/home/econser/usr/py-faster-rcnn/lib')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
#from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import json

#CLASSES = ('__background__', 'person', 'sunglasses', 'bench', 'helmet', 'horse', 'skateboard', 'shoes', 'bottle', 'table', 'building', 'bicycle', 'post')
CLASSES = ['__background__', 'person', 'beard', 'bench', 'horse', 'skateboard', 'helmet', 'sunglasses', 'pillow', 'couch'] # 10 classes
PERSON_CLASSES = ['woman', 'man', 'boy', 'child', 'girl', 'lady', 'driver', 'guy', 'kid']

def get_annotations(anno_path, dataset_name, classes, person_classes=[]):
    f = open(os.path.join(anno_path, 'sg_{}_annotations.json'.format(dataset_name)), 'rb')
    j = json.load(f)
    f.close()
    
    train_dict = {}
    for anno in j:
        key = anno['filename']
        train_dict[key] = []

        img_size = (anno['width'], anno['height'])
        obj_bboxes = []
        
        for obj in anno['objects']:
            valid = False
            for name in obj['names']:
                if name in classes:
                    cls_name = name
                    valid = True
                    
                if name in person_classes:
                    cls_name = 'person'
                    valid = True
                    break
            
            if not valid:
                continue
            
            x = obj['bbox']['x']
            y = obj['bbox']['y']
            w = obj['bbox']['w']
            h = obj['bbox']['h']
            bbox = [x, y, w+x, h+y]
            #bbox = [x, y, w, h]
            
            obj_bboxes.append((cls_name, bbox))
        train_dict[key] = (img_size, obj_bboxes)
    return train_dict



def vis_detections(im, class_name, dets, anno, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return False

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    # draw any GT bboxes
    for obj in anno[1]:
        if obj[0] == class_name:
            bbox = obj[1]
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - x
            h = bbox[3] - y
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=4.0)
            ax.add_patch(rect)
    
    ax.set_title(('{} detections with ''p({} | box) >= {:.2f}').format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    return True
    #plt.draw()



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    model_file = '/home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/test.prototxt'
    weights = '/home/econser/research/irsg_psu_pdx/models/model_weights/minivg_1M.caffemodel'
    net = caffe.Net(model_file, weights, caffe.TEST)

    out_dir = '/home/econser/research/irsg_psu_pdx/output/minivg_viz/'
    img_dir = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_test_images'
    img_fnames = os.listdir(img_dir)
    n_images = len(img_fnames)

    for cls in CLASSES[1:]:
        out_cls_dir = os.path.join(out_dir, cls)
        if not os.path.exists(out_cls_dir):
            os.makedirs(out_cls_dir)

    anno_path = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/'
    annos = get_annotations(anno_path, 'test', CLASSES, PERSON_CLASSES)
    for i, img_fname in enumerate(img_fnames):
        if img_fname.endswith('.gif'):
            continue
        
        print('generating image {} ({:0.2f}%)'.format(img_fname, i/float(n_images) * 100.0))
        
        img_fqname = os.path.join(img_dir, img_fname)
        
        image = cv2.imread(img_fqname)
        scores, boxes = im_detect(net, image)
        
        CONF_THRESH = 0.25
        NMS_THRESH = 0.5
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            #print('{} : {}'.format(cls, dets[:,-1][:5]))
            do_save = vis_detections(image, cls, dets, annos[img_fname], thresh=CONF_THRESH)
            if do_save:
                out_cls_dir = os.path.join(out_dir, cls)
                out_fname = img_fname
                out_fqname = os.path.join(out_cls_dir, out_fname)
                print('   {}   {}'.format(cls, out_cls_dir))
                plt.savefig(out_fqname)
                plt.close('all')
