#import _init_paths
import matplotlib; matplotlib.use('agg') #when running remotely
import sys
sys.path.append('/home/econser/School/Thesis/external/py-faster-rcnn/lib')

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'person', 'sunglasses', 'bench', 'helmet', 'horse', 'skateboard', 'shoes', 'bottle', 'table', 'building', 'bicycle', 'post')

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

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

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    #plt.draw()
    fname = '/home/econser/research/docs/mvg_{}.jpg'.format(class_name)
    print('saving {}'.format(fname))
    plt.savefig(fname)



if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    caffe.set_mode_gpu()
    caffe.set_device(0)
    cfg.GPU_ID = 0

    model_file = '/home/econser/research/irsg_psu_pdx/models/model_definitions/mini_vg/faster_rcnn_end2end/test.prototxt'
    weights = '/home/econser/research/irsg_psu_pdx/models/model_weights/minivg_50000.caffemodel'
    net = caffe.Net(model_file, weights, caffe.TEST)

    img_fname = '/home/econser/research/irsg_psu_pdx/data/sg_dataset/sg_train_images/1348163097_6d49676ae8_o.jpg'
    image = cv2.imread(img_fname)
    scores, boxes = im_detect(net, image)

    CONF_THRESH = 0.5
    NMS_THRESH = 0.5
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(image, cls, dets, thresh=CONF_THRESH)

    #plt.show()
