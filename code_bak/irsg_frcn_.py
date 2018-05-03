"""
ii.gen_boxes_and_scores('/home/econser/School/research/images/', '/home/econser/School/research/frcn_test/', classes)
"""
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

def get_config(version):
    classes = []
    model_spec = {}
    
    if version == 'dog_walking':
        classes = ['__background__', 'dog', 'dog_walker', 'leash']
        model_spec = {
            'base_dir': '/home/econser/School/research/models/',
            'prototxt': 'model_definitions/dog_walking/faster_rcnn_end2end/test.prototxt',
            'caffemodel_1': 'model_weights/dog_walking_faster_rcnn_final.caffemodel',
            'caffemodel_2': 'model_weights/vgg_cnn_m_1024_faster_rcnn_iter_10000.caffemodel',
            'caffemodel': 'model_weights/vgg_cnn_m_1024_faster_rcnn_iter_50000.caffemodel',
            'im_names': ['/home/econser/School/research/images/psu_dw/PortlandSimpleDogWalking/dog-walking301.jpg']}
    elif version == 'pingpong':
        classes = ['__background__', 'player', 'net', 'table']
        model_spec = {
            'base_dir': '/home/econser/School/research/models/',
            'prototxt': 'model_definitions/pingpong/faster_rcnn_end2end/test.prototxt',
            'caffemodel': 'model_weights/pingpong_frcn_50k.caffemodel',
            'im_names': ['/home/econser/School/research/data/PingPong/pingpong301.jpg']}
    
    return classes, model_spec



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
    plt.draw()



def demo(net, image_name, classes):
    """Detect object classes in an image using pre-computed object proposals."""
    
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        #TODO: calculate IoU
        #image_name_prefix = image_name.split('.')[0]
        #np.savetxt('{}{}{}'.format(output_path, cls, image_name_prefix), dets, fmt='%d, %d, %d, %d, %0.3f')
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)



def make_net(prototxt, caffemodel):
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    
    if prototxt == '':
        prototxt = '/home/econser/School/research/models/model_definitions/dog_walking/faster_rcnn_end2end/test.prototxt'
    if caffemodel == '':
        caffemodel = '/home/econser/School/research/models/model_weights/dog_walking_faster_rcnn_final.caffemodel'
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    return net



"""
    Outputs csv files in one dir for each object class
    CSV format: x_coord, y_coord, width, height, score, IOU
    IOU = -1 for images with no GT annotation
"""
def gen_boxes_and_scores(image_basedir, output_basedir, classes, image_list=None, prototxt='', caffemodel='', nms_thresh=-1.):
    import glob
    
    net = make_net(prototxt, caffemodel)
    warmup(net)
    
    if image_list is None:
        image_list = glob.glob('{}*.jpg'.format(image_basedir))
    
    for image in image_list: #change image_name -> image
        im_file = os.path.join(cfg.DATA_DIR, 'demo', image)
        im = cv2.imread(im_file)
        
        scores, boxes = im_detect(net, im)
        
        for cls_ind, cls in enumerate(classes[1:]):
            cls_ind += 1 # skip background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            
            cls_scores = scores[:, cls_ind]
            cls_scores = cls_scores[:, np.newaxis]
            
            ious = np.ones_like(cls_scores) * -1.
            
            dets = np.hstack((cls_boxes, cls_scores))
            dets = np.hstack((dets, ious))
            
            if nms_thresh > 0. and nms_thresh < 1.:
                keep = nms(dets, nms_thresh)
                dets = dets[keep, :]
            
            image_prefix = image.split('.')[0].split('/')[-1]
            class_dir = '{}{}'.format(output_basedir, cls)
            out_file = '{}/{}.csv'.format(class_dir, image_prefix)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            np.savetxt(out_file, dets, fmt='%d, %d, %d, %d, %0.6f, %0.6f')
    
    return None



def warmup(net, num=2):
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(num):
        _, _= im_detect(net, im)



if __name__ == '__main__':
    import sys
    
    model_cfg_name = 'pingpong'
    #todo:  read config from args
    for arg in sys.argv[1:]:
        im_name = arg
    
    classes, model_spec = get_config(model_cfg_name)
    prototxt = model_spec['base_dir'] + model_spec['prototxt']
    caffemodel = model_spec['base_dir'] + model_spec['caffemodel']
    
    net = make_net(prototxt, caffemodel)
    print '\n\nLoaded network {:s}'.format(caffemodel.split('/')[-1])
    
    print 'Warming up network...'
    warmup(net)
    
    im_names = model_spec['im_names']
    if len(im_names) == 0:
        im_names = ['/home/econser/School/research/images/o1.jpg']
    
    for im_name in im_names:
        print '============================================================================'
        print 'Processing image {}'.format(im_name.split('/')[-1])
        demo(net, im_name, classes)
    
    plt.show()
