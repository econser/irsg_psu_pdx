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
    model_spec = {}
    if version == 'dog_walking':
        model_spec = {
            'base_dir': '/home/econser/School/research/models/',
            'prototxt': 'model_definitions/dog_walking/faster_rcnn_end2end/test.prototxt',
            'caffemodel_1': 'model_weights/dog_walking_faster_rcnn_final.caffemodel',
            'caffemodel_2': 'model_weights/vgg_cnn_m_1024_faster_rcnn_iter_10000.caffemodel',
            'caffemodel': 'model_weights/vgg_cnn_m_1024_faster_rcnn_iter_50000.caffemodel',
            'im_names': ['/home/econser/School/research/images/psu_dw/PortlandSimpleDogWalking/dog-walking301.jpg'],
            'classes': ['__background__', 'dog', 'dog_walker', 'leash'],
            'boxes_per_class': [0,1,1,1]}
    elif version == 'pingpong':
        path_fmt = '/home/econser/School/research/data/PingPong/pingpong{}.jpg'
        im_names = []
        for i in range(301, 401):
            im_names.append(path_fmt.format(i))
        model_spec = {
            'base_dir': '/home/econser/School/research/models/',
            'prototxt': 'model_definitions/pingpong/faster_rcnn_end2end/test.prototxt',
            'caffemodel': 'model_weights/pingpong_frcn_50k.caffemodel',
            'im_names': im_names,
            'classes': ['__background__', 'player', 'net', 'table'],
            'boxes_per_class':[0,2,1,1]}
    
    return model_spec



def vis_detections(im, detections, output_filename):
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    
    for object_detection in detections:
        class_name = object_detection[0]
        bboxes = object_detection[1]
        boxes_to_draw = object_detection[2]
        for i in range(0, boxes_to_draw):
            bbox = bboxes[i, :4]
            score = bboxes[i, -1]
            
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
    
    #ax.set_title(('{} detections with 'p({} | box) >= {:.1f}').format(class_name, class_name, thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #plt.show()
    plt.savefig(output_filename)



def demo(net, image_name, classes, boxes_per_class, vizfile_dir, nms_threshold=0.3):
    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.8
    detections = []
    for cls_ind, cls in enumerate(classes[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_threshold)
        dets = dets[keep, :]
        
        detections.append((cls, dets, boxes_per_class[cls_ind]))
    
    img_filename = image_name.split('/')[-1]
    output_filename = os.path.join(vizfile_dir, 'viz_{}'.format(img_filename))
    vis_detections(im, detections, output_filename)

"""
dets[dets[:,4].argsort()][::-1][:5]
np.set_printoptions(suppress=True)
"""



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
    
    model_spec = get_config(model_cfg_name)
    prototxt = model_spec['base_dir'] + model_spec['prototxt']
    caffemodel = model_spec['base_dir'] + model_spec['caffemodel']
    classes = model_spec['classes']
    boxes_per_class = model_spec['boxes_per_class']
    vizfile_dir = '/home/econser/School/research/output/pingpong_viz/'
    
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
        demo(net, im_name, classes, boxes_per_class, vizfile_dir)
