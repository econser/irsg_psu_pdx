import sys
#sys.path.append('/home/econser/School/Thesis/external/caffe/python')
sys.path.append('/home/econser/School/Thesis/external/py-faster-rcnn/caffe-fast-rcnn/python')
sys.path.append('/home/econser/School/Thesis/external/py-faster-rcnn/lib')

def t():
    return go('/home/econser/School/research/images/people_in_kitchen.jpg')

def go(image_filename):
    import os
    import cv2
    import caffe
    import numpy as np
    from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
    
    # open the image
    image = cv2.imread(image_filename)
    
    # prep the image for net input
    blobs = {'data' : None, 'rois' : None}
    net_img = image.astype(np.float32, copy=True)
    net_img -= np.array([[[102.9801, 115.9465, 122.7717]]])
    
    im_shape = image.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    processed_ims = []
    im_scales = []
    
    target_size = 600
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1000:
        im_scale = float(1000) / float(im_size_max)
    resized_img = cv2.resize(net_img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scales.append(im_scale)
    processed_ims.append(resized_img)
    
    max_shape = np.array([resized_img.shape]).max(axis=0)
    num_images = 1
    blob = np.zeros((1, max_shape[0], max_shape[1], 3), dtype=np.float32)
    blob[0, 0:resized_img.shape[0], 0:resized_img.shape[1], :] = resized_img
    channel_swap = (0, 3, 1, 2)
    blobs['data'] = blob.transpose(channel_swap)
    
    dedup_scale_factor = 1. / 16.
    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]], dtype=np.float32)
    
    # create the net
    base_dir = '/home/econser/School/Thesis/external/py-faster-rcnn'
    proto_dir = 'models/pascal_voc/VGG16/faster_rcnn_alt_opt'
    model_dir = 'data/faster_rcnn_models'
    prototxt = os.path.join(base_dir, proto_dir, 'faster_rcnn_test.pt')
    caffemodel = os.path.join(base_dir, model_dir, 'VGG16_faster_rcnn_final.caffemodel')
    
    caffe.set_mode_cpu()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    
    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    import pdb; pdb.set_trace()
    blobs_out = net.forward(**forward_kwargs)
    
    rois = net.blobs['rois'].data.copy()
    boxes = rois[:, 1:5] / im_scales[0]
    scores = blobs_out['cls_prob'] # sofmax cal'd scores
    #scores = net.blobs['cls_score'].data # raw scores
    
    # Apply bounding-box regression deltas
    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, image.shape)
    
    return scores, pred_boxes
