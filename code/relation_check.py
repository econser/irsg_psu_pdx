import numpy as np
import irsg_utils as iu

class RelationScores (object):
    def __init__(self, relation_name, subject_bbox, object_bbox, probability, pdf_value, gmm_vec):
        self.relation_name = relation_name
        self.subject_bbox = subject_bbox
        self.object_bbox = object_bbox
        self.probability = probability
        self.pdf_value = pdf_value
        self.gmm_vec = gmm_vec
    
    """
    verbosity
        1 = probability
        2 = 1 & pdf value
        3 = 2 & subject bbox, object_bbox
        4 = 3 & gmm vec
    """
    def csv_print(self, verbosity=1):
        if verbosity == 1:
            return '{:0.6f}'.format(self.probability)
        elif verbosity == 2:
            return '{:0.6f}, {:0.6f}'.format(self.probability, self.pdf_value)
        elif verbosity == 3:
            return '{:0.6f}, {:0.6f}, {}, {}, {}, {}, {}, {}, {}, {}'.format(self.probability, self.pdf_value, self.subject_bbox[0], self.subject_bbox[1], self.subject_bbox[2], self.subject_bbox[3], self.object_bbox[0], self.object_bbox[1], self.object_bbox[2], self.object_bbox[3])
        else:
            return '{:0.6f}, {:0.6f}, {}, {}, {}, {}, {}, {}, {}, {}, {:0.3f}, {:0.3f}, {:0.3f}, {:0.3f}'.format(self.probability, self.pdf_value, self.subject_bbox[0], self.subject_bbox[1], self.subject_bbox[2], self.subject_bbox[3], self.object_bbox[0], self.object_bbox[1], self.object_bbox[2], self.object_bbox[3], self.gmm_vec[0], self.gmm_vec[1], self.gmm_vec[2], self.gmm_vec[3])



def get_relationship_score(gmm_params, subject_bbox, object_bbox):
    bbox_pair = np.array((subject_bbox, object_bbox))
    bbox_pair = bbox_pair[np.newaxis, :, :]
    
    input_vec = iu.get_gmm_features(bbox_pair, in_format='xywh')
    pdf_score = iu.gmm_pdf(input_vec, gmm_params.gmm_weights, gmm_params.gmm_mu, gmm_params.gmm_sigma)
    prob_score = 1. / (1. + np.exp(-(gmm_params.platt_a * pdf_score + gmm_params.platt_b)))
    return input_vec, pdf_score, prob_score



#===============================================================================
def get_dog_walking_scores(box_fn, annotation_file, gmms):
    f = open(annotation_file, 'rb')
    anno_line = f.readline()
    f.close()
    
    obj_dict = box_fn(anno_line)
    
    dog_walker_bbox = obj_dict['dog_walker']
    leash_bbox = obj_dict['leash']
    dog_bbox = obj_dict['dog']
    
    ret_dict = {}
    
    # get 'holding' score
    relation_key = 'holding'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, dog_walker_bbox, leash_bbox)
    rs = RelationScores(relation_key, dog_walker_bbox, leash_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    # get 'attached_to' score
    relation_key = 'attached_to'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, leash_bbox, dog_bbox)
    rs = RelationScores(relation_key, leash_bbox, dog_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    # get dog walked_by dw score
    relation_key = 'walked_by'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, dog_bbox, dog_walker_bbox)
    rs = RelationScores(relation_key, dog_bbox, dog_walker_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    return ret_dict



def get_pingpong_scores(box_fn, annotation_file, gmms):
    f = open(annotation_file, 'rb')
    anno_line = f.readline()
    f.close()
    
    obj_dict = box_fn(anno_line)
    
    player1_bbox = obj_dict['player__1']
    player2_bbox = obj_dict['player__2']
    table_bbox = obj_dict['table']
    net_bbox = obj_dict['net']
    
    ret_dict = {}
    
    # get player at table scores
    relation_key = 'at'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, player1_bbox, table_bbox)
    rs1 = RelationScores(relation_key, player1_bbox, table_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs1)
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, player2_bbox, table_bbox)
    rs2 = RelationScores(relation_key, player2_bbox, table_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs2)
    
    # get net on table score
    relation_key = 'on'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, net_bbox, table_bbox)
    rs = RelationScores(relation_key, net_bbox, table_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    # get player playing_pingpong_with player score
    relation_key = 'playing_pingpong_with'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, player1_bbox, player2_bbox)
    rs = RelationScores(relation_key, player1_bbox, player2_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    return ret_dict



def get_handshake_scores(box_fn, annotation_file, gmms):
    f = open(annotation_file, 'rb')
    anno_line = f.readline()
    f.close()
    
    obj_dict = box_fn(anno_line)
    
    p1_bbox = obj_dict['person__1']
    p2_bbox = obj_dict['person__2']
    handshake_bbox = obj_dict['handshake']
    
    ret_dict = {}
    
    # get person1 extending handshake score
    relation_key = 'extending'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, p1_bbox, handshake_bbox)
    rs1 = RelationScores(relation_key, p1_bbox, handshake_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs1)
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, p2_bbox, handshake_bbox)
    rs2 = RelationScores(relation_key, p2_bbox, handshake_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs2)
    
    # get person1 handshaking person2 score
    relation_key = 'handshaking'
    
    ret_dict[relation_key] = []
    gmm_params = gmms[relation_key]
    
    input_vec, pdf, prob = get_relationship_score(gmm_params, p1_bbox, p2_bbox)
    rs = RelationScores(relation_key, p1_bbox, p2_bbox, prob[0], pdf[0], input_vec[0])
    ret_dict[relation_key].append(rs)
    
    return ret_dict



#===============================================================================
def parse_args():
    import argparse
    model_choices = ['dog_walking', 'handshake', 'pingpong']
    parser = argparse.ArgumentParser(description='ground truth relation calculation')
    
    parser.add_argument('--model', dest='model', help='which modelset to run', default='dog_walking', choices=model_choices)
    parser.add_argument('--gmm_file', dest='gmm_file', help='location of GMM pkl file', required=True)
    parser.add_argument('--anno_dir', dest='anno_dir', help='directory of annotations', required=True)
    parser.add_argument('--fileset', dest='fileset', help='(optional) set of files to process')
    parser.add_argument('--output_dir', dest='output_dir', help='location of output', required=True)
    
    args = parser.parse_args()
    return args



"""
    --model      : dog_walking | handshake | pingpong
    --gmm_file   : fully qualified gmm filename
    --anno_dir   : dir of .labl files
    --fileset    : set of files to process, all files run if empty
    --output_dir : output dir
    
DOG WALKING:
python relation_check.py --gmm_file '/home/econser/School/research/data/dw_gmms_l1.pkl' --anno_dir '/home/econser/School/research/data/dog_walking' --output_dir '/home/econser/School/research/output'
    
PINGPONG:
python relation_check.py --model pingpong --gmm_file '/home/econser/School/research/data/pingpong_gmms.pkl' --anno_dir '/home/econser/School/research/data/PingPong' --output_dir '/home/econser/School/research/output'
    
HANDSHAKE:
python relation_check.py --model handshake --gmm_file '/home/econser/School/research/data/handshake_gmms.pkl' --anno_dir '/home/econser/School/research/data/Handshake' --output_dir '/home/econser/School/research/output'
"""
if __name__ == '__main__':
    import irsg_utils as iutl
    import os.path
    import cPickle
    
    box_fn_map = {
        'dog_walking' : iu.get_dw_boxes,
        'handshake'   : iu.get_hs_bboxes,
        'pingpong'    : iu.get_pp_bboxes
    }
    
    score_fn_map = {
        'dog_walking' : get_dog_walking_scores,
        'handshake'   : get_handshake_scores,
        'pingpong'    : get_pingpong_scores
    }
    
    rel_map = {
        'dog_walking' : ['holding', 'attached_to', 'walked_by'],
        'handshake'   : ['extending', 'handshaking'],
        'pingpong'    : ['at', 'on', 'playing_pingpong_with']
    }
    
    args = parse_args()
    
    # parse args
    model_name = args.model
    bbox_fn = box_fn_map[model_name]
    score_fn = score_fn_map[model_name]
    
    fq_gmm_file = args.gmm_file
    
    fileset = None
    if args.fileset is not None:
        fq_fileset = os.path.join(args.anno_dir, args.fileset)
        f = open(fq_fileset)
        fileset = f.readlines()
        f.close()
        fileset = [fn.rstrip('\n') for fn in fileset]
        fileset = [fn.rstrip('\r') for fn in fileset]
    else:
        fileset = os.listdir(args.anno_dir)
        fileset = filter(lambda f: '.labl' in f, fileset)
    
    anno_files = []
    for filename in fileset:
        fq_anno_file = os.path.join(args.anno_dir, filename)
        anno_files.append(fq_anno_file)
    
    # open the gmms
    f = open(fq_gmm_file, 'rb')
    gmms = cPickle.load(f)
    f.close()
    
    # get the scores
    output_dict = {}
    rel_keys = rel_map[model_name]
    for key in rel_keys:
        output_dict[key] = []
    
    for anno_file in anno_files:
        print('running {}'.format(anno_file))
        score_dict = score_fn(bbox_fn, anno_file, gmms)
        for key in rel_keys:
            base_fname = os.path.basename(anno_file)
            rel_scores = score_dict[key]
            for score in rel_scores:
                output_dict[key].append((base_fname, score))
    
    # store output
    for key in rel_keys:
        outfile = os.path.join(args.output_dir, '{}_{}.csv'.format(model_name, key))
        f = open(outfile, 'wb')
        f.write('filename, probability, pdf, sub_x, sub_y, sub_w, sub_h, obj_x, obj_y, obj_w, obj_h, rel_x, rel_y, rel_w, rel_h\n')
        for line in output_dict[key]:
            fname = line[0]
            rc = line[1]
            f.write('{}, {}\r\n'.format(fname, rc.csv_print(verbosity=4)))
        f.close()
