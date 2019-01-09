
import os
import sys
import json as j



'''
<MODEL_NAME>
   image_dir : './data/<MODEL_NAME>/'
   class_file : './data/<MODEL_NAME>_classes.txt'
   energy_output_dir : './output/full_runs/<MODEL_NAME>/'
<QUERY_STR>        :
<QUERY_FN>         :
<MODEL_SHORT_NAME> : 
<MODEL_NAME>       : 
<TRAIN_IMG_FNAMES> :
<RELATION_FNAME>   : minivg_gmms.pkl

-- need to add these: (base is ~/research/irsg_psu_pdx/)
<IMAGE_DIR> : ./images/psu_dw/PortlandSimpleDogWalking/
<RCNN_MODEL_DEF> : ./models/model_definitions/dog_walking/faster_rcnn_end2end/test.prototxt
<RCNN_MODEL_WEIGHTS> : ./models/model_weights/dog_walking_faster_rcnn_final.caffemodel
<CLASS_FILE> : ./data/dog_walking_classes.txt
<BBOX_CSV_OUTDIR> : ./run_results/dw_fullpos/


-- config filenames:
mvg_PersonHasBeard.json
mvg_PersonOnBench.json
mvg_PersonOnHorse.json
mvg_PersonOnSkateboard.json
mvg_PersonWearingHelmet.json
mvg_PersonWearingSunglasses.json
mvg_PillowOnCouch.json


-- generate training fnames
import os
fnames = os.listdir('/home/econser/research/irsg_psu_pdx/data/minivg_queries/PersonHasBeardTrain')
f = open('/home/econser/research/irsg_psu_pdx/data/minivg_queries/PersonHasBeardTrain_fnames.txt', 'wb')
for fname in anno_fnames:
   f.write('{}\n'.format(fname))
f.close()
'''
#===============================================================================
def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='config prep tool')
    parser.add_argument('--cfg', dest='json_cfg')
    args = parser.parse_args()
    
    json_fname = args.json_cfg
    f = open(json_fname, 'rb')
    import pdb; pdb.set_trace()
    json_cfg = j.load(f)
    
    return [json_cfg]



if __name__ == '__main__':
    cfg = get_cfg()
    json_cfg = cfg[0]

    # bail out if cfg is not complete
    import pdb; pdb.set_trace()
    if not 'MODEL_SHORT_NAME' in json_cfg:
        sys.exit('Missing MODEL_SHORT_NAME')

    # search and replace in template file
    template_dict = json_cfg['template_vals']
    
    snr_keys = template_dict.keys()
    srcdir = '/home/econser/research/irsg_psu_pdx/utils/query_prep'
    destdir = '/home/econser/research/irsg_psu_pdx/utils/query_prep/out'
    fname_tup = (os.path.join(srcdir, 'configs'), os.path.join(destdir, 'configs'), 'configs_nobrute_TEMPLATE.yml')
    
    # open the template file
    src_fname = os.path.join(fname_tup[0], fname_tup[2])
    with open(src_fname, 'rb') as f:
        fdata = f.read()

    # replace keys
    for snr_key in snr_keys:
        snr_val = template_dict[snr_key].encode('ascii', 'ignore')
        k = '<'+snr_key.encode('ascii', 'ignore')+'>'
        fdata = fdata.replace(k, snr_val)
    
    # create output file
    if not os.path.exists(fname_tup[1]):
        os.makedirs(fname_tup[1])
    out_fname = fname_tup[2].replace('TEMPLATE', template_dict['MODEL_SHORT_NAME'])
    print('Generating {}...'.format(out_fname))
    dest_fname = os.path.join(fname_tup[1], out_fname)
    with open(dest_fname, 'wb') as f:
        f.write(fdata)
