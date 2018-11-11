import os
import sys
import json as j



def get_search_and_replace_fnames(srcdir, destdir):
    return [
        (os.path.join(srcdir, 'configs'),    os.path.join(destdir, 'configs'), 'configs_TEMPLATE.yml'),
        (os.path.join(srcdir, 'scripts'),    os.path.join(destdir, 'scripts'), 'finetune_TEMPLATE.sh'),
        (os.path.join(srcdir, 'scripts'),    os.path.join(destdir, 'scripts'), 'ft_full_TEMPLATE.sh'),
        (os.path.join(srcdir, 'scripts'),    os.path.join(destdir, 'scripts'), 'ft_init_TEMPLATE.sh'),
        (os.path.join(srcdir, 'scripts'),    os.path.join(destdir, 'scripts'), 'generate_TEMPLATE_cnn_and_gmm.sh'),
        (os.path.join(srcdir, 'scripts'),    os.path.join(destdir, 'scripts'), 'run_all_TEMPLATE.sh'),
        (os.path.join(srcdir, 'imdb'),       os.path.join(srcdir, 'imdb'), 'TEMPLATE_imdb.py'),
        (os.path.join(srcdir, 'model_defs'), os.path.join(destdir, 'model_defs'), 'config.yml'),
        (os.path.join(srcdir, 'model_defs', 'faster_rcnn_end2end'), os.path.join(destdir, 'model_defs', 'faster_rcnn_end2end'), 'solver_full.prototxt'),
        (os.path.join(srcdir, 'model_defs', 'faster_rcnn_end2end'), os.path.join(destdir, 'model_defs', 'faster_rcnn_end2end'), 'solver_init.prototxt'),
        (os.path.join(srcdir, 'model_defs', 'faster_rcnn_end2end'), os.path.join(destdir, 'model_defs', 'faster_rcnn_end2end'), 'test.prototxt'),
        (os.path.join(srcdir, 'model_defs', 'faster_rcnn_end2end'), os.path.join(destdir, 'model_defs', 'faster_rcnn_end2end'), 'train_full.prototxt'),
        (os.path.join(srcdir, 'model_defs', 'faster_rcnn_end2end'), os.path.join(destdir, 'model_defs', 'faster_rcnn_end2end'), 'train_init.prototxt')
    ]



#===============================================================================
def get_cfg():
    import argparse
    parser = argparse.ArgumentParser(description='Query prep tool')
    parser.add_argument('--cfg', dest='json_cfg')
    args = parser.parse_args()
    
    json_fname = args.json_cfg
    f = open(json_fname, 'rb')
    json_cfg = j.load(f)
    
    return [json_cfg]



if __name__ == '__main__':
    cfg = get_cfg()
    json_cfg = cfg[0]

    # bail out if cfg is not complete
    if not 'MODEL_SHORT_NAME' in json_cfg:
        sys.exit('Missing MODEL_SHORT_NAME')

    # search and replace in template file
    template_dict = json_cfg['template_vals']
    
    snr_keys = template_dict.keys()
    srcdir = '/home/econser/research/irsg_psu_pdx/utils/query_prep'
    destdir = '/home/econser/research/irsg_psu_pdx/utils/query_prep/out'
    snr_fnames = get_search_and_replace_fnames(srcdir, destdir)
    
    for fname_tup in snr_fnames:
        # open the template file
        src_fname = os.path.join(fname_tup[0], fname_tup[2])
        with open(src_fname, 'rb') as f:
            fdata = f.read()

        # replace keys
        for snr_key in snr_keys:
            snr_val = template_dict[snr_key].encode('ascii', 'ignore')
            k = '<'+snr_key.encode('ascii', 'ignore')+'>'
            fdata = fdata.replace(k, snr_val)

        # special handling for the config yml file
        if fname_tup[2] == 'configs_TEMPLATE.yml':
            # for each entry in bbox_configs
            # search and replace the remaining keys

        # special handling for TEMPLATE_impd.py
            # replace the class string
            # generate get_classes_from_anno
        
        # create output file
        if not os.path.exists(fname_tup[1]):
            os.makedirs(fname_tup[1])
        out_fname = fname_tup[2].replace('TEMPLATE', template_dict['MODEL_SHORT_NAME'])
        print('Generating {}...'.format(out_fname))
        dest_fname = os.path.join(fname_tup[1], out_fname)
        with open(dest_fname, 'wb') as f:
            f.write(fdata)
            # chmod if it's a script
            if '/scripts/' in dest_fname:
                os.chmod(dest_fname, 0755)
    
    # generate class names file
    datadir = os.path.join(destdir, 'data')
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    fname = os.path.join(datadir, template_dict['CLASS_STR_FNAME'])
    print('Generating class file...')
    with open(fname, 'wb') as f:
        num_classes = int(template_dict['NUM_CLASSES']) - 1 #ignore background class
        for i in range(0, num_classes):
            k = 'CLASS_{}_STR'.format(i+1)
            f.write('{}\r\n'.format(template_dict[k]))

    # output factory string
    factory_code = "import {}_imdb\n".format(template_dict['MODEL_SHORT_NAME']) + "for split in ['train', 'test']:\n    name = '{}_{}'" + ".format('{}', split)\n    __sets[name] = (lambda split=split: {}(split, '{}'))".format(template_dict['MODEL_NAME'], template_dict['MODEL_NAME'], template_dict['PROJECT_ROOT_DIR'])
    print('\ninsert the following into faster rcnn factory.py:\r\n{}'.format(factory_code))
