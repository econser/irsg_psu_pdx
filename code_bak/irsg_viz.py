#sk.gen_viz_file(queries['simple_graphs'][3], 3, ifu.sg_to_str(queries['simple_graphs'][3].annotations), vgd['vg_data_test'], tp_simple[3], ifdata, "/home/econser/School/Thesis/data/batch_analysis/")
#sk.gen_viz_file(queries['simple_graphs'][i], i, ifu.sg_to_str(queries['simple_graphs'][i].annotations), vgd['vg_data_test'], tp_simple[i], ifdata, "/home/econser/School/Thesis/data/batch_analysis/")
def gen_viz_file(query, query_id, query_str, image_set, tp_indices, ifdata, output_path):
  import image_fetch_core as ifc; reload(ifc)
  
  filename = 'q_' + str(query_id) + '.csv'
  f = open(output_path+filename, 'w')
  
  for i in range(0, len(image_set)):
    print(i)
    ifdata.configure(i, query.annotations)
    gm, tracker = ifc.generate_pgm(ifdata, verbose=False)
    energy, best_match_ix, marginals = ifc.do_inference(gm)
    
    line = '{:03d}, 0, "{}"\n'.format(i, query_str)
    f.write(line)
    line = '{:03d}, 2, {:0.4f}\n'.format(i, energy)
    f.write(line)
    
    if i in tp_indices:
      line = '{:03d}, 3, "match"\n'.format(i)
    else:
      line = '{:03d}, 3, "no match"\n'.format(i)
    f.write(line)
    
    for obj_ix in range(0, len(best_match_ix)):
      obj_name = tracker.object_names[obj_ix]
      box_ix = best_match_ix[obj_ix]
      bc = tracker.box_coords[box_ix]
      line = '{:03d}, 1, {}, "{}", {}, {}, {}, {}\n'.format(i, obj_ix, obj_name, int(bc[0]), int(bc[1]), int(bc[2]), int(bc[3]))
      f.write(line)
  
  f.close()
