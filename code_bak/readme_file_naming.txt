<query>_<dataset>_<method>_<output>.csv

  query:
    dw_cycle - dog_walker holding leash attached to dog walked_by dog_walker
    stanford_dw_cycle - dog_walker holding leash attached to dog walked_by dog_walker
    handshake - person extending handshake, person handshaking person
    pingpong - player playing_pingpong_with player, player at table, net on table
   
  dataset:
    postest - the positive set, test only
    fullneg - the full negative set of 5k images
    hardneg - the hard negative set for the query
    origneg - the original negative set of dog walking images

  method:
    pgm - the IRSG graphical model method
	geo - geometric mean of top-scorimg class boxes

  output:
    energy - energy values
	ratk - r@k values
