import sys
#from rng_seeds import *
seeds = [6598903756360202179, 2908409715321502665, 6126375328734039552, 1447957147463681860, 8611858271322161001, 1129180857020570158, 6362222119948958210, 7116573423379052515, 6183551438103583226, 4025455056998962241, 3253052445978017587, 8447055112402476503, 5958072666039141800, 704315598608973559, 1273141716491599966, 8030825590436937002, 6692768176035969914, 8405559442957414941, 5375803109627817298, 1491660193757141856, 3436611086188602011, 3271002097187013328, 4006294871837743001, 7473817498436254932, 7891796310200224764, 3130952787727334893, 697469171142516880, 133987617360269051, 1978176412643604703, 3541943493395593807, 5679145832406031548, 5942005640162452699, 5170695982942106620, 3168218038949114546, 9211443340810713278, 675545486074597116, 3672488441186673791, 6678020899892900267, 2416379871103035344, 8662874560817543122, 2122645477319220395, 2405200782555244715, 6145921643610737337, 5436563232962849112, 8616414727199277108, 3514934091557929937, 6828532625327352397, 4198622582999611227, 1404664771100695607, 2109913995355226572, 7499239331133290294, 1663854912663070382, 8773050872378084951, 847059168652279875, 2080440852605950627, 842456810578794799, 2969610112218411619, 8028963261673713765, 8849431138779094918, 6906452636298562639, 8279891918456160432, 3007521703390185509, 7384090506069372457, 2587992914778556505, 7951640286729988102, 812903075765965116, 4795333953396378316, 1140497104356211676, 8624839892588303806, 5867085452069993348, 8978621560802611959, 8687506047153117100, 1433098622112610322, 2329673189788559167, 1697681906179453583, 1151871187140419944, 7331838985682630168, 2010690807327394179, 8961362099735442061, 3782928183186245068, 8730275423842935904, 2250089307129376711, 6729072114456627667, 6426359511845339057, 1543504526754215874, 6764758859303816569, 438430728757175362, 850249168946095159, 7241624624529922339, 633139235530929889, 8443344843613690342, 5097223086273121, 3838826661110586915, 7425568686759148634, 5814866864074983273, 5375799982976616117, 6540402714944055605, 448708351215739494, 5101380446889426970, 8035666378249198606]



POPULATION_SIZE = 20
NUMBER_OF_ITERATIONS = 16
ELITISM = 2 #number of individuals to copy to the next generation
TOURNAMENT = 2
PROB_CROSSOVER = 0.85
PROB_MUTATION = 0.1
RUN = len(sys.argv) > 1 and int(sys.argv[1]) or 0
SEED = seeds[RUN]
ADD_PHENOTYPE_TO_JSON_OBJECT = True
sampling_snap = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
