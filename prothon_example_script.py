from Prothon import Prothon
import numpy as np

data = ['Q99.dcd','Q75.dcd','Q80.dcd','Q85.dcd','Q90.dcd','Q95.dcd']
topology = 'topology.pdb'

prothon = Prothon(data = data, topology = topology)
ensembles = prothon.ensemble_representation(measure = 'CBCN')
x_min, x_max = (np.min(ensembles), np.max(ensembles))

dissimilarity = []

for ensemble in ensembles[1:]:
    d = prothon.dissimilarity(ensemble, ensembles[0], x_min=x_min, x_max=x_max)
    dissimilarity.append(d[0])

print(dissimilarity)
