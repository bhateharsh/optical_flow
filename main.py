import numpy as np
from optical_flow import opticalFlow
from utils import flow_to_color, read_flow_file
import os

save_path = 'results/flowplot.png'
filepaths = ('images/Drop/',
             'images/GrassSky/',
             'images/RubberWhale/',
             'images/Urban/')
# Note: Drop is the image used for optimal solution

# hs = opticalFlow(filepaths[0], nos_itr=500)

# # Plotting Iteration Errors
# # for itr in range(100,2000, 100):
# #     u,v = hs.horn_schunck(nos_itr=itr)
# #     hs.benchmark_flow(u,v)
# # hs.plot_iteration_stats()

# itr = 750

# # Plotting Alpha Errors
# for Lambda in np.arange(0.5, 5, 0.5):
#     u,v = hs.horn_schunck(alpha=Lambda ,nos_itr=itr)
#     hs.benchmark_flow(u,v)
# hs.plot_alpha_stats()

# Plotting general Plot
itr = 750
Lambda = 2
target = ('results/Drop.png',
             'results/GrassSky.png',
             'results/RubberWhale.png',
             'results/Urban.png')

for i in range(0,4,1):
    hs = opticalFlow(filepaths[i], nos_itr=itr)
    u,v = hs.horn_schunck(nos_itr=itr, alpha=Lambda)
    hs.plot_flow(u,v)
    os.rename(save_path, target[i])
    break