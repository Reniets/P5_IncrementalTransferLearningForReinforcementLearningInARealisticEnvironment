import pickle
import stable_baselines
import numpy

model_pkl_path = './log/PPO2_MULTI_DISCRETE_seconds-35_policy-CnnLstmPolicy_Cars-64_LowEntropy_Baby_0_best.pkl'

with open(model_pkl_path, 'rb') as f:
    data = pickle.load(f)

# Print settings
hypPars = data[0]
for par in hypPars:
    print(par, ":", hypPars[par])

# allWeights = data[1]
# for weights in allWeights:
#     print(weights, ":", allWeights[weights])
