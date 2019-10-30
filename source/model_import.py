import pickle
import stable_baselines
import numpy

# All weights and biases that make up an agent
possiblePars = [
    'model/c1/w:0',
    'model/c1/b:0',
    'model/c2/w:0',
    'model/c2/b:0',
    'model/c3/w:0',
    'model/c3/b:0',
    'model/fc1/w:0',
    'model/fc1/b:0',
    'model/lstm1/wx:0',
    'model/lstm1/wh:0',
    'model/lstm1/b:0',
    'model/vf/w:0',
    'model/vf/b:0',
    'model/pi/w:0',
    'model/pi/w:0',
    'model/q/w:0',
    'model/q/b:0'  # Seems to always be 0
]

model_pkl_path = './log/PPO2_MULTI_DISCRETE_seconds-35_policy-CnnLstmPolicy_Cars-64_LowEntropy_Baby_10.pkl'

with open(model_pkl_path, 'rb') as f:
    data = pickle.load(f)

# Print settings
# hypPars = data[0]
# for par in hypPars:
#     print(par, ":", hypPars[par])

allWeights = data[1]
for weights in allWeights:
    print(weights, ":", allWeights[weights])
