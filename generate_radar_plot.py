import numpy as np
import matplotlib.pyplot as plt
from math import pi
import argparse
import os



# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('InputFile', metavar='inputfile', type=str, help='the input file to process')
# args = parser.parse_args()

# filename, _ = os.path.splitext(args.InputFile)

scores = np.load('results_kNN.npy')
print("Folds:\\n", scores)
scores = np.mean(scores, axis=1).T

metrics=['Precision', 'Specificity', 'F1', 'BAC']
methods=['None','ROS', 'SMOTE', 'RUS']

N = len(metrics) 
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], metrics)

ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)

for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)

plt.savefig(f"radar_kNN", dpi=200)
