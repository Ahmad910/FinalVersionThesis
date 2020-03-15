import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


# bara ge 2 intervall och plotta
CI1 = [ 0.7067493553330252 ,  0.8476841485350571 ]
CI2 = [ 0.7064200498871225 ,  0.840854317460451 ]
CIs = [CI1, CI2]
plt.boxplot(CIs)
plt.show()