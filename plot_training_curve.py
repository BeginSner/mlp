import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.style as style
path = "BN+RC_block/result_outputs/summary.csv"
training_data = pd.read_csv(path, sep=',', header=0)
training_data = np.array(training_data)
style.use('ggplot')
epoch = np.arange(100) + 1
ax1 = plt.subplot(211)
ax1.plot(epoch, training_data[:,0], 'r-', linewidth=1, label='train')
ax1.plot(epoch, training_data[:,2], 'b-', linewidth=1, label='val')
ax1.set_xlabel('Epoch number')
ax1.set_ylabel('Accuracy')
ax1.set_title("Acc of training and validation")
ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(epoch, training_data[:,1], 'r-', linewidth=1, label='train')
ax2.plot(epoch, training_data[:,3], 'b-', linewidth=1, label='val')
ax2.set_xlabel('Epoch number')
ax2.set_ylabel('Loss')
ax2.set_title("Loss of training and validation")
ax1.grid(color='black',linewidth=0.5)
ax2.grid(color='black',linewidth=0.5)
ax1.legend()
ax2.legend()
plt.xlim(0,100)
plt.tight_layout()
# plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0.5, wspace = 0)
plt.margins(0,0)
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig("BN+RC_training_curve.pdf", bbox_inches='tight',pad_inches = 0)