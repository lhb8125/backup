import matplotlib.pyplot as plt
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description='Parses log file and generates train/val curves')
parser.add_argument('--log-file-1', type=str,default="resnet50_big.log",
                            help='the path of log file')
parser.add_argument('--log-file-2', type=str,default="resnet50_big_globalbn.log",
                            help='the path of log file')
args = parser.parse_args()

#VA_RE = re.compile('.*?]\sValidation-IOU=([\d\.]+)')
VA_RE = re.compile('.*?]\sTrain-AccWithIgnore=([\d\.]+)')

log1 = open(args.log_file_1).read()
log2 = open(args.log_file_2).read()

log_va_1 = [float(x) for x in VA_RE.findall(log1)]
log_va_2 = [float(x) for x in VA_RE.findall(log2)]
idx = np.arange(len(log_va_1))

#log_va_2 = log_va_2[:len(log_va_1)]
plt.figure(figsize=(8, 6))
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.plot(idx, log_va_1, ',', linestyle='-', color="r",
                 label="Train Accuracy")

plt.plot(idx, log_va_2, ',', linestyle='-', color="b",
                 label="Train Accuracy with global BN")

plt.legend(loc="best")
plt.xticks(np.arange(min(idx), max(idx)+1, 10))
plt.yticks(np.arange(0.8, 1, 0.02))
plt.ylim([0.8,1])
plt.show()
plt.savefig("training_curve.png",dpi=300)
