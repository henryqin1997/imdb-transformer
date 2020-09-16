from matplotlib import pyplot as plt
import json
import math

# tacc = json.load(open('imdb_batch128_lr1.0_epoch5_exp-rt.json'))
# tacc = json.load(open('imdb_lamb_batch128_lr1.0_epoch5_exp-rt.json'))

tacc = json.load(open('imdb_adam_batch128_lr1.0_epoch5_exp-rt.json'))
low = math.log2(1e-5)
high = math.log2(10)
log_neg_one = math.log2(0.1)
log_neg_two = math.log2(0.01)
x= [2 ** (low + (high - low) * i / 196 / 5) for i in range(len(tacc))]
tacc = tacc[0:int(len(tacc)*(log_neg_one-low)/(high-low))]
x = x[0:int(len(x)*(log_neg_one-low)/(high-low))]
plt.plot(x,tacc)
plt.xscale('log')
plt.xlabel('learning rate')
plt.ylabel('loss')
plt.title('IMDB-TRANSFORMER ADAM EXP-RT')
plt.show()

# tacc,vacc = json.load(open('imdb_lamb_batch128_lr0.01_epoch20onecycle.json'))
# plt.plot(tacc,label='training accuracy')
# plt.plot(vacc,label='test accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.title('LAMB on IMDB-TRANSFORMER with One Cycle')
# plt.legend()
# plt.show()