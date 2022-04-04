from joblib import Parallel, delayed
import numpy as np
np.random.seed
import time

np.random.seed()

def par_test():
    time.sleep(2)

for _ in range(10):
    par_test()

Parallel(n_jobs=-1,verbose=0)(delayed(par_test)() for _ in range(10)) # 1e2这种写法表示的是float而不是int