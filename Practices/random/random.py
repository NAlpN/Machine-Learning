import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000
d = 10
toplam = 0
secilen = []

for n in range(0,N):
    ad = random.randrange(d)
    secilen.append(ad)
    odul = data.values[n,ad]
    toplam = toplam + odul

plt.hist(secilen)
plt.show()