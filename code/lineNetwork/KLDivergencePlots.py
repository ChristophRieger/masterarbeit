# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:35:09 2023

@author: chris
"""

import matplotlib.pyplot as plt

plt.close("all")

# fPrior = 0
# tauDecay = 0.015
fInputWithout15 =          [10     , 20     , 30     , 34     , 36     , 38     , 40     , 42     , 44     , 46     , 48     , 50     , 60     , 70     , 80     , 90]
KLDivergenceWithout15 =    [0.1916 , 0.0935 , 0.0424 , 0.0319 , 0.0274 , 0.0255 , 0.0251 , 0.0241 , 0.0246 , 0.0255 , 0.0284 , 0.0305 , 0.0545 , 0.0915 , 0.1372 , 0.1898]                      
KLDivergenceStdWithout15 = [0.00130, 0.00293, 0.00230, 0.00174, 0.00148, 0.00132, 0.00159, 0.00159, 0.00114, 0.00182, 0.00147, 0.00235, 0.00304, 0.00400, 0.00443, 0.00472]                                                      
plt.figure()
plt.errorbar(fInputWithout15, KLDivergenceWithout15, KLDivergenceStdWithout15, marker='.', ecolor="Red",)
plt.xlabel("Input firing rate [Hz]")
plt.ylabel("KL divergence")
plt.title("Search for best $f_{input}$")
plt.savefig("KLDvsfInput_fPrior0tau15" + ".svg")  
plt.savefig("KLDvsfInput_fPrior0tau15" + ".png")
# fPrior = 0
# tauDecay = 0.004
fInputWithout4 =          [50     , 80     , 84     , 86     , 88     , 90     , 92     , 94     , 100    , 110    , 120    , 130]
KLDivergenceWithout4 =    [0.0727 , 0.0261 , 0.0242 , 0.0238 , 0.0233 , 0.0239 , 0.0244 , 0.0250 , 0.0278 , 0.0367 , 0.0504 , 0.0674]
KLDivergenceStdWithout4 = [0.00232, 0.00118, 0.00138, 0.00144, 0.00145, 0.00160, 0.00121, 0.00138, 0.00195, 0.00218, 0.00206, 0.00206]
plt.figure()
plt.errorbar(fInputWithout4, KLDivergenceWithout4, KLDivergenceStdWithout4, marker='.', ecolor="Red",)
plt.xlabel("Input firing rate [Hz]")
plt.ylabel("KL divergence")
plt.title("Search for best $f_{input}$")
plt.savefig("KLDvsfInput_fPrior0tau4" + ".svg")  
plt.savefig("KLDvsfInput_fPrior0tau4" + ".png")

#  With Prior
# tauDecay = 0.015
# fInput = 42
fPriorWith15 =          [160    , 180    , 200    , 216    , 218    , 220    , 222    , 224    , 226    , 228    , 230    , 240]
KLDivergenceWith15 =    [0.0454 , 0.0316 , 0.0224 , 0.0187 , 0.0187 , 0.0184 , 0.0188 , 0.0185 , 0.0181 , 0.0181 , 0.0184 , 0.0192]
KLDivergenceStdWith15 = [0.00230, 0.00167, 0.00138, 0.00140, 0.00172, 0.00105, 0.00145, 0.00131, 0.00136, 0.00111, 0.00178, 0.00143]
plt.figure()
plt.errorbar(fPriorWith15, KLDivergenceWith15, KLDivergenceStdWith15, marker='.', ecolor="Red",)
plt.xlabel("Prior firing rate [Hz]")
plt.ylabel("KL divergence")
plt.title("Search for best $f_{prior}$")
plt.savefig("KLDvsfPrior_fInput42tau15" + ".svg")  
plt.savefig("KLDvsfPrior_fInput42tau15" + ".png")
# tauDecay = 0.004
# fInput = 88
fPriorWith4 =          [360    , 380    , 400    , 420    , 434    , 436    , 438    , 440    , 442    , 444    , 446    , 460]
KLDivergenceWith4 =    [0.0214 , 0.0162 , 0.0125 , 0.0114 , 0.0112 , 0.0112 , 0.0112 , 0.0112 , 0.0112 , 0.0112 , 0.0114 , 0.0125]
KLDivergenceStdWith4 = [0.00158, 0.00108, 0.00100, 0.00082, 0.00077, 0.00107, 0.00108, 0.00088, 0.00150, 0.00113, 0.00087, 0.00109]
plt.figure()
plt.errorbar(fPriorWith4, KLDivergenceWith4, KLDivergenceStdWith4, marker='.', ecolor="Red",)
plt.xlabel("Prior firing rate [Hz]")
plt.ylabel("KL divergence")
plt.title("Search for best $f_{prior}$")
plt.savefig("KLDvsfPrior_fInput88tau4" + ".svg")  
plt.savefig("KLDvsfPrior_fInput88tau4" + ".png")


# Last: 
# fPrior = 440
# tauDecay = 0.004
fInput =          [88     , 90     , 92     , 94     , 96     , 98     , 100    , 102,     104    , 106    , 108    , 110]
KLDivergence =    [0.0111 , 0.0103 , 0.0104 , 0.0102 , 0.0102 , 0.0101 , 0.0102 , 0.0108 , 0.0111 , 0.0122 , 0.0123 , 0.0134]
KLDivergenceStd = [0.00101, 0.00127, 0.00085, 0.00112, 0.00082, 0.00095, 0.00088, 0.00094, 0.00118, 0.00106, 0.00130, 0.00091]
plt.figure()
plt.errorbar(fInput, KLDivergence, KLDivergenceStd, marker='.', ecolor="Red",)
plt.xlabel("Input firing rate [Hz]")
plt.ylabel("KL divergence")
plt.title("Search for best $f_{input}$")
plt.savefig("KLDvsfInput_fPrior440tau4" + ".svg")  
plt.savefig("KLDvsfInput_fPrior440tau4" + ".png")

# Last: 
# fInput = 98
# fPrior = 440
# tauDecay = 0.004
c =          [1, 2, 3, 4]
KLDivergence =    [0.0625, 0.0377, 0.0342, 0.0447]
KLDivergenceStd = [0.00238, 0.00194, 0.00160, 0.00242]
plt.figure()
plt.errorbar(c, KLDivergence, KLDivergenceStd, marker='.', ecolor="Red",)
plt.xlabel("c")
plt.ylabel("KL divergence")
plt.title("Search for best c")
plt.savefig("KLD_cvsfInput98_fPrior440tau4" + ".svg")  
plt.savefig("KLD_cvsfInput98_fPrior440tau4" + ".png")







