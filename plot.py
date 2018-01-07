import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import curve_fit
import scipy.constants.constants as const
from uncertainties import ufloat
from uncertainties import unumpy

#plot1
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})

x, y = np.genfromtxt('content/Messwerte1.txt', unpack=True)



plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.yscale('log')
plt.grid(True, which='both')


# Fitvorschrift
def f(x, A, B):
    return A*x + B      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(f, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(x, f(x, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")



plt.tight_layout()
plt.legend()
plt.savefig('build/messung1.pdf')
plt.clf()

#plot2
x, y = np.genfromtxt('content/Messwerte2.txt', unpack=True)

U_0 = 12.99
z = y/U_0

plt.xlabel(r'$t/\si{\second}$')
plt.ylabel(r'$U/\si{\volt}$')
plt.xscale('log')
plt.grid(True, which='both')

plt.plot(x, z, ".", color="xkcd:blue", label="Messwerte")

# Fitvorschrift
def g(x, A):
    return z/np.sqrt(1+(2*np.pi*x)**2*A**2)      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(g, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

plt.plot(x, g(x, *params), "xkcd:orange", label=r'Regression' )
plt.tight_layout()
plt.legend()
plt.savefig('build/messung2.pdf')
plt.clf()
