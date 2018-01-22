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

lin = np.linspace(x[0], x[-1], 1000)
plt.plot(lin, f(lin, *params), "xkcd:orange", label=r'Regression' )
plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")



plt.tight_layout()
plt.legend()
plt.savefig('build/messung1.pdf')
plt.clf()

#plot2
x, y = np.genfromtxt('content/Messwerte2.txt', unpack=True)

U_0 = 12.99
y/=U_0
plt.xlabel(r'$t/\si[per-mode=reciprocal]{\per\second}$')
plt.ylabel(r'$A/U_0$')
plt.xscale('log')
plt.grid(True, which='both')

plt.plot(x, y, ".", color="xkcd:blue", label="Messwerte")

# Fitvorschrift
def g(x, A):
    return 1/np.sqrt(1+(2*np.pi*x)**2*A**2)      #jeweilige Fitfunktion auswaehlen:

params, covar = curve_fit(g, x, y)            #eigene Messwerte hier uebergeben
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('A') + i), "=" , uparams[i])
print()

lin = np.linspace(x[0], x[-1], 10000)
plt.plot(lin, g(lin, *params), "xkcd:orange", label=r'Regression' )
plt.tight_layout()
plt.legend()
plt.savefig('build/messung2.pdf')
plt.clf()


data = np.genfromtxt("content/messwerte3tab.txt", unpack=True)
phi = 2*np.pi*data[0]/data[1]

def h(f, RC):
    return np.arctan(-2*np.pi*f*RC)


params, covar = curve_fit(h, data[2], phi)
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))

print(uparams)

lin = np.linspace(data[2][0], data[2][-3], 10000)

plt.plot(lin, h(lin, *params), "xkcd:orange", label="Regression")
plt.xlabel(r"$f/\si{\hertz}$")
plt.ylabel(r"$\phi$")
plt.xscale("log")
plt.plot(data[2], phi, ".", color="xkcd:blue", label="Messwerte")
plt.tight_layout()
plt.grid(True, which="both")
plt.legend()
plt.savefig("build/messung3.pdf")
plt.clf()

A = np.array(  [12.51,  11.48,  7.29,   2.53,   0.768,  0.131])
Phi = np.array([0.252,  0.478,  0.981,  1.357,  1.496,  1.533])

A/= U_0

lin = np.linspace(0, np.pi/2, 1000)

ax = plt.subplot(111, projection='polar')
ax.plot(lin, np.cos(lin), "-", color="xkcd:orange", label="Theoriekurve")
ax.plot(Phi, A, ".", color="xkcd:blue", label="Messwerte")
ax.set_rticks([0, 0.25, 0.5, 0.75, 1])
ax.set_rlabel_position(-22.5)

plt.legend(loc="best")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("build/polar.pdf")
plt.clf()
