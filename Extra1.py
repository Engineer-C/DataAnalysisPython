"""
CE614 Homework1

The goal of this prject is visualizing the difference between Amplification Factor.
P/P_{cr} ranging from 0 to 0.6, the error between original amplicifation factor
to approximate expression for the amplification factor is less than 2%.
No input data is required.
"""
import numpy as np
import matplotlib.pyplot as plt


POverPcr = np.linspace(0, .6, 1000)
u = np.pi / 2 * np.sqrt(POverPcr)
Chi = 3 * (np.tan(u) - u) / u ** 3
Etha = 12 * (2 * 1 / np.cos(u) - 2 - u ** 2) / (5 * u ** 4)
Lambda = 2 * (1 - np.cos(u)) / (u ** 2 * np.cos(u))

plt.figure(figsize=[6, 5], dpi=300, tight_layout=True)
plt.suptitle(r'Comparison of $\mathregular{\chi(u)}$, '
             r'$\mathregular{\eta(u)}$, '
             r'$\mathregular{\lambda(u)}$, '
             r'$\mathregular{\frac{1}{1 - \frac{P}{P_{cr}}}}$',
             fontweight='bold')
plt.plot(POverPcr, Chi,
         color='coral',
         label=r'$\chi(u)=\frac{3(\mathrm{tan}(u)-u)}{u^{3}}$')
plt.plot(POverPcr, Etha,
         color='goldenrod',
         label=r'$\eta(u)=\frac{12(2\mathrm{sec}(u)-2-u^{2})}{5u^{4}}$')
plt.plot(POverPcr, Lambda,
         color='darkcyan',
         label=r'$\lambda(u)=\frac{2(1-\mathrm{cos}(u))}{u^{2}\mathrm{cos}(u)}$')
plt.plot(POverPcr, 1 / (1 - POverPcr),
         color='black',
         linewidth=2,
         dashes=(5, 2),
         label=r'$\mathregular{\frac{1}{1 - P / P_{cr}}}$')
plt.text(0.02, 0.65, r'$\mathregular{u = \frac{\pi}{2} \sqrt{P / P_{cr}}}$',
         transform=plt.gca().transAxes,
         verticalalignment='top')
plt.xlim([0, .6])
plt.xlabel(r'$\mathregular{P / P_{cr}}$')
plt.ylim([1, 2.5])
plt.legend()
plt.savefig('Result.png')
