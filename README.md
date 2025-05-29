# Core spectroscopy for [PySCF](https://github.com/pyscf/pyscf)
This is a work in process, and mostly for myself because I have tons of the same code scattered everywhere. I thought it would be super convenient to just pip install a plugin.

**TODO:**
- [x] Implement ZORA
- [x] Add Core-valence separation
- [ ] Option for direct diagonalization of A/B matricies
- [ ] Option to disable $f_\text{xc}$ term
- [ ] Better documentation
- [ ] More tests

## Background
Core spectroscopy often involves excitations from a relatively small number of core orbitals. This is a huge advantage for linear response Time-Dependent Density Functional Theory (TDDFT) since you only need to consider core and valence orbitals. This is an application of core-valence separation. The theory behind this extends beyond response theory, but basically, core orbitals and valence orbitals have such vastly different localizations and energies that they are separable in the Schrödinger equation to good approximation.[^1]

PySCF already contains a good framework for TDDFT calculations. However, two things are inconvenient for core-level spectroscopy:

1. **Davidson diagonalization** is comically slow, around 100x slower than direct diagonalization under conditions relevant to our work. The number of excitations (occupied times virtual) is relatively small. For example, the K-edge spectrum of closed-shell systems can involve excitations out of one (1) occupied orbital. Also, we often require hundreds of states in our TDDFT calculations, sometimes around half of the total number of excitations. Performing a **direct diagonalization** using `*.linalg.eigh` is simply the better option here.

2. **Exchange and correlation** terms are often the most computationally expensive part of response TDDFT calculations. However, recent results from Pak and Nascimento[^2] show that the term is unnecessary for qualitatively-accurate X-ray absorption spectra. Also, the exclusion of the $f_\text{xc}$ term would remove the warning when using non-local correlation functionals (present at the time of writing).

3. **No ZORA.** The best scalar-relativistic correction.[^3]

## Usage
The Zeroth-Order Regular Approximation (ZORA) can be applied to any HF/KS object by appending the `zora` method.
```py
from pyscf import gto, scf
import pyscf.zora
mol = gto.M(...)
mf = scf.RHF(mol).zora() # wow! so easy
mf.run()
```
It works by replacing the core Hamiltonian of the SCF object with its scalar-relativistic counterpart.

You can specify excitations out of core orbitals by adding a `core_idx` attribute to the TDHF/TDDFT object after importing `pyscf.cvs`.
```py
from pyscf import gto, dft
from pyscf.tdscf import TDHF
import pyscf.cvs
mol = gto.M(...)
mf = dft.UKS(mol).run()

tdobj = TDDFT(mf)
tdobj.nstates = 80
tdobj.core_idx = ([0,1,2], [0,1,2]) # wow! so easy
tdobj.kernel()
```
In the above example, excitations out of the alpha and beta orbitals are specified in a tuple. Note that this technique is destructive to your SCF object. I might fix that later.

## Installation
The recommended installation method is to use `pip` with some kind of virtual environment (venv, conda, etc.)

### Pip
This software has been uploaded to [PyPI](https://pypi.org/project/core-spec-pyscf/), so it can be installed with
```sh
pip install core-spec-pyscf
```
Alternatively, install the latest version from the [GitHub](https://github.com/NathanGillispie/core-spec-pyscf) repo with
```sh
pip install git+https://github.com/NathanGillispie/core-spec-pyscf.git
```
It's highly recommended to use a virtual environment, or install locally with `pip install --user ...`. On Arch linux (I use Arch btw), you may still `--break-system-packages` with a `--user` install. To see why, try it yourself. Or just read this quote from [PEP 668](https://peps.python.org/pep-0668/):

> The `python3` executable available to the users of the distro and the `python3` executable available as a dependency for other software in the distro are typically the same binary. This means that if an end user installs a Python package using a tool like `pip` outside the context of a virtual environment, that package is visible to Python-language software shipped by the distro. If the newly-installed package (or one of its dependencies) is a newer, backwards-incompatible version of a package that was installed through the distro, it may break software shipped by the distro.

### Conda
If using `conda`, use the `pip` installed in your environment. The steps are the same as above. Some call this "bad practice", I call it time spent *not* running core-valence separated TDDFT calculations.

### Source build
This should only be done if you know what you're doing. After [installing and building](https://pyscf.org/user/install.html#build-from-source) PySCF, add the `pyscf` dir of this repo to the `PYSCF_EXT_PATH` environment variable, but be warned, this variable causes problems for pip installations of PySCF.

### Development mode
`pip` has a handy feature called editable installations. In a virtual environment with PySCF and its dependencies, run
```sh
pip install -e ./core-spec-tddft
```
Also, you can run some basic tests with `pytest`.

You can find details on other extensions in the [extensions](https://pyscf.org/user/extensions.html#how-to-install-extensions) page of the [PySCF website](https://pyscf.org).

[^1]: Cederbaum, L. S.; Domcke, W.; Schirmer, J. Many-Body Theory of Core Holes. _Phys. Rev. A_ **1980**, _22_ (1), 206–222. [doi.org/10.1103/PhysRevA.22.206](https://doi.org/10.1103/PhysRevA.22.206).

[^2]: Pak, S.; Nascimento, D. R. The Role of the Coupling Matrix Elements in Time-Dependent Density Functional Theory on the Simulation of Core-Level Spectra of Transition Metal Complexes. _Electron. Struct._ **2024**, _6_ (1), 015014. [doi.org/10.1088/2516-1075/ad2693](https://doi.org/10.1088/2516-1075/ad2693).

[^3]: In my opinion.
