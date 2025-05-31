# Core spectroscopy for [PySCF](https://github.com/pyscf/pyscf)
[![pytest](https://github.com/NathanGillispie/core-spec-pyscf/actions/workflows/ci.yml/badge.svg)](https://github.com/NathanGillispie/core-spec-pyscf/actions/workflows/ci.yml)

## Background
Core spectroscopy often involves excitations from a relatively small number of core orbitals. This is a huge advantage for linear response Time-Dependent Density Functional Theory (TDDFT) since you can apply core-valence separation. In theory, core orbitals and valence orbitals have such vastly different localizations and energies that they are separable in the Schrödinger equation to good approximation.[^1]

PySCF provides a good basis for TDDFT calculations. However, some things are inconvenient for core-level spectroscopy:

1. **Davidson diagonalization** is comically slow, around 100x slower than direct diagonalization under conditions relevant to our work, due to excitations from a small number of core orbitals. Also, we often require hundreds of states in our TDDFT calculations, outweighing the benefits of the Davidson scheme. A **direct diagonalization** of the hermitian AB matrices using `*.linalg.eigh` is simply the better option here.

2. **Exchange and correlation** terms are often the most computationally expensive part of response TDDFT calculations. However, recent results from Pak and Nascimento[^2] show that the term is unnecessary for qualitatively-accurate X-ray absorption spectra.

3. **No ZORA.** The best scalar-relativistic correction.[^3]

### Details
- The ZORA correction uses a model basis. The exact values come from [NWCHEM](https://nwchemgit.github.io/).
- The diagonalization of Casida's equation[^4]
```math
\begin{pmatrix}\mathbf{A} & \mathbf{B}\\ \mathbf{-B}&\mathbf{-A}\end{pmatrix}\begin{pmatrix}\mathbf{X}\\ \mathbf{Y}\end{pmatrix}=\Omega \begin{pmatrix}\mathbf{X}\\ \mathbf{Y}\end{pmatrix}
```
is done in its hermitian form, assuming $(\mathbf{A}-\mathbf{B})$ and $(\mathbf{A}+\mathbf{B})$ are positive semi-definite:
```math
\begin{gather}\mathbf{CZ}=\Omega^2 \mathbf{Z}\\ \mathbf{C} = (\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{A}+\mathbf{B})(\mathbf{A}-\mathbf{B})^{1/2}\\ \mathbf{Z} = (\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{X}-\mathbf{Y})\end{gather}
```
- When removing the $f_\text{xc}$ term, the exact Hartree exchange is included, regardless of the functional used.

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
from pyscf.tdscf import TDA, TDDFT, TDHF # etc.
import pyscf.cvs
mol = gto.M(...)
mf = dft.RKS(mol).run()

tdobj = TDDFT(mf)
tdobj.nstates = 80
tdobj.core_idx = [0,1,2] # wow! so easy
tdobj.kernel()
```
For unrestricted references, excitations out of the alpha and beta orbitals are specified in a tuple. Note that this is destructive to the SCFs `mo_coeff`, `mo_occ`, `mo_energy` and MOLs `nelec`. I might fix that later.

To disable the $f_\text{xc}$ term, set the `no_fxc` attribute or keyword argument of the `kernel` function. The same syntax is used for direct diagonalizaton (`direct_diag`). Note that `pyscf.cvs` must still be imported as all the direct diagonalization code lives there.
```py
import pyscf.cvs

tdobj = TDHF(mf)
tdobj.no_fxc = True
tdobj.direct_diag = True
tdobj.kernel()
```

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
If using `conda`, use the `pip` installed in your environment. Some call this "bad practice", I call it time spent *not* running core-valence separated TDDFT calculations.

### Source build
This should only be done if you know what you're doing. After [installing and building](https://pyscf.org/user/install.html#build-from-source) PySCF, add the `pyscf` dir of this repo to the `PYSCF_EXT_PATH` environment variable. But be warned, this variable causes problems for pip installations of PySCF.

### Development mode
`pip` has a handy feature called editable installations. In a virtual environment with PySCF and its dependencies, run
```sh
pip install -e ./core-spec-tddft
```
Also, you can run my tests with `pytest`.

You can find details on other extensions in the [extensions](https://pyscf.org/user/extensions.html#how-to-install-extensions) page of the [PySCF website](https://pyscf.org).

[^1]: Cederbaum, L. S.; Domcke, W.; Schirmer, J. Many-Body Theory of Core Holes. _Phys. Rev. A_ **1980**, _22_ (1), 206–222. [doi.org/10.1103/PhysRevA.22.206](https://doi.org/10.1103/PhysRevA.22.206).

[^2]: Pak, S.; Nascimento, D. R. The Role of the Coupling Matrix Elements in Time-Dependent Density Functional Theory on the Simulation of Core-Level Spectra of Transition Metal Complexes. _Electron. Struct._ **2024**, _6_ (1), 015014. [doi.org/10.1088/2516-1075/ad2693](https://doi.org/10.1088/2516-1075/ad2693).

[^3]: In my opinion.

[^4]: Casida, M. E. Time-Dependent Density Functional Response Theory for Molecules. In _Recent Advances in Density Functional Methods_; Recent Advances in Computational Chemistry; World Scientific, **1995**; Vol. 1, pp 155–192. [doi.org/10.1142/9789812830586_0005](https://doi.org/10.1142/9789812830586_0005)
