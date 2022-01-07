What is deal.II-qc?
===================

deal.II-qc is a C++ program library to conduct coarse-grained atomistic simulations
using the fully nonlocal quasicontinuum method.

Requirements
------------

In short, deal.II-qc is configured using CMake and as its name suggests
it requires [deal.II](https://github.com/dealii/dealii).
Requirements to build deal.II-qc can be met with significant ease using a [spack environment](https://spack.readthedocs.io/en/latest/environments.html).
Make sure you have `spack` installed by following the instructions provided [here](https://spack.readthedocs.io/en/latest/getting_started.html#installation).
Using the `spack.yaml` manifest file:
```bash
>> cat ./spack.yaml
# This is a Spack Environment file.
#
# It describes a set of packages to be installed, along with
# configuration settings.
spack:
  # add package specs to the `specs` list
  specs:
  - cmake@3.21.4
  - dealii@9.2.0 +mpi +trilinos ~int64 ~gmsh ~ginkgo ^trilinos@12.18.1+rol
  - doxygen@1.8.20
  - numdiff
  - python@3.8.12
  - py-ase@3.21.1
  - py-setuptools@57.4.0
  concretization: together
  view: true
  packages:
    openmpi:
      version: [4.1.1]
```
a spack environment named `dealiiqc` can be created by:
```bash
spack env create dealiiqc ./spack.yaml
```
and do
```bash
spack env activate dealiiqc
```
to activate the environment.

Now that all the dependencies for deal.II-qc are installed, the project can be
built using the following commands:
```bash
cd /path/to/pull/deal.ii-qc/from/github
git clone https://github.com/ltm-erlangen/deal.ii-qc
cd deal.ii-qc
mkdir _build
cd _bluid
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install/deal.ii-qc
make -j 4 && make install
```
and
```bash
ctest
```
to run the test suite.

License
-------

deal.II-qc is licensed under the LGL v2 or newer.
