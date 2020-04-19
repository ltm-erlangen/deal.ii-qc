# -*- coding: utf-8 -*-

"""
Augmenting existing ASE chemical elements with
other fictitious chemical elements used by the project.
"""

import numpy as np

from ase.data import atomic_masses
from ase.data import atomic_names
from ase.data import atomic_numbers
from ase.data import chemical_symbols


__all__ = ['barium_titanate_cs_bonds',
           'barium_titanate_cs_charges',
           'barium_titanate_cs_masses',
           'barium_titanate_cs_molecule_ids',
           'barium_titanate_cs_positions'
           ]

chemical_symbols.append('Bs')
chemical_symbols.append('Ts')
chemical_symbols.append('Os')
atomic_numbers['Bs'] = 119
atomic_numbers['Ts'] = 120
atomic_numbers['Os'] = 121
atomic_names.append('Bs')
atomic_names.append('Ts')
atomic_names.append('Os')
np.append(atomic_masses, 2.000)
np.append(atomic_masses, 2.000)
np.append(atomic_masses, 2.000)


barium_titanate_cs_bonds = np.array([
    [1, 1, 1, 2],
    [2, 2, 3, 4],
    [3, 3, 5, 6],
    [4, 3, 7, 8],
    [5, 3, 9, 10]
])

barium_titanate_cs_charges = np.array([
    05.042, # Ba core  # Ba
    -2.870, # Ba shell # Bs
    04.616, # Ti core  # Ti
    -1.544, # Ti shell # Ts
    00.970, # O  core  # O
    -2.718, # O  shell # Os
    00.970, # O  core  # O
    -2.718, # O  shell # Os
    00.970, # O  core  # O
    -2.718  # O  shell # Os
])

barium_titanate_cs_masses =  np.array([
    135.327, # Ba core  # Ba
    002.000, # Ba shell # Bs
    045.867, # Ti core  # Ti
    002.000, # Ti shell # Ts
    013.999, # O  core  # O
    002.000, # O  shell # Os
    013.999, # O  core  # O
    002.000, # O  shell # Os
    013.999, # O  core  # O
    002.000  # O  shell # Os
])

barium_titanate_cs_positions = np.array([
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0.0, 0.5, 0.5),
    (0.0, 0.5, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.0, 0.5),
    (0.5, 0.5, 0.0),
    (0.5, 0.5, 0.0)
])
