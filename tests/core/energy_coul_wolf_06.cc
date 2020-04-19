
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of a single barium atom, consisting of a core and a shell,
// interacting exclusively through Coulomb interactions using QC approach with
// full atomistic resolution.
// There exists a bond between the core and the shell.
// However, their bond energy here is zero as no bond style is declared.
// The blessed output is created through the LAMMPS input script
// included at the end.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run(const double &blessed_energy);
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run(
  const double &blessed_energy)
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->update_neighbor_lists();

  MPI_Barrier(this->mpi_communicator);

  Testing::SequentialFileStream write_sequentially(this->mpi_communicator);

  deallog << "picked up: "
          << this->cell_molecule_data.cell_energy_molecules.size()
          << " energy molecule(s)." << std::endl;

  const double energy =
    this->template compute<false>(this->locally_relevant_gradient);

  this->pcout << "The energy computed using PairCoulWolfManager "
              << "of charged atomistic system is: " << energy << " eV."
              << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(this->neighbor_lists.size(),
                                this->mpi_communicator);

  this->pcout << "Total number of neighbors " << total_n_neighbors << std::endl;

  // Accurate to only 1e-5, probably due to erfc() computations
  AssertThrow(std::fabs(energy - blessed_energy) <
                1e11 * std::numeric_limits<double>::epsilon(),
              ExcInternalError());
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 2." << std::endl
          << "    set Y center = 2." << std::endl
          << "    set Z center = 2." << std::endl
          << "    set X extent = 4." << std::endl
          << "    set Y extent = 4." << std::endl
          << "    set Z extent = 4." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 100" << std::endl
          << "  set Pair potential type = Coulomb Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 14.5" << std::endl
          << "  set Factor coul = 0." << std::endl
          << "  set Bond type = None" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = -1." << std::endl
          << "  set Cluster radius = 100" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "2 atoms" << std::endl
          << "1 bonds" << std::endl
          << std::endl
          << "2  atom types" << std::endl
          << "1  bond types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1  5.042 0. 0. 0." << std::endl
          << "2 1 2 -2.870 2. 2. 2." << std::endl
          << std::endl
          << "Bonds #" << std::endl
          << std::endl
          << "1 1 1 2" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager, 2> problem(config);
      problem.partial_run(46.87773021953579 /*blessed energy from LAMMPS*/);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}



/*
LAMMPS input script

Vishal Boddu 08.05.2017

// actual code below
#! /usr/bin/python3
"""
LAMMPS input script for energy_coul_wolf_06 test
"""

import math
import numpy

from lammps import lammps
from data   import barium_titanate_cs_charges

with open("test.data", 'w') as f:
    f.write(
"""LAMMPS Description

2 atoms
1 bonds

2  atom types
1  bond types

0.0 4  xlo xhi
0.0 4  ylo yhi
0.0 4  zlo zhi

Masses

  1 1.
  2 1.

Atoms

1 1 1  05.042 0. 0. 0.
2 1 2  -2.870 2. 2. 2.

Bonds

1 1 1 2
""")

cmds = [
    "units metal",
    "atom_style full",
    "boundary f f f",
    "read_data    test.data",
    "pair_style  coul/wolf 0.25 14.5",
    "pair_coeff	 *	*",
    "pair_modify tail no",
    "thermo_style custom step etotal pe evdwl ecoul elong ebond",
    "neigh_modify one 4000",
    "run 0",
    "variable ebond equal ebond",
    "variable ecoul equal ecoul",
    "variable evdwl equal evdwl",
    "variable pe equal pe",
    "print \"Bond Energy: ${ebond}\"",
    "print \"Coul Energy: ${ecoul}\"",
    "print \"Vdwl Energy: ${evdwl}\"",
    "print \"Pot  Energy: ${pe}\""
]

lmp = lammps()

for each in cmds:
    lmp.command(each)

def self_energy_factor(alpha, rc):
    return math.erfc(alpha*rc)/2/rc + alpha / math.sqrt(math.pi)

charges = numpy.array([05.042, -2.870])

selfce = self_energy_factor(0.25, 14.5) * 14.399645 * numpy.inner(charges,
charges) print("Self Energy", selfce)

# Bond Energy: 0
# Coul Energy: 21.4841128121098
# Pot  Energy: 21.4841128121098
# Self Energy 68.36184303164559
*/
