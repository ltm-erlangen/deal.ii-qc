
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of a single BaTiO3 molecule,
// whose atoms interact through BornCutClass2CoulWolfManager,
// using QC approach with full atomistic resolution.
// Because the distance between cores and shells of respective atoms is
// non-zero, the bond energy is non-zero.
// The blessed output is created using LAMMPS python script.
// The script is included at the end.



template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  partial_run(const double &blessed_energy);
  void
  displace_shells(const dealii::Point<dim> &displacement);
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::displace_shells(
  const dealii::Point<dim> &displacement)
{
  for (auto &entry : this->cell_molecule_data.cell_energy_molecules)
    for (auto &atom : entry.second.atoms)
      if (atom.type % 2) // if the atom type is odd (i.e., shells)
        atom.position += displacement;
}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::partial_run(
  const double &blessed_energy)
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->displace_shells(dealii::Point<dim>(.1, .1, .1));
  this->update_neighbor_lists();

  MPI_Barrier(this->mpi_communicator);

  Testing::SequentialFileStream write_sequentially(this->mpi_communicator);

  const auto &cell_molecules = this->cell_molecule_data.cell_energy_molecules;

  deallog << "picked up: " << cell_molecules.size() << " energy molecule(s)."
          << std::endl;

  const double energy =
    this->template compute<false>(this->locally_relevant_gradient);

  this->pcout << "The energy computed using PairBornCutClass2CoulWolfManager "
              << "of atomistic system is: " << std::fixed
              << std::setprecision(3) << energy << " eV." << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(this->neighbor_lists.size(),
                                this->mpi_communicator);

  this->pcout << "Total number of neighbors " << total_n_neighbors << std::endl;

  for (auto entry : this->neighbor_lists)
    this->pcout << "Molecule I: " << entry.second.first->second.global_index
                << '\t'
                << "Molecule J: " << entry.second.second->second.global_index
                << std::endl;

  // Accurate to only 1e-3, probably due to erf() and overlapping cores & shells
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
          << "  set Number of atom types = 6" << std::endl
          << "  set Maximum cutoff radius = 100" << std::endl
          << "  set Pair potential type = Born Class2 Coul Wolf" << std::endl
          << "  set Pair global coefficients = 0.25, 14.5, 16.0" << std::endl
          << "  set Factor coul = 0." << std::endl
          << "  set Pair specific coefficients = "
             "*, *,    0.00,	1.0000,	0.000,	0.0000,	0.000;"
             "2, 6, 7149.81,	0.3019,	0.000,	0.0000,	0.000;"
             "4, 6, 7200.27,	0.2303,	0.000,	0.0000,	0.000;"
             "6, 6, 3719.60,	0.3408,	0.000,	597.17,	0.000;"
          << std::endl
          << "  set Bond type = Class2" << std::endl
          << "  set Bond specific coefficients = "
             "1,  2,  0.000, 149.255, 0.0,   0.0000000;"
             "3,  4,  0.000, 153.070, 0.0,  20.83333333;"
             "5,  6,  0.000,  18.465, 0.0, 208.33333333;"
          << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/BaTiO3_cs_1x1x1_qcatom.data" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = -1." << std::endl
          << "  set Cluster radius = 100" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairBornCutClass2CoulWolfManager, 10> problem(
        config);
      problem.partial_run(77.06623890330121 /*blessed energy from LAMMPS*/);
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
LAMMPS input script for energy_born_class2_coul_wolf_04 test
"""
import math
import numpy

from lammps import lammps
from data   import barium_titanate_cs_charges

with open("test.data", 'w') as f:
    f.write(
"""BaTiO3_cs_1x1x1_atom.data (written by ASE) modified to displace cells

10 	 atoms
5 	 bonds
6  atom types
3  bond types

0.0                       4  xlo xhi
0.0                       4  ylo yhi
0.0                       4  zlo zhi

Masses

     1                 135.327
     2                       2
     3      45.866999999999997
     4                       2
     5      13.999000000000001
     6                       2

Atoms

1   1   1  5.042    0       0       0
2   1   2  -2.87    0.1     0.1     0.1
3   2   3  4.616    2       2       2
4   2   4 -1.544    2.1     2.1     2.1
5   3   5   0.97    0       2       2
6   3   6 -2.718    0.1     2.1     2.1
7   4   5   0.97    2       0       2
8   4   6 -2.718    2.1     0.1     2.1
9   5   5   0.97    2       2       0
10  5   6 -2.718    2.1     2.1     0.1

Bonds

1	1	1	2
2	2	3	4
3	3	5	6
4	3	7	8
5	3	9	10
""")

cmds = [
    "units metal",
    "atom_style full",
    "boundary   f f f",
    "read_data  test.data",
    "pair_style born/coul/wolf/cs 0.25 16.0 14.5",
    "pair_coeff	*	*	 0.0000	1.0000	0.000	0.0000	0.000",
    "pair_coeff	2	6	7149.81	0.3019	0.000	0.0000	0.000",
    "pair_coeff	4	6	7200.27	0.2303	0.000	0.0000	0.000",
    "pair_coeff	6	6	3719.60	0.3408	0.000	597.17	0.000",
    "pair_modify tail no",
    "bond_style class2",
    "bond_coeff	1	0.0	149.255	0.0		   0.0000000",
    "bond_coeff	2	0.0	153.070	0.0		 20.83333333",
    "bond_coeff	3	0.0	 18.465	0.0		208.33333333",
    "thermo_style custom step etotal pe ke fnorm ecoul elong ebond",
    "neigh_modify one 4000",
    "run 0",
    "variable ebond equal ebond",
    "variable ecoul equal ecoul",
    "variable pe equal pe",
    "print \"Bond Energy: ${ebond}\"",
    "print \"Coul Energy: ${ecoul}\"",
    "print \"Pot  Energy: ${pe}\""
]

lmp = lammps()

for each in cmds:
    lmp.command(each)

def self_energy_factor(alpha, rc):
    return math.erfc(alpha*rc)/2/rc + alpha / math.sqrt(math.pi)

print(self_energy_factor(0.25, 14.5))
sum_q2 = numpy.inner(barium_titanate_cs_charges, barium_titanate_cs_charges)

selfce = self_energy_factor(0.25, 14.5) * 14.399645 * sum_q2
print("Self Energy", selfce)

# Bond Energy: 11.312849999988
# Coul Energy: -106.234122015795
# Pot  Energy: -90.159494378776
# Self Energy 167.2257332820772

*/
