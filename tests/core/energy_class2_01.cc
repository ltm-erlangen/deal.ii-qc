
#include <deal.II-qc/core/compute_tools.h>
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of of 2 atoms interacting exclusively
// through Class2 using QC approach with full atomistic resolution.
// The blessed output is created using LAMMPS python script.
// The script is included at the end.



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

  const auto &cell_molecules = this->cell_molecule_data.cell_energy_molecules;

  deallog << "picked up: " << cell_molecules.size() << " energy molecule(s)."
          << std::endl;

  const std::shared_ptr<const PotentialType> potential_ptr =
    std::const_pointer_cast<const PotentialType>(
      std::static_pointer_cast<PotentialType>(
        this->configure_qc.get_potential()));

  const double intra_molecular_energy =
    ComputeTools::energy_and_gradient<PotentialType, dim, atomicity, false>(
      *potential_ptr,
      cell_molecules.begin()->second,
      this->cell_molecule_data.bonds)
      .first;

  const double energy =
    this->template compute<false>(this->locally_relevant_gradient);

  this->pcout << "The energy computed using PairClass2Manager "
              << "of atomistic system is: " << std::fixed
              << std::setprecision(5) << energy << " eV." << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(this->neighbor_lists.size(),
                                this->mpi_communicator);

  this->pcout << "Total number of neighbors " << total_n_neighbors << std::endl;

  for (auto entry : this->neighbor_lists)
    std::cout << "Molecule I: " << entry.second.first->second.global_index
              << '\t'
              << "Molecule J: " << entry.second.second->second.global_index
              << std::endl;

  AssertThrow(std::fabs(energy - intra_molecular_energy) <
                200 * std::numeric_limits<double>::epsilon(),
              ExcInternalError());

  // Accurate to 1e-9 // TODO Check unit and conversions
  AssertThrow(std::fabs(energy - blessed_energy) <
                1e7 * std::numeric_limits<double>::epsilon(),
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
          << "    set X center = 1." << std::endl
          << "    set Y center = 1." << std::endl
          << "    set Z center = 1." << std::endl
          << "    set X extent = 2." << std::endl
          << "    set Y extent = 2." << std::endl
          << "    set Z extent = 2." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "    set Z repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Number of atom types = 2" << std::endl
          << "  set Maximum cutoff radius = 100" << std::endl
          << "  set Pair potential type = Class2" << std::endl
          << "  set Bond type = Class2" << std::endl
          << "  set Bond specific coefficients = *, *, 0.12, 2.1, 3.12, 3.7"
          << std::endl
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
          << "1 1 1  0.0 0.00 0. 0." << std::endl
          << "2 1 2  0.0 0.25 0. 0." << std::endl
          << std::endl
          << "Bonds #" << std::endl
          << "1 1 1 2" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config(prm_stream);

      // Define Problem
      Problem<dim, Potential::PairClass2Manager, 2> problem(config);
      problem.partial_run(0.043401397 /*blessed energy from LAMMPS*/);
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
LAMMPS input script for energy_class2_01 test
"""
from lammps import lammps

with open("test.data", 'w') as f:
    f.write(
"""LAMMPS Description

2 atoms
1 bonds

2  atom types
1  bond types

0.0 0.5  xlo xhi
0.0 0.5  ylo yhi
0.0 0.5  zlo zhi

Masses

  1 1.
  2 1.

Atoms

1 1 1  0.0 0.00 0. 0.
2 1 2  0.0 0.25 0. 0.

Bonds

1 1 1 2
""")

cmds = ["units metal",
        "atom_style full",
        "read_data test.data",
        "pair_style lj/cut 1",
        "pair_coeff * * 0 .25",
        "bond_style class2",
        "bond_coeff 1 0.12 2.1 3.12 3.7",
        "run 0",
        "variable energy equal emol",
        "variable energy_1 equal ${energy}",
        "print       \"Energy: ${energy_1}\""
        ]

lmp = lammps()

for each in cmds:
    lmp.command(each)

# Result: 0.043401397
*/
