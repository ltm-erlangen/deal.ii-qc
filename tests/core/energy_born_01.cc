
#include <deal.II-qc/core/qc.h>

#include "../tests.h"

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of nano-crystal of 8 charged atoms
// interacting with Born pair potential with full atomistic resolution.
// The blessed output is created through the LAMMPS input script included at
// the end.



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

  this->pcout << "The energy computed using PairBornCutManager "
              << "of charged atomistic system is: " << energy << " eV."
              << std::endl;

  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(this->neighbor_lists.size(),
                                this->mpi_communicator);

  this->pcout << "Total number of neighbors " << total_n_neighbors << std::endl;

  for (auto entry : this->neighbor_lists)
    std::cout << "Molecule I: " << entry.second.first->second.global_index
              << '\t'
              << "Molecule J: " << entry.second.second->second.global_index
              << std::endl;

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
      oss
        << "set Dimension = " << dim << std::endl

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
        << "  set Pair potential type = Born" << std::endl
        << "  set Pair global coefficients = 2.5" << std::endl
        << "  set Pair specific coefficients = *, *, 0.877, 1., 1.55, 0.877, 0"
        << std::endl
        << "  set Atom data file = "
        << SOURCE_DIR "/../data/NaCl_1x1x1_molecule.data" << std::endl
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
      Problem<dim, Potential::PairBornCutManager, 8> problem(config);
      problem.partial_run(21.2500353330642 /*blessed energy from LAMMPS*/);
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
LAMMPS input script for energy_born_01 test
"""
from lammps import lammps
lmp = lammps()
lmp.command("units           metal")
lmp.command("dimension       3")
lmp.command("boundary        s s s")
lmp.command("atom_style      charge")
lmp.command("read_data NaCl_1x1x1_atom.data")
lmp.command("thermo_style custom step epair evdwl ecoul elong fnorm fmax")
lmp.command("thermo_modify format 3 %20.16g")
lmp.command("thermo_modify format 7 %20.16g")
lmp.command("pair_style born 2.5")
lmp.command("pair_coeff * * 0.877 1. 1.55 0.877 0")
lmp.command("run 0")
lmp.command("variable energy equal epair")
lmp.command("variable energy_1 equal ${energy}")
lmp.command("print       \"Energy: ${energy_1}\"")

# lammps_result = 21.2500353330642
*/
