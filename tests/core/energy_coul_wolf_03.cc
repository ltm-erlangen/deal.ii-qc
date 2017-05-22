
#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of NaCl nano-crystal of 512 charged atoms
// interacting exclusively through Coulomb interactions using QC approach with
// full atomistic resolution.
// The blessed output is created through the LAMMPS input script included at
// the end.



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const ConfigureQC &);
  void partial_run (const double &blessed_energy);
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run(const double &blessed_energy)
{

  unsigned int n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(QC<dim, PotentialType>::mpi_communicator)),
           this_mpi_process(dealii::Utilities::MPI::this_mpi_process(QC<dim, PotentialType>::mpi_communicator));

  for (unsigned int p = 0; p < n_mpi_processes; p++)
    {
      MPI_Barrier(QC<dim, PotentialType>::mpi_communicator);
      if (p == this_mpi_process)
        std::cout << "Process: " << p
                  << " picked up : "
                  << QC<dim, PotentialType>::atom_data.energy_atoms.size()
                  << " number of energy atoms."
                  << std::endl;
    }
  MPI_Barrier(QC<dim, PotentialType>::mpi_communicator);

  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();

  const double energy = QC<dim, PotentialType>::template
                        calculate_energy_gradient<false> (QC<dim, PotentialType>::gradient);

  QC<dim, PotentialType>::pcout
      << "The energy computed using PairCoulWolfManager "
      <<    "of 2 charged atom system is: "
      << energy
      << " eV."
      << std::endl;


  const unsigned int total_n_neighbors =
    dealii::Utilities::MPI::sum(QC<dim, PotentialType>::neighbor_lists.size(),
                                QC<dim, PotentialType>::mpi_communicator);

  QC<dim, PotentialType>::pcout
      << "Total number of neighbors "
      << total_n_neighbors
      << std::endl;

  // Accurate to 1e-9 // TODO Check unit and conversions
  AssertThrow (std::fabs(energy-blessed_energy) < 1e7 * std::numeric_limits<double>::epsilon(),
               ExcInternalError());

}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 3;
      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = 4."                          << std::endl
          << "    set Y center = 4."                          << std::endl
          << "    set Z center = 4."                          << std::endl
          << "    set X extent = 8."                          << std::endl
          << "    set Y extent = 8."                          << std::endl
          << "    set Z extent = 8."                          << std::endl
          << "    set X repetitions = 1"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 2" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 100"              << std::endl
          << "  set Pair potential type = Coulomb Wolf"       << std::endl
          << "  set Pair global coefficients = 0.4, 1.5"      << std::endl
          << "  set Atom data file = " << SOURCE_DIR "/../data/8_NaCl_atom.data"
          << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 100.1"       << std::endl
          << "  set Cluster radius = 100"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
      problem.partial_run (-4748.564251019102/*blessed energy from LAMMPS*/);

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
 <
  # LAMMPS input script for energy_coul_wolf_03 test

  units           metal
  dimension       3
  boundary        s s s

  atom_style      full

  read_data 8_NaCl_atom.data

  thermo_style custom step epair evdwl ecoul elong fnorm fmax
  thermo_modify format 3 %20.16g
  thermo_modify format 7 %20.16g

  velocity all create 0.0000000000001 146981634 dist gaussian mom yes rot no

  # first call
  pair_style coul/wolf 0.4 1.5
  pair_coeff * *
  run 0
  variable energy equal epair
  variable energy_1 equal ${energy}

  print       "Energy: ${energy_1}"
 >
*/
