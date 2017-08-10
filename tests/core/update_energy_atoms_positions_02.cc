
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>

#include <deal.II-qc/core/qc.h>

using namespace dealii;
using namespace dealiiqc;



// Compute the energy of the system of 3 charged atoms
// interacting exclusively through Coulomb interactions.
// Check that they the energy remains for a single configuration.
//
// +-------+-------+
// |       |       |   *, +  - vertices
// |       |       |   *     - atoms
// |       |       |
// *-------*-------*
//
// Update the positions of atoms



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
public:
  Problem (const ConfigureQC &);
  void partial_run ();
};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{
  QC<dim, PotentialType>::setup_cell_energy_molecules();
  QC<dim, PotentialType>::setup_system();
  QC<dim, PotentialType>::setup_fe_values_objects();
  QC<dim, PotentialType>::update_neighbor_lists();
}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::partial_run()
{
  // --- Energy at zero displacement.

  typename QC<dim, PotentialType>::vector_t u;
  u.reinit (QC<dim, PotentialType>::dof_handler.locally_owned_dofs(),
            QC<dim, PotentialType>::mpi_communicator);

  u = 0.;

  QC<dim, PotentialType>::locally_relevant_displacement = u;
  QC<dim, PotentialType>::update_positions();
  const double energy_0 =
    QC<dim, PotentialType>::template
    compute(QC<dim, PotentialType>::locally_relevant_gradient);

  // --- Random displacements.
  // manually get locally owned elements parallel vector:

  std::uniform_real_distribution<double> dist (0, 1.);
  std::default_random_engine engine;

  const IndexSet locally_owned_set =
    QC<dim, PotentialType>::dof_handler.locally_owned_dofs();

  for (unsigned int i = 0; i < locally_owned_set.n_elements(); ++i)
    u(locally_owned_set.nth_index_in_set(i)) = dist(engine);

  QC<dim, PotentialType>::locally_relevant_displacement = u;
  QC<dim, PotentialType>::update_positions();
  const double energy_1 =
    QC<dim, PotentialType>::template
    compute(QC<dim, PotentialType>::locally_relevant_gradient);
  (void)energy_1;

  // --- Reset displacement to zero.

  u = 0.;

  QC<dim, PotentialType>::locally_relevant_displacement = u;
  QC<dim, PotentialType>::update_positions();
  const double energy_2 =
    QC<dim, PotentialType>::template
    compute(QC<dim, PotentialType>::locally_relevant_gradient);

  AssertThrow (energy_0 == energy_2,
               ExcInternalError());

  QC<dim, PotentialType>::pcout << "TEST PASSED!" << std::endl;
}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 2;
      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = 1."                          << std::endl
          << "    set Y center = .5"                          << std::endl
          << "    set Z center = .5"                          << std::endl
          << "    set X extent = 2."                          << std::endl
          << "    set Y extent = 1."                          << std::endl
          << "    set Z extent = 1."                          << std::endl
          << "    set X repetitions = 2"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 9.0"              << std::endl
          << "  set Pair potential type = Coulomb Wolf"       << std::endl
          << "  set Pair global coefficients = 0.25, 8.25 "   << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 10.0"        << std::endl
          << "  set Cluster radius = 99.0"                    << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"              << std::endl   << std::endl
          << "3 atoms"                         << std::endl   << std::endl
          << "2  atom types"                   << std::endl   << std::endl
          << "Atoms #"                         << std::endl   << std::endl
          << "1 1 1  1.0 0.0 0.0 0."                          << std::endl
          << "2 2 2 -1.0 1.0 0.0 0."                          << std::endl
          << "3 3 1  1.0 2.0 0.0 0."                          << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      // Define Problem
      Problem<dim, Potential::PairCoulWolfManager> problem(config);
      problem.partial_run ();
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
