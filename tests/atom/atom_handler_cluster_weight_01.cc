
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <dealiiqc/atom/cluster_weights.h>
#include <dealiiqc/atom/atom_handler.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to compute the number of locally relevant thrown atoms.
// The tria consists of only one cell
// 8 thrown atoms
// 2 cluster atom
// Cluster_Weight is 5 for cluster atoms.

template<int dim>
class TestAtomHandler : public AtomHandler<dim>
{
public:

  TestAtomHandler(const ConfigureQC &config)
    :
    AtomHandler<dim>( config),
    config(config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    GridGenerator::hyper_cube( triangulation, 0., 8., true );
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler);
    Cluster::WeightsByCell<dim> weights_by_cell(config);
    weights_by_cell.update_cluster_weights( AtomHandler<dim>::n_thrown_atoms_per_cell,
                                            AtomHandler<dim>::energy_atoms);
    for ( const auto &cell_atom : AtomHandler<dim>::energy_atoms )
      std::cout << "Atom: " << cell_atom.second.position << " "
                << "Cluster_Weight: "
                << cell_atom.second.cluster_weight << std::endl;
    std::cout << std::endl;
  }

private:
  const ConfigureQC &config;
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc,
          argv,
          numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum energy radius = 1.9"             << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Max search radius = 1.9"                  << std::endl
          << "  set Cluster radius = 1.9"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl << std::endl
          << "10 atoms"                  << std::endl << std::endl
          << "1  atom types"             << std::endl << std::endl
          << "Atoms #"                   << std::endl << std::endl
          << "1 1 1 1.0 2. 2. 2."      << std::endl
          << "2 2 1 1.0 6. 2. 2."      << std::endl
          << "3 3 1 1.0 2. 6. 2."      << std::endl
          << "4 4 1 1.0 2. 2. 6."      << std::endl
          << "5 5 1 1.0 6. 6. 2."      << std::endl
          << "6 6 1 1.0 6. 2. 6."      << std::endl
          << "7 7 1 1.0 2. 6. 6."      << std::endl
          << "8 8 1 1.0 6. 6. 6."      << std::endl
          << "9 9 1 1.0 7. 7. 7.9"     << std::endl
          << "10 10 1 1.0 1. 1. 0.1"   << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      ConfigureQC config( prm_stream );

      TestAtomHandler<3> problem (config);
      problem.run();

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