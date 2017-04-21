#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <dealiiqc/atom/atom_handler.h>
#include <dealiiqc/io/data_out_atom_data.h>
#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/utilities.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to check parse_atoms_and_assign_to_cells() function of AtomHandler
// doesn't throw any errors.

template<int dim>
class TestAtomHandler : public AtomHandler<dim>
{
public:

  TestAtomHandler(const ConfigureQC &config)
    :
    AtomHandler<dim>( config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    GridGenerator::hyper_cube( triangulation, 0., 16., true );
    triangulation.refine_global (3);
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler, atom_data);
    atom_data.neighbor_lists =
    AtomHandler<dim>::get_neighbor_lists( atom_data.energy_atoms);
    for ( auto entry : atom_data.neighbor_lists)
      std::cout << "Atom I: "  << entry.second.first->second.global_index << " "
                << "Atom J: "  << entry.second.second->second.global_index << std::endl;
    std::cout << std::endl;
  }

private:
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;
  AtomData<dim> atom_data;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc,
          argv,
          numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum energy radius = 16.0"             << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Max search radius = 2.0"                  << std::endl
          << "  set Cluster radius = 4.0"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl << std::endl
          << "8 atoms"                   << std::endl << std::endl
          << "1  atom types"             << std::endl << std::endl
          << "Atoms #"                   << std::endl << std::endl
          << "1 1 1 1.0 2. 2. 2."      << std::endl
          << "2 2 1 1.0 6. 2. 2."      << std::endl
          << "3 3 1 1.0 2. 6. 2."      << std::endl
          << "4 4 1 1.0 2. 2. 6."      << std::endl
          << "5 5 1 1.0 6. 6. 2."      << std::endl
          << "6 6 1 1.0 6. 2. 6."      << std::endl
          << "7 7 1 1.0 2. 6. 6."      << std::endl
          << "8 8 1 1.0 6. 6. 6."      << std::endl;


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
