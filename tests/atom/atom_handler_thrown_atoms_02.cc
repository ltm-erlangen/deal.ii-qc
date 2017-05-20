
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/atom_handler.h>
#include <deal.II-qc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to compute the number of locally relevant thrown atoms.
// The max energy radius is large enough that none of the atoms are thrown.

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
    for ( const auto &entry : atom_data.n_thrown_atoms_per_cell)
      std::cout << entry.first << ":" << entry.second << std::endl;
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
      oss << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/16_NaCl_atom.data"          << std::endl
          << "  set Maximum energy radius = 16.0"             << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2."          << std::endl
          << "  set Cluster radius = 4.0"                     << std::endl
          << "end"                                            << std::endl;


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
