#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Test CellMolecueTools::compute_molecule_density().
// Consider a simple hyper_cube with 8 atoms for computation (single process).



template<int dim>
class TestCellMoleculeTools
{
public:

  TestCellMoleculeTools(const ConfigureQC &config)
    :
    config(config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices,
                   -1.)
  {}

  void run()
  {
    GridGenerator::hyper_cube (triangulation, 0., 1., true );
    triangulation.setup_ghost_cells();

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (*config.get_stream(),
                                     triangulation);

    const double molecule_density =
      CellMoleculeTools::
      compute_molecule_density (triangulation,
                                cell_molecule_data.cell_molecules);

    // Check that the number of molecules picked up.
    std::cout << "The number of molecules picked up: "
              << cell_molecule_data.cell_molecules.size()
              << std::endl;

    // Stream out computed density.
    std::cout << "Molecule density: "
              << molecule_density
              << std::endl;
  }

private:
  const ConfigureQC &config;
  dealiiqc::parallel::shared::Triangulation<dim> triangulation;
  CellMoleculeData<dim> cell_molecule_data;
};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
          dealii::numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss
          << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 1.01"             << std::endl
          << "  set Pair global coefficients = .1"            << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "8 atoms"                   << std::endl         << std::endl
          << "1  atom types"             << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 1. 0. 0."        << std::endl
          << "2 2 1 1.0 0. 1. 0."        << std::endl
          << "3 3 1 1.0 0. 0. 1."        << std::endl
          << "4 4 1 1.0 1. 1. 0."        << std::endl
          << "5 5 1 1.0 1. 0. 1."        << std::endl
          << "6 6 1 1.0 0. 1. 1."        << std::endl
          << "7 7 1 1.0 1. 1. 1."        << std::endl
          << "8 8 1 1.0 0. 0. 0."        << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      ConfigureQC config( prm_stream );

      TestCellMoleculeTools<3> problem (config);
      problem.run();

      std::cout << "OK" << std::endl;

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
