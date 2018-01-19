#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/base/quadrature_lib.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Test WeightsByOptimalSummationRules::update_cluster_weights().
// Consider a simple hyper_cube with 8 atoms for computation (single process).
// All the atoms should be picked up as vertex-type sampling points and receive
// cluster weights of 1.



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

    std::shared_ptr<Cluster::WeightsByBase<dim> > cluster_weights_method =
      config.get_cluster_weights<dim>();

    cluster_weights_method->initialize (triangulation,
                                        QTrapezWithMidpoint<dim>());

    cell_molecule_data.cell_energy_molecules =
      cluster_weights_method->
      update_cluster_weights (triangulation,
                              cell_molecule_data.cell_molecules);

    // Check that the number of molecules picked up.
    std::cout << "The number of molecules picked up: "
              << cell_molecule_data.cell_molecules.size()
              << std::endl;

    for (const auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
      std::cout << "Molecule ID: " << cell_molecule.second.global_index << "\t"
                << "Weight: "      << cell_molecule.second.cluster_weight
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
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 9.01"        << std::endl
          << "  set Cluster radius = 0"                       << std::endl
          << "  set Cluster weights by type = OptimalSummationRules" << std::endl
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
