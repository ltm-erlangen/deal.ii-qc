#include "../../tests.h"

#include <iostream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/base/quadrature_lib.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Test WeightsByOptimalSummationRules::update_cluster_weights().
//
//
// x---x---x-------x
// |   |   |       |          x  - vertices/atoms
// x---x---x       |
// |   |   |       |
// x---x---x--- ---x
//
// All the atoms should be picked up as vertex-type sampling points and receive
// cluster weights of 1.
// Molecule 10 is an inner-element sampling point.



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
    const std::vector<unsigned int> repetitions = {2, 1};
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               Point<dim>(0., 0.),
                                               Point<dim>(4., 2.));
    triangulation.begin_active()->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
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

    MPI_Barrier(MPI_COMM_WORLD);

    Testing::SequentialFileStream write_sequentially(MPI_COMM_WORLD);

    for (const auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
      if (cell_molecule.first->is_locally_owned())
        deallog << "Molecule ID: " << cell_molecule.second.global_index << "\t"
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
          << "set Dimension = 2"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 1.01"             << std::endl
          << "  set Pair global coefficients = .1"            << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 9.01"        << std::endl
          << "  set Cluster radius = 0"                       << std::endl
          << "  set Cluster weights by type = OptimalSummationRules" << std::endl
          << "  set Representative distance = .5"            << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "12 atoms"                   << std::endl        << std::endl
          << "1  atom types"             << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 0. 0. 0."        << std::endl
          << "2 2 1 1.0 1. 0. 0."        << std::endl
          << "3 3 1 1.0 0. 1. 0."        << std::endl
          << "4 4 1 1.0 1. 1. 0."        << std::endl
          << "5 5 1 1.0 2. 0. 0."        << std::endl
          << "6 6 1 1.0 0. 2. 0."        << std::endl
          << "7 7 1 1.0 2. 1. 0."        << std::endl
          << "8 8 1 1.0 1. 2. 0."        << std::endl
          << "9 9 1 1.0 2. 2. 0."        << std::endl
          << "10 10 1 1.0 4. 0. 0."      << std::endl
          << "11 11 1 1.0 3. 1. 0."      << std::endl
          << "12 12 1 1.0 4. 2. 0."      << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      TestCellMoleculeTools<2> problem (config);
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
