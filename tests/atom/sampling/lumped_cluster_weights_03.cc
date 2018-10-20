
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/molecule_handler.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>

#include <deal.II-qc/grid/shared_tria.h>

#include <iostream>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;

// A test case for cluser weights by lumped vertices
// The tria consists of only two cells
// 3 total atoms
// 2 cluster atom
// Blessed output created by hand calculation

//   ___________________
//   |       *|*       |
//   |        |  *     |
//   |________|________|
//

template <int dim>
class Test
{
public:
  Test(const ConfigureQC &config)
    : config(config)
    , triangulation(
        MPI_COMM_WORLD,
        // guarantee that the mesh also does not change by more than refinement
        // level across vertices that might connect two cells:
        Triangulation<dim>::limit_level_difference_at_vertices,
        -1.)
    , mpi_communicator(MPI_COMM_WORLD)
  {}

  void
  run()
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back(2);
    for (int i = 1; i < dim; ++i)
      repetitions.push_back(1);

    const dealii::Point<dim> p1(0., 0., 0.);
    const dealii::Point<dim> p2(2., 1., 1.);

    //   ___________________
    //   |        |        |
    //   |        |        |
    //   |________|________|
    //
    GridGenerator::subdivided_hyper_rectangle(
      triangulation, repetitions, p1, p2, true);
    triangulation.setup_ghost_cells();

    cell_molecule_data =
      CellMoleculeTools::build_cell_molecule_data<dim>(*config.get_stream(),
                                                       triangulation,
                                                       GridTools::Cache<dim>(
                                                         triangulation));

    Cluster::WeightsByLumpedVertex<dim> weights_by_lumped_vertex(
      config.get_cluster_radius(), config.get_maximum_cutoff_radius());

    weights_by_lumped_vertex.initialize(triangulation, QTrapez<dim>());

    const auto &cell_molecules = cell_molecule_data.cell_molecules;

    cell_molecule_data.cell_energy_molecules =
      weights_by_lumped_vertex.update_cluster_weights(triangulation,
                                                      cell_molecules);

    unsigned int this_mpi_process =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    dealii::ConditionalOStream pcout(std::cout, this_mpi_process == 0);

    for (const auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
      if (this_mpi_process == 0)
        pcout << "Atom: " << cell_molecule.second.atoms[0].position << " : "
              << "Cluster_Weight: " << cell_molecule.second.cluster_weight
              << std::endl;
  }

private:
  const ConfigureQC &                            config;
  dealiiqc::parallel::shared::Triangulation<dim> triangulation;
  MPI_Comm                                       mpi_communicator;
  CellMoleculeData<dim>                          cell_molecule_data;
};


int
main(int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss << "set Dimension = 3" << std::endl
          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 5" << std::endl
          << "end" << std::endl
          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 5.1" << std::endl
          << "  set Cluster radius = 0.49" << std::endl
          << "  set Cluster weights by type = Cell" << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description" << std::endl
          << std::endl
          << "3 atoms" << std::endl
          << std::endl
          << "1  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1 1.0 0.9 0.9 0.9" << std::endl
          << "2 2 1 1.0 1.1 0.9 0.9" << std::endl
          << "3 3 1 1.0 1.5 0.5 0.5" << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      ConfigureQC config(prm_stream);

      Test<3> problem(config);
      problem.run();
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
