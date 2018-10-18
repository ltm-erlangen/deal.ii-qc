
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Test the correctness of derived class implementation of
// Cluster::WeightsByBase::update_cluster_weights() when a hanging node
// is present.
//
// +--+--+-----+
// |  |  |     |          +  - vertices
// +--+--+* *  |          *  - atoms
// |  | *| *   |
// +--+--+-----+
//
// Three energy atoms among which two are cluster atoms.
// Total 4 cell atoms.



template<int dim>
class Test
{
public:

  Test (const ConfigureQC &config)
    :
    config(config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices,
                   -1.),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back(2);
    for (int i = 1; i < dim; ++i)
      repetitions.push_back(1);

    Point<dim> p1, p2;

    for (int d = 0; d < dim; ++d)
      p2[d] = 1.;

    p2[0] = 2.;

    // +-----+-----+
    // |     |     |
    // |     |     |
    // |     |     |
    // +-----+-----+
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               p1,
                                               p2,
                                               true);

    triangulation.begin_active()->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();
    triangulation.setup_ghost_cells();

    // +--+--+-----+
    // |  |  |     |
    // +--+--+     |
    // |  |  |     |
    // +--+--+-----+
  }

  void run()
  {
    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (*config.get_stream(),
                                     triangulation,
                                     GridTools::Cache<dim>(triangulation));

    const auto &cell_molecules = cell_molecule_data.cell_molecules;

    std::cout << "Total number of cell atoms: "
              << cell_molecules.size()
              << std::endl;

    // --- WeightsBySamplingPoints
    {
      Cluster::WeightsBySamplingPoints<dim>
      weights_by_sampling_points (config.get_cluster_radius(),
                                  config.get_maximum_cutoff_radius());

      weights_by_sampling_points.initialize (triangulation,
                                             QTrapez<dim>());

      cell_molecule_data.cell_energy_molecules =
        weights_by_sampling_points.update_cluster_weights (triangulation,
                                                           cell_molecules);

      for (const auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
        std::cout << "::WeightsBySamplingPoints: "
                  << "Atom: " << cell_molecule.second.atoms[0].position << " : "
                  << "Cluster_Weight: "
                  << cell_molecule.second.cluster_weight << std::endl;
    }

    //  --- WeightsByCell
    {
      Cluster::WeightsByCell<dim>
      weights_by_cell (config.get_cluster_radius(),
                       config.get_maximum_cutoff_radius());

      weights_by_cell.initialize (triangulation,
                                  QTrapez<dim>());

      cell_molecule_data.cell_energy_molecules =
        weights_by_cell.update_cluster_weights (triangulation,
                                                cell_molecules);

      for (const auto &cell_molecule : cell_molecule_data.cell_energy_molecules)
        std::cout << "::WeightsByCell: "
                  << "Atom: " << cell_molecule.second.atoms[0].position << " "
                  << "Cluster_Weight: "
                  << cell_molecule.second.cluster_weight << std::endl;
    }

  }

private:
  const ConfigureQC &config;
  dealiiqc::parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;
  CellMoleculeData<dim> cell_molecule_data;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc,argv,
          dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss << "set Dimension = 2"                            << std::endl

          << "subsection Configure atoms"                   << std::endl
          << "  set Maximum cutoff radius = 0.2"            << std::endl
          << "  set Pair potential type = LJ"               << std::endl
          << "  set Pair global coefficients = 0.19 "       << std::endl
          << "end"                                          << std::endl

          << "subsection Configure QC"                      << std::endl
          << "  set Ghost cell layer thickness = 0.21"      << std::endl
          << "  set Cluster radius = 0.11"                  << std::endl
          << "  set Cluster weights by type = Cell"         << std::endl
          << "end"                                          << std::endl
          << "#end-of-parameter-section"                    << std::endl

          << "LAMMPS Description"        << std::endl       << std::endl
          << "4 atoms"                   << std::endl       << std::endl
          << "1  atom types"             << std::endl       << std::endl
          << "Atoms #"                   << std::endl       << std::endl
          << "1 1 1 1.0 0.95  0.45 0."   << std::endl
          << "2 2 1 1.0 1.05  0.45 0."   << std::endl
          << "3 3 1 1.0 1.10  0.45 0."   << std::endl
          << "4 4 1 1.0 1.50  0.50 0."   << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      Test<2> problem (config);
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
