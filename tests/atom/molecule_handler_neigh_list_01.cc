#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/molecule_handler.h>
#include <deal.II-qc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;



// Short test to check MoleculeHandler::get_neighbor_lists() function.
// All atoms are interacting with each other.



template<int dim>
class TestMoleculeHandler
{
public:

  TestMoleculeHandler(const ConfigureQC &config)
    :
    config(config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    dof_handler    (triangulation)
  {}

  void run()
  {
    GridGenerator::hyper_cube( triangulation, 0., 16., true );
    triangulation.refine_global (3);

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (*config.get_stream(),
                                     dof_handler,
                                     config.get_ghost_cell_layer_thickness());

    cell_molecule_data.cell_energy_molecules =
      config.get_cluster_weights<dim>()->
      update_cluster_weights (dof_handler,
                              cell_molecule_data.cell_molecules);

    MoleculeHandler<dim> molecule_handler(config);

    auto neighbor_lists =
        molecule_handler.get_neighbor_lists (cell_molecule_data.cell_energy_molecules);

    for (const auto &entry : neighbor_lists)
      std::cout << "Atom I: "  << entry.second.first->second.atoms[0].global_index << " "
                << "Atom J: "  << entry.second.second->second.atoms[0].global_index << std::endl;
    std::cout << std::endl;
  }

private:
  const ConfigureQC &config;
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
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
          << "  set Maximum cutoff radius = 16.0"             << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 16.1"        << std::endl
          << "  set Cluster radius = 4.0"                     << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
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

      TestMoleculeHandler<3> problem (config);
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
