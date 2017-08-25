#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Short test to check functions of CellMoleculeTools doesn't throw any errors.



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
                   -1.),
    dof_handler    (triangulation)
  {}

  void run()
  {
    GridGenerator::hyper_cube (triangulation, 0., 8., true );
    triangulation.refine_global (1);
    triangulation.setup_ghost_cells();

    const std::string atom_data_file = config.get_atom_data_file();
    std::fstream fin(atom_data_file, std::fstream::in );

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (fin,
                                     triangulation);

    std::shared_ptr<Cluster::WeightsByBase<dim> > cluster_weights_method =
      config.get_cluster_weights<dim>();

    cluster_weights_method->initialize (triangulation,
                                        QTrapez<dim>());

    cell_molecule_data.cell_energy_molecules =
      cluster_weights_method->
      update_cluster_weights (triangulation,
                              cell_molecule_data.cell_molecules);

    // Check that the number of energy molecules picked up.
    // Since atomicity is 1, energy molecules are energy atoms.
    // is so and so and shouldn't change each time this test is run.
    std::cout << "The number of energy atoms picked up : "
              << cell_molecule_data.cell_energy_molecules.size()
              << std::endl;
  }

private:
  const ConfigureQC &config;
  dealiiqc::parallel::shared::Triangulation<dim> triangulation;
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
          << "  set Maximum cutoff radius = 0.01"             << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/8_NaCl_atom.data"           << std::endl
          << "  set Pair global coefficients = .001"          << std::endl
          << "end" << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Cluster radius = 1.99"                    << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end"                                            << std::endl;

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
