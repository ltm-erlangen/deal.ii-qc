
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>
#include <deal.II-qc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;



// Short test to compute the number of thrown molecules per cell using
// CellMoleculeTools functions.
//
// The maximum cutoff radius is large enough that none of the molecules are
// thrown.
// The tria is refined three times globally, contains 512 cells and none of
// the cells have thrown atoms.



template<int dim>
class TestCellMoleculeTools
{
public:

  TestCellMoleculeTools(const ConfigureQC &config)
    :
    config(config),
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

    const std::string atom_data_file = config.get_atom_data_file();
    std::fstream fin(atom_data_file, std::fstream::in );

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (fin,
                                     dof_handler,
                                     config.get_ghost_cell_layer_thickness());

    cell_molecule_data.cell_energy_molecules =
      config.get_cluster_weights<dim>()->
      update_cluster_weights (dof_handler,
                              cell_molecule_data.cell_molecules);

    const auto &cell_molecules        = cell_molecule_data.cell_molecules;
    const auto &cell_energy_molecules = cell_molecule_data.cell_energy_molecules;

    for (auto
         entry  = cell_energy_molecules.begin();
         entry != cell_energy_molecules.end();
         entry  = cell_energy_molecules.upper_bound(entry->first))
      {

        const unsigned int n_molecules =
          CellMoleculeTools::molecules_range_in_cell<dim> (entry->first,
                                                           cell_molecules).second;
        const unsigned int n_energy_molecules =
          CellMoleculeTools::molecules_range_in_cell<dim> (entry->first,
                                                           cell_energy_molecules).second;
        std::cout << entry->first
                  << ":"
                  << n_molecules-n_energy_molecules
                  << std::endl;
        AssertThrow(cell_molecules.count(entry->first) == n_molecules,
                    ExcInternalError());
        AssertThrow(cell_energy_molecules.count(entry->first) == n_energy_molecules,
                    ExcInternalError())
      }
    std::cout << std::endl;
  }

private:
  const ConfigureQC &config;
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;
  CellMoleculeData<dim> cell_molecule_data;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      std::ostringstream oss;
      oss << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/16_NaCl_atom.data"          << std::endl
          << "  set Maximum cutoff radius = 16.0"             << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 16.1"        << std::endl
          << "  set Cluster radius = 4.0"                     << std::endl
          << "end"                                            << std::endl;


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
