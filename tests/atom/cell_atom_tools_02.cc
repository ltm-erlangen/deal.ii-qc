
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/atom_handler.h>
#include <deal.II-qc/atom/cell_atom_tools.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>

using namespace dealii;
using namespace dealiiqc;

// This test is a close copy of atom_handler_thrown_atoms_02,
// the blessed output is exactly the same.
// Short test to compute the number of locally relevant thrown atoms.
// The maximum cutoff radius is large enough that none of the atoms are thrown.

template<int dim>
class TestAtomHandler : public AtomHandler<dim>
{
public:

  TestAtomHandler(const ConfigureQC &config)
    :
    AtomHandler<dim>( config),
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
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler, atom_data);
    const Cluster::WeightsByCell<dim>
    weights_by_cell (config.get_cluster_radius(),
                     config.get_maximum_cutoff_radius());
    atom_data.energy_atoms =
      weights_by_cell.update_cluster_weights (dof_handler,
                                              atom_data.atoms);
    for (auto
         entry  = atom_data.energy_atoms.begin();
         entry != atom_data.energy_atoms.end();
         entry  = atom_data.energy_atoms.upper_bound(entry->first))
      {

        const unsigned int n_atoms =
          CellAtomTools::atoms_range_in_cell(entry->first,
                                             atom_data.atoms).second;
        const unsigned int n_energy_atoms =
          CellAtomTools::atoms_range_in_cell(entry->first,
                                             atom_data.energy_atoms).second;
        std::cout << entry->first
                  << ":"
                  << n_atoms-n_energy_atoms
                  << std::endl;
        AssertThrow(atom_data.atoms.count(entry->first) == n_atoms,
                    ExcInternalError());
        AssertThrow(atom_data.energy_atoms.count(entry->first) == n_energy_atoms,
                    ExcInternalError())
      }
    std::cout << std::endl;
  }

private:
  const ConfigureQC &config;
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;
  AtomData<dim> atom_data;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
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
