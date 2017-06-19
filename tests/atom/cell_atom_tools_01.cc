
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/atom_handler.h>
#include <deal.II-qc/atom/cell_atom_tools.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>

using namespace dealii;
using namespace dealiiqc;

// This test is a close copy of atom_handler_thrown_atoms_01
// and the blessed output is exactly the same.
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
    GridGenerator::hyper_cube( triangulation, 0., 8., true );
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler, atom_data);
    const Cluster::WeightsByCell<dim>
    weights_by_cell (config.get_cluster_radius(),
                     config.get_maximum_cutoff_radius());
    atom_data.cell_energy_molecules =
      weights_by_cell.update_cluster_weights (dof_handler,
                                              atom_data.cell_molecules);
    for (auto
         entry  = atom_data.cell_energy_molecules.begin();
         entry != atom_data.cell_energy_molecules.end();
         entry  = atom_data.cell_energy_molecules.upper_bound(entry->first))
      {

        const unsigned int n_atoms =
          CellAtomTools::atoms_range_in_cell(entry->first,
                                             atom_data.cell_molecules).second;
        const unsigned int n_energy_atoms =
          CellAtomTools::atoms_range_in_cell(entry->first,
                                             atom_data.cell_energy_molecules).second;

        std::cout << "Cell: "
                  << entry->first
                  << " has "
                  << n_atoms-n_energy_atoms
                  << " thrown atoms."
                  << std::endl;

        std::cout << "Number of energy atoms: "
                  << atom_data.cell_energy_molecules.size() << std::endl;

        std::cout << std::endl;

        AssertThrow(atom_data.cell_molecules.count(entry->first) == n_atoms,
                    ExcInternalError());
        AssertThrow(atom_data.cell_energy_molecules.count(entry->first) == n_energy_atoms,
                    ExcInternalError())
      }
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
          << "  set Maximum cutoff radius = 1.1"              << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 1.9"         << std::endl
          << "  set Cluster radius = 1.1"                     << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "9 atoms"                   << std::endl         << std::endl
          << "1  atom types"             << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 2. 2. 2."        << std::endl
          << "2 2 1 1.0 6. 2. 2."        << std::endl
          << "3 3 1 1.0 2. 6. 2."        << std::endl
          << "4 4 1 1.0 2. 2. 6."        << std::endl
          << "5 5 1 1.0 6. 6. 2."        << std::endl
          << "6 6 1 1.0 6. 2. 6."        << std::endl
          << "7 7 1 1.0 2. 6. 6."        << std::endl
          << "8 8 1 1.0 6. 6. 6."        << std::endl
          << "9 9 1 1.0 7. 8. 7.9"       << std::endl;


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