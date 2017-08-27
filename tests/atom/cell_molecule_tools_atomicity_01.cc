
#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_generator.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/sampling/cluster_weights_by_cell.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;



// Short test to compute the number of thrown and energy molecules per cell
// using CellMoleculeTools functions.

// The tria consists of only one cell with a total of 4 molecules
// each containing two atoms.
//
// 1 thrown molecules
// 3 energy molecule (= 6 energy atoms)



template<int dim, int atomicity>
class TestCellMoleculeTools
{
public:

  TestCellMoleculeTools(const ConfigureQC &config)
    :
    config(config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim, spacedim>::limit_level_difference_at_vertices,
                   config.get_ghost_cell_layer_thickness()),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    GridGenerator::hyper_cube (triangulation, 0., 8., true);
    triangulation.setup_ghost_cells();

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim, atomicity, spacedim> (*config.get_stream(),
                                                          triangulation);

    std::cout <<  "Masses: ";
    for (const auto &mass : cell_molecule_data.masses)
      std::cout << mass << "\t";
    std::cout << std::endl;

    std::shared_ptr<Cluster::WeightsByBase<dim, atomicity, spacedim> >
    cluster_weights_method = config.get_cluster_weights<dim, atomicity, spacedim>();

    cluster_weights_method->initialize (triangulation,
                                        QTrapez<dim>());

    cell_molecule_data.cell_energy_molecules =
      cluster_weights_method->
      update_cluster_weights (triangulation,
                              cell_molecule_data.cell_molecules);

    const auto &cell_molecules        = cell_molecule_data.cell_molecules;
    const auto &cell_energy_molecules = cell_molecule_data.cell_energy_molecules;

    for (auto
         entry  = cell_energy_molecules.begin();
         entry != cell_energy_molecules.end();
         entry  = cell_energy_molecules.upper_bound(entry->first))
      {

        const unsigned int n_molecules =
          CellMoleculeTools::molecules_range_in_cell<dim, atomicity, spacedim>
          (entry->first, cell_molecules).second;

        const unsigned int n_energy_molecules =
          CellMoleculeTools::molecules_range_in_cell<dim, atomicity, spacedim>
          (entry->first, cell_energy_molecules).second;

        std::cout << "Thrown molecules: "
                  << n_molecules-n_energy_molecules
                  << std::endl;

        std::cout << "Energy molecules: "
                  << cell_energy_molecules.size() << std::endl << std::endl;

        unsigned int count_atoms = 0, count_molecules = 0;
        for (const auto &cell_energy_molecule : cell_energy_molecules)
          {
            std::cout << "Energy molecule "
                      << count_molecules++ << ": "
                      << "Cluster weight: "
                      << cell_energy_molecule.second.cluster_weight << "\n";
            for (int atom_stamp = 0; atom_stamp < atomicity; ++atom_stamp)
              std::cout << "Atom "
                        << count_atoms++ << ": Position "
                        << cell_energy_molecule.second.atoms[atom_stamp].position
                        << std::endl;
            std::cout << std::endl;
          }

        AssertThrow(count_atoms==n_energy_molecules*atomicity,
                    ExcInternalError());

        AssertThrow(cell_molecules.count(entry->first) == n_molecules,
                    ExcInternalError());

        AssertThrow(cell_energy_molecules.count(entry->first) == n_energy_molecules,
                    ExcInternalError());
      }
  }

private:
  static const int spacedim = dim;
  const ConfigureQC &config;
  dealiiqc::parallel::shared::Triangulation<dim, spacedim> triangulation;
  MPI_Comm mpi_communicator;
  CellMoleculeData<dim, atomicity, spacedim> cell_molecule_data;

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
      oss << "set Dimension = 1"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 1.1"              << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = -1.9"         << std::endl
          << "  set Cluster radius = 1.1"                     << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "8 atoms"                   << std::endl         << std::endl
          << "2  atom types"             << std::endl         << std::endl
          << "Masses"                    << std::endl         << std::endl
          << "    1   0.7"               << std::endl         << std::endl
          << "    2   0.2"               << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 0. 0. 0."        << std::endl
          << "2 1 2 1.0 1. 0. 0."        << std::endl
          << "3 2 1 1.0 2. 0. 0."        << std::endl
          << "4 2 2 1.0 3. 0. 0."        << std::endl
          << "5 3 1 1.0 4. 0. 0."        << std::endl
          << "6 3 2 1.0 5. 0. 0."        << std::endl
          << "7 4 1 1.0 6. 0. 0."        << std::endl
          << "8 4 2 1.0 7. 0. 0."        << std::endl;


      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());


      ConfigureQC config( prm_stream );

      TestCellMoleculeTools<1, 2> problem (config);
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
