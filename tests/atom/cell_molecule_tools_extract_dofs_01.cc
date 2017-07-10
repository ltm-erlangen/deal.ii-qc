
#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;

// #define WRITE_GRID



// Test to check correctness of
// CellMoleculeTools::extract_locally_relevant_dofs().
// The tria consists of 8 cell.



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
    fe(FE_Q<dim>(1), dim),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    std::vector<unsigned int> repetitions;
    repetitions.push_back(8);
    for (int i = 1; i < dim; ++i)
      repetitions.push_back(1);

    dealii::Point<dim> p1, p2;

    for (int d = 0; d < dim; ++d)
      p2[d] = 1.;

    p2[0] = 8.;

    //   .___.___.___.___.___.___.___.___.
    //   |   |   |   |   |   |   |   |   |        . atom
    //   |___|___|___|___|___|___|___|___|
    //
    GridGenerator::subdivided_hyper_rectangle (triangulation,
                                               repetitions,
                                               p1,
                                               p2,
                                               true);
    std::ofstream g("tria.vtk");
    GridOut().write_vtk (triangulation, g);

    dof_handler.distribute_dofs (fe);

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (*config.get_stream(),
                                     dof_handler,
                                     config.get_ghost_cell_layer_thickness());

    const IndexSet locally_relevant_set =
      CellMoleculeTools::
      extract_locally_relevant_dofs (dof_handler,
                                     cell_molecule_data.cell_molecules);

    unsigned int
    n_mpi_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator),
    this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);


    for (unsigned int p = 0; p < n_mpi_processes; p++)
      {
        MPI_Barrier(mpi_communicator);
        if (p == this_mpi_process)
          {
            std::cout << "Process "
                      << p
                      << " has "
                      << locally_relevant_set.n_elements()
                      << " locally relevant dofs."
                      << std::endl;
          }
        MPI_Barrier(mpi_communicator);
      }

#ifdef WRITE_GRID
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
      {
        std::map<dealii::types::global_dof_index, Point<dim> > support_points;
        DoFTools::map_dofs_to_support_points (MappingQ1<dim>(),
                                              dof_handler,
                                              support_points);

        const std::string filename =
          "grid" + dealii::Utilities::int_to_string(dim) + ".gp";
        std::ofstream f(filename.c_str());

        f << "set terminal png size 400,410 enhanced font \"Helvetica,8\""
          << std::endl
          << "set output \"grid"  << dealii::Utilities::int_to_string(dim)
          << ".png\""             << std::endl
          << "set size square"    << std::endl
          << "set view equal xy"  << std::endl
          << "unset xtics"        << std::endl
          << "unset ytics"        << std::endl
          << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
        GridOut().write_gnuplot (triangulation, f);
        f << "e" << std::endl;

        DoFTools::write_gnuplot_dof_support_point_info(f,
                                                       support_points);
        f << "e" << std::endl;
      }
#endif
  }

private:
  const ConfigureQC &config;
  parallel::shared::Triangulation<dim> triangulation;
  FESystem<dim>        fe;
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
      oss << "set Dimension = 2"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 0.9"              << std::endl
          << "  set Pair potential type = LJ"                 << std::endl
          << "  set Pair global coefficients = 0.89 "         << std::endl
          << "end"                                            << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 0.99"        << std::endl
          << "  set Cluster radius = 0.1"                     << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "9 atoms"                   << std::endl         << std::endl
          << "1  atom types"             << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 0. 1. 0."        << std::endl
          << "2 2 1 1.0 1. 1. 0."        << std::endl
          << "3 3 1 1.0 2. 1. 0."        << std::endl
          << "4 4 1 1.0 3. 1. 0."        << std::endl
          << "5 5 1 1.0 4. 1. 0."        << std::endl
          << "6 6 1 1.0 5. 1. 0."        << std::endl
          << "7 7 1 1.0 6. 1. 0."        << std::endl
          << "8 8 1 1.0 7. 1. 0."        << std::endl
          << "9 9 1 1.0 8. 1. 0."        << std::endl;

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
