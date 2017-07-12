#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/data_out_atom_data.h>
#include <deal.II-qc/configure/configure_qc.h>

using namespace dealii;
using namespace dealiiqc;


// Short test to check write_vtp() function of DataOutAtomData.
// This test makes sure that the vtp files are legal even if an MPI process
// doesn't have any locally owned atoms.

// Comment below line to skip writing atom data in vtp file format
#define WRITE_ATOM_DATA


template<int dim>
class TestDataOutAtomData
{
public:

  TestDataOutAtomData(const ConfigureQC &config)
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
    {
      Point<dim> bl(0,0,0);
      Point<dim> tr(8,8,8);
      // create a mesh which is not aligned with crystal structure
      std::vector<unsigned int> repetitions(dim,7);
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                bl,
                                                tr);
    }

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (*config.get_stream(),
                                     dof_handler,
                                     config.get_ghost_cell_layer_thickness());

    for (auto &cell_molecule : cell_molecule_data.cell_molecules)
      cell_molecule.second.cluster_weight = 0.;

    write_output();
  }

  void write_output()
  {

    unsigned int n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
             this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator));

    dealii::DataOutBase::VtkFlags flags( std::numeric_limits<double>::min(),
                                         std::numeric_limits<unsigned int>::min(),
                                         false);

    DataOutAtomData atom_data_out;

#ifdef WRITE_ATOM_DATA
    std::string vtp_file_name = "atoms-" + dealii::Utilities::int_to_string(this_mpi_process,3) + ".vtp";

    std::ofstream vtp_file;
    vtp_file.open (vtp_file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

    atom_data_out.write_vtp<3> (cell_molecule_data.cell_molecules,
                                flags,
                                vtp_file);
    vtp_file.close();
    if ( this_mpi_process == 0 )
      {
        std::vector<std::string> vtp_file_names;
        for (unsigned int i=0; i<n_mpi_processes; ++i)
          vtp_file_names.push_back ("atoms-" + dealii::Utilities::int_to_string(i,3) + ".vtp");
        std::ofstream pvtp_file;
        pvtp_file.open("atoms-.pvtp", std::ofstream::out | std::ofstream::trunc);
        atom_data_out.write_pvtp_record( vtp_file_names, flags, pvtp_file );
        pvtp_file.close();

        GridOut gridout;
        std::ofstream f("tria.vtk");
        gridout.write_vtk(triangulation, f);
      }
#endif

    for (unsigned int p = 0; p < n_mpi_processes; p++)
      {
        MPI_Barrier(mpi_communicator);
        if (p == this_mpi_process)
          atom_data_out.write_vtp<3> (cell_molecule_data.cell_molecules,
                                      flags,
                                      std::cout);
      }
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
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv,
          dealii::numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 1.01"             << std::endl
          << "end" << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 1.1"         << std::endl
          << "  set Cluster radius = 0.7"                     << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end" << std::endl
          << "#end-of-parameter-section" << std::endl
          << "LAMMPS Description"        << std::endl         << std::endl
          << "7 atoms"                   << std::endl         << std::endl
          << "1  atom types"             << std::endl         << std::endl
          << "Atoms #"                   << std::endl         << std::endl
          << "1 1 1 1.0 2. 2. 2."        << std::endl
          << "2 2 1 1.0 6. 2. 2."        << std::endl
          << "3 3 1 1.0 2. 6. 2."        << std::endl
          << "4 4 1 1.0 2. 2. 6."        << std::endl
          << "5 5 1 1.0 6. 6. 2."        << std::endl
          << "6 6 1 1.0 6. 2. 6."        << std::endl
          << "7 7 1 1.0 2. 6. 6."        << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      ConfigureQC config( prm_stream );

      TestDataOutAtomData<3> problem (config);
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
