#include <iostream>
#include <sstream>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/grid_generator.h>

#include <dealiiqc/atom/atom_handler.h>
#include <dealiiqc/io/data_out_atom_data.h>
#include <dealiiqc/io/configure_qc.h>
#include <dealiiqc/utilities.h>

using namespace dealii;
using namespace dealiiqc;

// Short test to check write_vtp() function of DataOutAtomData

template<int dim>
class TestAtomHandler : public AtomHandler<dim>
{
public:

  TestAtomHandler(const ConfigureQC &config)
    :
    AtomHandler<dim>( config),
    triangulation (MPI_COMM_WORLD,
                   // guarantee that the mesh also does not change by more than refinement level across vertices that might connect two cells:
                   Triangulation<dim>::limit_level_difference_at_vertices),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    GridGenerator::hyper_cube( triangulation, 0., 16., true );
    triangulation.refine_global (2);
    AtomHandler<dim>::parse_atoms_and_assign_to_cells( dof_handler);
    write_output();
  }

  void write_output()
  {
    DataOutAtomData<dim> atom_data_out;
    unsigned int n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator)),
             this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator));

    std::string vtp_file_name = "atoms-" + dealii::Utilities::int_to_string(this_mpi_process,3) + ".vtp";

    std::ofstream vtp_file;
    vtp_file.open (vtp_file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

    dealii::DataOutBase::VtkFlags flags;
    atom_data_out.write_vtp( AtomHandler<dim>::atoms,
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

        std::cout << "OK" << std::endl;
      }

  }

private:
  parallel::shared::Triangulation<dim> triangulation;
  DoFHandler<dim>      dof_handler;
  MPI_Comm mpi_communicator;

};


int main (int argc, char **argv)
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization( argc,
          argv,
          numbers::invalid_unsigned_int);
      std::ostringstream oss;
      oss << "set Dimension = 3"                              << std::endl
          << "subsection Configure atoms"                     << std::endl
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/16_NaCl_atom.data"          << std::endl
          << "end" << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Max search radius = 1.9"                  << std::endl
          << "end" << std::endl;

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
