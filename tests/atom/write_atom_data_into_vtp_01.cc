
#include "../tests.h"

#include <iostream>
#include <sstream>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/atom/data_out_atom_data.h>
#include <deal.II-qc/configure/configure_qc.h>
#include <deal.II-qc/grid/shared_tria.h>

using namespace dealii;
using namespace dealiiqc;


// Short test to check write_vtp() function of DataOutAtomData

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
                   Triangulation<dim>::limit_level_difference_at_vertices,
                   -1.),
    dof_handler    (triangulation),
    mpi_communicator(MPI_COMM_WORLD)
  {}

  void run()
  {
    {
      Point<dim> bl(0,0,0);
      Point<dim> tr(16,16,16);
      // create a mesh which is not aligned with crystal structure
      std::vector<unsigned int> repetitions(dim,7);
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                repetitions,
                                                bl,
                                                tr);
      triangulation.setup_ghost_cells();
    }

    const std::string atom_data_file = config.get_atom_data_file();
    std::fstream fin(atom_data_file, std::fstream::in );

    cell_molecule_data =
      CellMoleculeTools::
      build_cell_molecule_data<dim> (fin,
                                     triangulation);

    std::shared_ptr<Cluster::WeightsByBase<dim>> cluster_weights_method =
                                                config.get_cluster_weights<dim>();

    cluster_weights_method->initialize (triangulation,
                                        QTrapez<dim>());

    cell_molecule_data.cell_energy_molecules =
      cluster_weights_method->
      update_cluster_weights (triangulation,
                              cell_molecule_data.cell_molecules);

    write_output();
  }

  void write_output()
  {

    const unsigned int
    n_mpi_processes  = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator),
    this_mpi_process = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    dealii::DataOutBase::VtkFlags flags( std::numeric_limits<double>::min(),
                                         std::numeric_limits<unsigned int>::min(),
                                         false);

    DataOutAtomData atom_data_out;

    {
      std::stringstream ss;
      atom_data_out.write_vtp<3> (cell_molecule_data.cell_energy_molecules,
                                  flags,
                                  ss);

      const std::string
      output_file_name = "output" +
                         dealii::Utilities::int_to_string(this_mpi_process);

      std::ofstream output_file;
      output_file.open(output_file_name, std::ofstream::trunc);

      // Remove binary content in-between <DataArray> </DataArray>.
      Testing::filter_out_xml_key(ss, "DataArray", output_file);

      output_file << std::endl;

      MPI_Barrier(mpi_communicator);

      // Generating "output" file for testing without using deallog.
      {
        if (this_mpi_process==0)
          for (unsigned int p=0; p<n_mpi_processes; ++p)
            {
              const std::string file_name = "output" + dealii::Utilities::int_to_string(p);
              std::ifstream f(file_name);
              std::string line;
              while (std::getline(f, line))
                std::cout << line << std::endl;
            }
      }
    }

    MPI_Barrier(mpi_communicator);

#ifdef WRITE_ATOM_DATA
    // Write individual vtp files.
    const std::string
    vtp_file_name = "atoms-" +
                    dealii::Utilities::int_to_string(this_mpi_process,3) +
                    ".vtp";

    std::ofstream vtp_file;
    vtp_file.open (vtp_file_name.c_str(), std::ofstream::out | std::ofstream::trunc);

    atom_data_out.write_vtp<3> (cell_molecule_data.cell_energy_molecules,
                                flags,
                                vtp_file);

    vtp_file.close();

    // Write pvtp master file.
    if (this_mpi_process == 0)
      {
        std::vector<std::string> vtp_file_names;
        for (unsigned int i=0; i<n_mpi_processes; ++i)
          vtp_file_names.push_back ("atoms-" +
                                    dealii::Utilities::int_to_string(i,3) +
                                    ".vtp");

        std::ofstream pvtp_file;
        pvtp_file.open("atoms-.pvtp", std::ofstream::out | std::ofstream::trunc);
        atom_data_out.write_pvtp_record (vtp_file_names, flags, pvtp_file);
        pvtp_file.close();

        GridOut gridout;
        std::ofstream f("tria.vtk");
        gridout.write_vtk(triangulation, f);
      }
#endif
  }

private:
  const ConfigureQC                              &config;
  dealiiqc::parallel::shared::Triangulation<dim>  triangulation;
  DoFHandler<dim>                                 dof_handler;
  const MPI_Comm                                  mpi_communicator;
  CellMoleculeData<dim>                           cell_molecule_data;

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
          << "  set Atom data file = "
          << SOURCE_DIR "/../data/16_NaCl_atom.data"          << std::endl
          << "end" << std::endl
          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2.2"         << std::endl
          << "  set Cluster radius = 1.01"                    << std::endl
          << "  set Cluster weights by type = Cell"           << std::endl
          << "end" << std::endl;

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
