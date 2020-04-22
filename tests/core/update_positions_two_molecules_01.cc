
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/solver_fire.h>

#include <deal.II-qc/atom/cell_molecule_tools.h>

#include <deal.II-qc/core/qc.h>

#include <fstream>
#include <iostream>
#include <sstream>

using namespace dealii;
using namespace dealiiqc;


// Check QC::update_positions() function by changing
// the QC::locally_relevant_displacement manually.
// Only one cell and two molecules are being used in this test for simplicity.


template <int dim, typename PotentialType, int atomicity>
class Problem : public QC<dim, PotentialType, atomicity>
{
public:
  Problem(const ConfigureQC &);
  void
  test();
};



template <int dim, typename PotentialType, int atomicity>
Problem<dim, PotentialType, atomicity>::Problem(const ConfigureQC &config)
  : QC<dim, PotentialType, atomicity>(config)
{}



template <int dim, typename PotentialType, int atomicity>
void
Problem<dim, PotentialType, atomicity>::test()
{
  this->setup_cell_energy_molecules();
  this->setup_system();
  this->setup_fe_values_objects();
  this->update_neighbor_lists();

  auto &displacement = this->locally_relevant_displacement;

  unsigned int i = 0;
  for (auto entry = displacement.begin(); entry != displacement.end(); entry++)
    *entry = 0.001 * i++;

  AssertThrow(i == this->dof_handler.n_dofs(), ExcInternalError());

  displacement.compress(VectorOperation::insert);

  this->locally_relevant_displacement.print(std::cout);

  this->update_positions();

  if (dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
    this->pcout << "Updated positions of atoms:" << std::endl;

  for (const auto &entry : this->cell_molecule_data.cell_energy_molecules)
    for (const auto &atom : entry.second.atoms)
      this->pcout << "Atom ID: " << atom.global_index
                  << " Position: " << std::fixed << std::setprecision(6)
                  << atom.position << std::endl;
}



int
main(int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, dealii::numbers::invalid_unsigned_int);

      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        deallog.depth_console(10);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 1;

      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = 1." << std::endl
          << "    set X extent = 2." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Number of atom types = 2" << std::endl
          << "  set Maximum cutoff radius = 2.0" << std::endl
          << "  set Pair potential type = LJ" << std::endl
          << "  set Pair global coefficients = 1.99 " << std::endl
          << "  set Pair specific coefficients = "
          << "                       1, 1, 0.877, 1.01;"
          << "                       1, 2, 0.877, 1.01;"
          << "                       2, 2, 0.877, 1.01;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 2.0" << std::endl
          << "end" << std::endl

          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "4 atoms" << std::endl
          << std::endl
          << "2  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1 .0 0.0 0.0 0." << std::endl
          << "2 1 2 .0 1.0 0.0 0." << std::endl
          << "3 2 1 .0 2.0 0.0 0." << std::endl
          << "4 2 2 .0 3.0 0.0 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      Problem<dim, Potential::PairLJCutManager, 2> problem(prm_stream);
      problem.test();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
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
      std::cerr << std::endl
                << std::endl
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
