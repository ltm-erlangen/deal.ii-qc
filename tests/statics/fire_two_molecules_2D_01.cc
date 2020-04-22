
#include <deal.II/lac/solver_fire.h>

#include <deal.II-qc/core/qc.h>

#include <sstream>

using namespace dealii;
using namespace dealiiqc;


// Check whether the boundary conditions are corectly applied during
// energy minimization of the two molecule system inside a single cell
// in two dimensions.
//
// +-------+
// |       |
// o       o          o    - atoms
// o       o
// o-------o
//
// Since (dim==2 && atomicity==3) is true, the size of the list of boundary
// function expressions should be 6 for correctly specifying the boundary
// conditions at any boundary id.
//
// Boundary expressions are provided to impose dirichlet boundary conditions.
// At boundary_0 x coordinates of all atom stamps are constrained to obey
// homogeneous dirichlet conditions and at boundary_1 y coordinates of all
// atom stamps are constrained to obey homogeneous dirichlet conditions.


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
  this->update_positions();

  this->minimize_energy(-1);

  if (dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator) == 1)
    this->pcout << "Positions of atoms in the relaxed state:" << std::endl;

  for (const auto &entry : this->cell_molecule_data.cell_energy_molecules)
    for (const auto &atom : entry.second.atoms)
      this->pcout << "Atom ID: " << atom.global_index
                  << " Position: " << std::fixed << std::setprecision(10)
                  << atom.position << std::endl;

  this->pcout << "Energy: " << this->compute(this->locally_relevant_gradient)
              << std::endl;
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
      const unsigned int dim = 2;

      std::ostringstream oss;
      oss << "set Dimension = " << dim << std::endl

          << "subsection Geometry" << std::endl
          << "  set Type = Box" << std::endl
          << "  subsection Box" << std::endl
          << "    set X center = .5" << std::endl
          << "    set Y center = 2." << std::endl
          << "    set X extent = 1." << std::endl
          << "    set Y extent = 4." << std::endl
          << "    set X repetitions = 1" << std::endl
          << "    set Y repetitions = 1" << std::endl
          << "  end" << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end" << std::endl

          << "subsection Configure atoms" << std::endl
          << "  set Maximum cutoff radius = 2.0" << std::endl
          << "  set Pair potential type = LJ" << std::endl
          << "  set Pair global coefficients = 1.99 " << std::endl
          << "  set Pair specific coefficients = "
          << "                       1, 1, 0.877, 1.01;"
          << "                       1, 2, 0.877, 1.01;"
          << "                       1, 3, 0.877, 1.01;"
          << "                       2, 2, 0.877, 1.01;"
          << "                       2, 3, 0.877, 1.01;"
          << "                       3, 3, 0.877, 1.01;" << std::endl
          << "end" << std::endl

          << "subsection Configure QC" << std::endl
          << "  set Ghost cell layer thickness = 2.01" << std::endl
          << "  set Cluster radius = 2.0" << std::endl
          << "end" << std::endl

          << "subsection boundary_0" << std::endl
          << "  set Function expressions = 0.,,0.,,0.,," << std::endl
          << "end" << std::endl

          << "subsection boundary_1" << std::endl
          << "  set Function expressions = ,0.,,0.,,0." << std::endl
          << "end" << std::endl

          << "subsection Minimizer settings" << std::endl
          << "  set Max steps = 1000" << std::endl
          << "  set Tolerance = 1e-14" << std::endl
          << "  set Log history   = true" << std::endl
          << "  set Log frequency = 100" << std::endl
          << "  set Log result    = true" << std::endl

          << "  set Minimizer     = FIRE" << std::endl
          << "  subsection FIRE" << std::endl
          << "    set Initial time step = .005" << std::endl
          << "    set Maximum time step = .025" << std::endl
          << "    set Maximum linfty norm = .025" << std::endl
          << "  end" << std::endl
          << "end" << std::endl

          << "#end-of-parameter-section" << std::endl

          << "LAMMPS Description" << std::endl
          << std::endl
          << "6 atoms" << std::endl
          << std::endl
          << "3  atom types" << std::endl
          << std::endl
          << "Atoms #" << std::endl
          << std::endl
          << "1 1 1 .0 0.0 0.0 0." << std::endl
          << "2 1 2 .0 0.0 1.0 0." << std::endl
          << "3 1 3 .0 0.0 2.0 0." << std::endl
          << "4 2 1 .0 1.0 0.0 0." << std::endl
          << "5 2 2 .0 1.0 1.0 0." << std::endl
          << "6 2 3 .0 1.0 2.0 0." << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      Problem<dim, Potential::PairLJCutManager, 3> problem(prm_stream);
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
