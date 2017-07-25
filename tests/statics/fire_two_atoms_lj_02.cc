
// Check the convergence of FIRE minimization scheme applied to fully atomistic
// QC system with 2 atoms interacting via Lennard-Jones potential.
//
// *-------o
// |       |          o,*  - vertices
// |       |          o    - atoms
// |       |
// o-------*
//
// 6 entries of the gradient of the total energy are zeros.
// The displacement of the atom at the origin is constrained to zero.



#include <iostream>
#include <fstream>
#include <sstream>

#include <deal.II-qc/atom/cell_molecule_tools.h>
#include <deal.II-qc/core/qc.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/lac/solver_fire.h>


using namespace dealii;
using namespace dealiiqc;

// #define WRITE_GRID



template <int dim, typename PotentialType>
class Problem : public QC<dim, PotentialType>
{
  using vector_t = typename QC<dim, PotentialType>::vector_t;

public:
  Problem (const ConfigureQC &);
  void statics (const double tol);
  double compute (vector_t &,
                  const vector_t &);

};



template <int dim, typename PotentialType>
Problem<dim, PotentialType>::Problem (const ConfigureQC &config)
  :
  QC<dim, PotentialType>(config)
{}



template <int dim, typename PotentialType>
double Problem<dim, PotentialType>::compute (vector_t       &G,
                                             const vector_t &u)
{
  QC<dim, PotentialType>::locally_relevant_displacement = u;
  QC<dim, PotentialType>::update_positions ();

  const double energy =
    QC<dim, PotentialType>::template compute<true> (QC<dim, PotentialType>::locally_relevant_gradient);

  G = QC<dim, PotentialType>::locally_relevant_gradient;

  return energy;
}



template <int dim, typename PotentialType>
void Problem<dim, PotentialType>::statics (const double tol)
{
  QC<dim, PotentialType>::run();

  vector_t u (QC<dim, PotentialType>::dof_handler.locally_owned_dofs(),
              QC<dim, PotentialType>::mpi_communicator);

  // Use this to initialize DiagonalMatrix
  u = 1.;

  // Create inverse diagonal matrix.
  DiagonalMatrix<vector_t> inv_mass;
  inv_mass.reinit(u);

  auto additional_data =
    typename SolverFIRE<vector_t>::AdditionalData(0.15, 0.15, 0.15);

  SolverControl solver_control (1e04, tol);

  SolverFIRE<vector_t> fire (solver_control, additional_data);

  std::function<double(vector_t &,  const vector_t &)> compute_function =
    [&]               (vector_t &G, const vector_t &U) -> double
  {
    return this->compute(G, U);
  };

  u = 0.;

  fire.solve(compute_function, u, inv_mass);

  const unsigned int n_iterations = solver_control.last_step();

  AssertThrow ((n_iterations > 78) &&
               (n_iterations < 82),
               ExcInternalError("Need to re-adjust iteration bounds."
                                "It appears that FIRE took more or less "
                                "number of iterations to converge on this "
                                "machine."));

  QC<dim, PotentialType>::pcout
      << "SolverFIRE minimized energy to "
      << QC<dim, PotentialType>:: template
      compute<false> (QC<dim, PotentialType>::locally_relevant_gradient)
      << "eV within 79 to 81 iterations."
      << std::endl;

  QC<dim, PotentialType>::pcout
      << "Final positions of the two atoms:"
      << std::endl;

  for (const auto &entry :
       QC<dim, PotentialType>::cell_molecule_data.cell_energy_molecules)
    QC<dim, PotentialType>::pcout
        << entry.second.atoms[0].position << std::endl;
}



int main (int argc, char *argv[])
{
  try
    {
      dealii::Utilities::MPI::MPI_InitFinalize
      mpi_initialization (argc,
                          argv,
                          dealii::numbers::invalid_unsigned_int);

      // Allow the restriction that user must provide Dimension of the problem
      const unsigned int dim = 2;

      std::ostringstream oss;
      oss << "set Dimension = " << dim                        << std::endl

          << "subsection Geometry"                            << std::endl
          << "  set Type = Box"                               << std::endl
          << "  subsection Box"                               << std::endl
          << "    set X center = .5"                          << std::endl
          << "    set Y center = .5"                          << std::endl
          << "    set Z center = .5"                          << std::endl
          << "    set X extent = 1."                          << std::endl
          << "    set Y extent = 1."                          << std::endl
          << "    set Z extent = 1."                          << std::endl
          << "    set X repetitions = 1"                      << std::endl
          << "    set Y repetitions = 1"                      << std::endl
          << "    set Z repetitions = 1"                      << std::endl
          << "  end"                                          << std::endl
          << "  set Number of initial global refinements = 0" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure atoms"                     << std::endl
          << "  set Maximum cutoff radius = 2.0"              << std::endl
          << "  set Pair potential type = LJ"                 << std::endl
          << "  set Pair global coefficients = 1.99 "         << std::endl
          << "  set Pair specific coefficients = 0, 0, 0.877, 1.55;" << std::endl
          << "end"                                            << std::endl

          << "subsection Configure QC"                        << std::endl
          << "  set Ghost cell layer thickness = 2.01"        << std::endl
          << "  set Cluster radius = 2.0"                     << std::endl
          << "  set Boundary conditions = 2: 0."              << std::endl
          << "end"                                            << std::endl
          << "#end-of-parameter-section"                      << std::endl

          << "LAMMPS Description"              << std::endl   << std::endl
          << "2 atoms"                         << std::endl   << std::endl
          << "1  atom types"                   << std::endl   << std::endl
          << "Atoms #"                         << std::endl   << std::endl
          << "1 1 1 .0 0.0 0.0 0."                            << std::endl
          << "2 2 1 .0 1.0 1.0 0."                            << std::endl;

      std::shared_ptr<std::istream> prm_stream =
        std::make_shared<std::istringstream>(oss.str().c_str());

      Problem<dim, Potential::PairLJCutManager> problem(prm_stream);
      problem.statics (1e-15);
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
