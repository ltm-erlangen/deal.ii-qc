
#include <deal.II-qc/statics/fire.h>

#include <deal.II-qc/potentials/pair_lj_cut.h>


using namespace dealii;
using namespace dealiiqc;



// Test to verify correctness of SolverFIRE::sovle()
// The objective function is the Lennard-Jones function
//
// f(r) = eps * ( (rm/r)^12 - (rm/r)^6  ).
//
// In this test eps = 0.877 and
// rm = 1.55 (the mean distance between the two atoms).
//
// The value of r that minimizes f(r) is rm.



using vector_t = typename dealii::Vector<double>;


double compute (vector_t &G, const vector_t &u, const double a)
{
  AssertThrow (u.size() == 1 && G.size() == 1,
               ExcInternalError());

  const std::vector<double> lj_params = { 0.877, 1.55};

  Potential::PairLJCutManager lj (20);
  lj.declare_interactions (0,
                           1,
                           Potential::InteractionTypes::LJ,
                           lj_params);

  const double x = u(0) + a;

  std::pair<double, double> energy_force_0 =
    lj.energy_and_gradient<true> ( 0, 1, x*x);

  G(0) = energy_force_0.second;

  return energy_force_0.first;
}


void test (const double a,
           const double tol)
{

  vector_t tmp_mass;

  tmp_mass.reinit(1, true);

  // Use this to initialize DiagonalMatrix
  tmp_mass = 1.;

  DiagonalMatrix<vector_t> inv_mass;

  // Initialize inverse diagonal matrix.
  inv_mass.reinit(tmp_mass);

  auto additional_data =
    statics::SolverFIRE<vector_t>::AdditionalData(1e-3, 1e-3, 0.001);

  SolverControl solver_control (1e06, tol);

  statics::SolverFIRE<vector_t> fire (solver_control, additional_data);

  vector_t u;
  u.reinit(1, false);

  auto compute_function =
    [&](vector_t &G, const vector_t &U) -> double
  {
    return compute(G, U, a);
  };

  fire.solve(compute_function, u, inv_mass);

  std::cout << u(0) + a << std::endl;

}


int main ()
{
  test (3.1, 1e-15 );
  test (0.1, 1e-15 );
  test (9.1, 1e-15 );

}
