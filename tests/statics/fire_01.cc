
#include <deal.II/lac/diagonal_matrix.h>

#include <deal.II-qc/statics/fire.h>


using namespace dealii;
using namespace dealiiqc;



// Test to verify correctiness of SolverFIRE::sovle()
// The objective function is f(x) = x^2;



using vector_t = typename dealii::Vector<double>;


double compute (vector_t &g, const vector_t &x)
{
  AssertThrow (x.size() == 1 && g.size() == 1,
               ExcInternalError());

  g(0) = 2*x(0);

  return x(0)*x(0);
}


int main ()
{
  vector_t x, g;

  x.reinit(1, true);
  g.reinit(1, true);

  // Set initial iterate.
  x(0) = 1.;

  DiagonalMatrix<vector_t> inv_mass;
  inv_mass.reinit(x);

  std::shared_ptr<const DiagonalMatrix<vector_t> >
  inverse_mass =
    std::make_shared<const DiagonalMatrix<vector_t>>(inv_mass);

  auto additional_data =
    statics::SolverFIRE<vector_t>::AdditionalData(1e-3, 1e-1, 0.1, inverse_mass);

  SolverControl solver_control (1e03, 1e-03);

  statics::SolverFIRE<vector_t> fire (solver_control, additional_data);

  fire.solve(compute, x);

  x.print(std::cout);


}
