
#include <deal.II-qc/statics/fire.h>

#include <deal.II/lac/la_vector.h>


using namespace dealii;
using namespace dealiiqc;



// Test to verify correctiness of SolverFIRE::sovle()
// The objective function is f(x) = x^2;



using vector_t = typename dealii::Vector<double>;


double compute (vector_t &g, const vector_t &x)
{
  AssertThrow (x.size() == 1 && g.size() == 1,
               ExcInternalError());

  g[0] = 2*x[0];

  return x[0]*x[0];
}


int main ()
{
  vector_t x, g;

  x.reinit(1, true);
  g.reinit(1, true);

  // Set initial iterate.
  x[0] = -4;

  auto additional_data =
      statics::SolverFIRE<vector_t>::AdditionalData(1e-3, 1e-1, 0.1, nullptr);

  SolverControl solver_control (1e03, 1e-03);

  statics::SolverFIRE<vector_t> fire (solver_control, additional_data);


}
