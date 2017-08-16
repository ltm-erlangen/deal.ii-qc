
#include <deal.II-qc/potentials/spherical_nano_indentor.h>

using namespace dealiiqc;
using namespace dealii;


// Test to check correctness of PotentialFieldParser value function.


template <int dim>
void test (const Point<dim>      &p,
           const double          q,
           const Tensor<1, dim> &dir,
           const double          time)
{
  SphericalNanoIndentor<dim, 2> potential (Point<dim>(),
                                           dir,
                                           0.001,
                                           4.1,
                                           0.);

  potential.initialize ((dim==3) ? "x,y,z,t" :
                        (dim==2  ? "x,y,t"   : "x,t"),
                        "t",
                        typename FunctionParser<dim>::ConstMap(),
                        true);

  potential.set_time(time);

  std::cout << time << ": Value: \t" << potential.value(p,q) << std::endl;
  std::cout << time << ": Gradient: \t" << potential.gradient(p,q) << std::endl;
}


int main ()
{
  const Point<1> p1 (1.235);
  const Point<2> p2 (1.235, -0.0433);
  const Point<3> p3 (1.235, -0.0433, 514.135);

  double a1[1] = {1};
  double a2[2] = {1, 0.};
  double a3[3] = {1, 0., 0.};

  const Tensor<1, 1> d1 (a1);
  const Tensor<1, 2> d2 (a2);
  const Tensor<1, 3> d3 (a3);

  const double q1 (.5654);

  test<1> (p1, q1, d1, 0.1);
  test<1> (p1, q1, d1, 1.1);

  test<2> (p2, q1, d2, 0.2);
  test<2> (p2, q1, d2, 1.2);

  test<3> (p3, q1, d3, 0.3);
  test<3> (p3, q1, d3, 1.3);
}



