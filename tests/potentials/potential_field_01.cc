
#include <deal.II-qc/potentials/potential_field.h>

using namespace dealiiqc;
using namespace dealii;


// Test to check correctness of PotentialField value function.


template <int dim>
void test (const bool        is_electric_field,
           const Point<dim> &p,
           const double      q)
{
  PotentialField<dim> potential (is_electric_field, 0.);

  potential.initialize ((dim==3) ? "x,y,z,t" :
                        (dim==2  ? "x,y,t"   : "x,t"),
                        (dim==3) ? "x*x+y*y+z*z+t" :
                        (dim==2  ? "x*x+y*y    +t"   : "x*x + t"),
                        typename FunctionParser<dim>::ConstMap(),
                        true);

  if (is_electric_field)
    {
      AssertThrow (potential.value(p,q) == p.norm_square() * q,
                   ExcInternalError());
    }
  else
    {
      AssertThrow (potential.value(p,q) == p.norm_square(),
                   ExcInternalError());
    }
}


int main ()
{
  Point<1> p1 (1.235);
  Point<2> p2 (1.235, -0.0433);
  Point<3> p3 (1.235, -0.0433, 514.135);
  const double q1 (.5654);

  test<1> (false, p1, q1);
  test<1> (true,  p1, q1);

  test<2> (false, p2, q1);
  test<2> (true,  p2, q1);

  test<3> (false, p3, q1);
  test<3> (true,  p3, q1);

  std::cout << "TEST PASSED!" << std::endl;
}



