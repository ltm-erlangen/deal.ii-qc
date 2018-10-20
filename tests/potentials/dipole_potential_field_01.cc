
#include <deal.II-qc/potentials/dipole_potential_field.h>

#include <iostream>

using namespace dealiiqc;
using namespace dealii;


// Test to check value and gradient functions of DipolePotentialField class.


template <int dim>
void
test(const Point<dim> &    dipole_location,
     const Tensor<1, dim> &dipole_orientation,
     const double          dipole_moment,
     const Point<dim> &    p,
     const double          q)
{
  DipolePotentialField<dim> dipole(dipole_location,
                                   dipole_orientation,
                                   dipole_moment,
                                   0.);

  std::cout << "Energy: \t\t" << dipole.value(p, q) << std::endl;

  Tensor<1, dim> gradient = dipole.gradient(p, q);

  for (int i = 0; i < dim; ++i)
    std::cout << "Gradient[" << i << "]: \t" << gradient[i] << std::endl;
}


int
main()
{
  const Point<1> p1(1.235);
  const Point<2> p2(1.235, -0.0433);
  const Point<3> p3(1.235, -0.0433, 4.135);

  const Tensor<1, 1> orientation_1({0.12});
  const Tensor<1, 2> orientation_2({0.12, 0.32});
  const Tensor<1, 3> orientation_3({0.12, 0.32, .43});

  Point<1> location_1(3.24);
  Point<2> location_2(3.24, 2.54);
  Point<3> location_3(3.24, 2.54, 4.4);

  const double q1(.5654);

  test<1>(location_1, orientation_1, 2.1, p1, q1);
  test<2>(location_2, orientation_2, 2.1, p2, q1);
  test<3>(location_3, orientation_3, 2.1, p3, q1);
}
