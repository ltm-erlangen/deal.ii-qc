
#include <deal.II/base/logstream.h>

#include <deal.II-qc/utilities.h>

#include <iostream>

#include "../tests.h"

// Short test to check correctness of
// dealiiqc::Utilities::hyperball_segment_volume()
// Test for extreme values of d i.e., 0 and radius

using namespace dealii;

template <int dim>
void
test(const double radius, const double d)
{
  const double volume =
    dealiiqc::Utilities::hyperball_segment_volume<dim>(radius, d);

  AssertThrow(dim == 2 || dim == 3, ExcNotImplemented());

  const double computed_volume =
    (dim == 2) ? dealii::numbers::PI * radius * radius :
                 dealii::numbers::PI * radius * radius * radius * 4. / 3.;

  if (d == 0)
    {
      AssertThrow(Testing::almost_equal(computed_volume / 2., volume, 100),
                  ExcInternalError());
    }
  else if (d == radius)
    {
      AssertThrow(Testing::almost_equal(0., volume, 10), ExcInternalError());
    }


  std::cout << "Radius: " << radius << std::endl
            << "Distance: " << d << std::endl
            << "Volume: " << volume << std::endl
            << std::endl;
}

int
main()
{
  test<2>(1.0354, 0);
  test<2>(1.0354, 1.0354);
  test<2>(2.8756, 0);
  test<2>(2.8756, 2.8756);
  test<3>(1.0354, 0);
  test<3>(1.0354, 1.0354);
  test<3>(2.8756, 0);
  test<3>(2.8756, 2.8756);
}
