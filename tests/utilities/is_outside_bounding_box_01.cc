
// Short test to check correctness of
// dealiiqc::Utilities::is_outside_bounding_box()

#include <deal.II/base/point.h>

#include <deal.II-qc/utilities.h>

using namespace dealii;

int main()
{

  Point<1> minx(0.);
  Point<1> maxx(1.);
  Point<1> ax(1.75);
  Point<1> bx(1.);

  AssertThrow( dealiiqc::Utilities::is_outside_bounding_box( minx, maxx, ax ),
               ExcMessage("Test Fails"));

  // If the point is on the boundary, the point is not outside
  AssertThrow( !dealiiqc::Utilities::is_outside_bounding_box( minx, maxx, bx ),
               ExcMessage("Test Fails"));

  Point<2> minxy(0.,   0.);
  Point<2> maxxy(1.,   1.);
  Point<2> axy  (1.5, .75);
  Point<2> bxy  (0.,   1.);

  AssertThrow( dealiiqc::Utilities::is_outside_bounding_box( minxy, maxxy, axy ),
               ExcMessage("Test Fails"));

  // If the point is on the boundary, the point is not outside
  AssertThrow( !dealiiqc::Utilities::is_outside_bounding_box( minxy, maxxy, bxy ),
               ExcMessage("Test Fails"));

  Point<3> minxyz(0.,   0., 0.);
  Point<3> maxxyz(1.,   1., 1.);
  Point<3> axyz  (1.5, .75, 1.);
  Point<3> bxyz  (0.,   1., 1.);

  AssertThrow( dealiiqc::Utilities::is_outside_bounding_box( minxyz, maxxyz, axyz ),
               ExcMessage("Test Fails"));

  // If the point is on the boundary, the point is not outside
  AssertThrow( !dealiiqc::Utilities::is_outside_bounding_box( minxyz, maxxyz, bxyz ),
               ExcMessage("Test Fails"));

  std::cout << "OK" << std::endl;
}
