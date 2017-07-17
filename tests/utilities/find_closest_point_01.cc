
#include <deal.II-qc/utilities.h>

#include <deal.II/base/point.h>
#include <deal.II/grid/grid_generator.h>

using namespace dealii;
using namespace dealiiqc;



// Short test to check correctness of
// dealiiqc::Utilities::find_closest_point()



template<int dim>
void test()
{
  Triangulation<dim> tria;

  GridGenerator::hyper_cube (tria, 0., 1., false);

  const std::vector<Point<dim>> &points = tria.get_vertices();

  const Point<dim> p = (dim==3) ? Point<dim>(0.2,0.3,0.2) :
                       (dim==2  ? Point<dim>(0.2,0.3    ) : Point<dim>(0.2));

  const auto closest_point =
    dealiiqc::Utilities::find_closest_point (p,
                                             points);

  std::cout << closest_point.first  << " " << closest_point.second << std::endl;
}

int main()
{
  test<1>();
  test<2>();
  test<3>();
}
