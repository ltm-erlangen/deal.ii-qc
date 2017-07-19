
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

  Point<dim> a = (dim==3) ? Point<dim>(0.23,0.37,1.33) :
                 (dim==2  ? Point<dim>(0.23,0.37     ) : Point<dim>(0.23));

  Point<dim> b = (dim==3) ? Point<dim>(0.47,0.63,1.77) :
                 (dim==2  ? Point<dim>(0.47,0.63     ) : Point<dim>(0.47));

  GridGenerator::hyper_rectangle (tria, a, b, false);

  const std::vector<Point<dim>> &points_1 = tria.get_vertices();

  Point<dim>  p = (dim==3) ? Point<dim>(0.33,0.54,1.47) :
                  (dim==2  ? Point<dim>(0.33,0.54     ) : Point<dim>(0.33));

  const auto closest =
    dealiiqc::Utilities::find_closest_point (p,
                                             points_1);

  const auto closest_vertex =
    dealiiqc::Utilities::find_closest_vertex (p,
                                              tria.begin_active());

  AssertThrow (closest.second == closest_vertex.second,
               ExcInternalError());

  std::cout << closest.first  << " " << closest.second << std::endl;
}

int main()
{
  test<1>();
  test<2>();
  test<3>();
}
