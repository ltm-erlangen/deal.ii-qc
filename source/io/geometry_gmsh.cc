
#include <fstream>

#include <deal.II/grid/grid_in.h>

#include <dealiiqc/io/geometry_gmsh.h>

namespace dealiiqc
{

  using namespace dealii;

  namespace Geometry
  {



    template <int dim>
    Gmsh<dim>::Gmsh()
      :
      Base<dim>()
    {}



    template <int dim>
    Gmsh<dim>::~Gmsh()
    {}



    template <int dim>
    void Gmsh<dim>::create_coarse_mesh (parallel::shared::Triangulation<dim> &coarse_grid) const
    {
      GridIn<dim> gridin;
      gridin.attach_triangulation (coarse_grid);
      std::ifstream mesh_stream (mesh_file.c_str());
      gridin.read_msh (mesh_stream);
    }



    template <int dim>
    void Gmsh<dim>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.enter_subsection("Gmsh");
        {
          mesh_file = prm.get("File");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void Gmsh<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.enter_subsection("Gmsh");
        {
          prm.declare_entry ("File", "",
                             Patterns::Anything (),
                             "Input mesh file.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    // instantiation
    template class Gmsh<1>;
    template class Gmsh<2>;
    template class Gmsh<3>;


  } // namespace Geometry

} // namespace dealiiqc



