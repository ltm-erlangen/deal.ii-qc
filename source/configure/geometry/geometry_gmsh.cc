
#include <deal.II/grid/grid_in.h>

#include <deal.II-qc/configure/geometry/geometry_gmsh.h>

#include <fstream>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Geometry
{
  template <int dim>
  Gmsh<dim>::Gmsh()
    : Base<dim>()
  {}



  template <int dim>
  Gmsh<dim>::~Gmsh()
  {}



  template <int dim>
  void
  Gmsh<dim>::create_mesh(
    dealii::parallel::shared::Triangulation<dim> &mesh) const
  {
    GridIn<dim> gridin;
    gridin.attach_triangulation(mesh);
    std::ifstream mesh_stream(mesh_file.c_str());
    gridin.read_msh(mesh_stream);
    mesh.refine_global(Base<dim>::n_initial_global_refinements);
  }



  template <int dim>
  void
  Gmsh<dim>::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      Base<dim>::n_initial_global_refinements =
        prm.get_integer("Number of initial global refinements");
      prm.enter_subsection("Gmsh");
      {
        mesh_file = prm.get("File");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }



  template <int dim>
  void
  Gmsh<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Geometry");
    {
      prm.enter_subsection("Gmsh");
      {
        prm.declare_entry("File", "", Patterns::Anything(), "Input mesh file.");
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


DEAL_II_QC_NAMESPACE_CLOSE
