
#include <dealiiqc/configure/geometry/geometry_base.h>
#include <dealiiqc/configure/geometry/geometry_box.h>
#include <dealiiqc/configure/geometry/geometry_gmsh.h>

namespace dealiiqc
{

  using namespace dealii;

  namespace Geometry
  {


    template <int dim>
    Base<dim>::Base()
      :
      n_initial_global_refinements(numbers::invalid_unsigned_int)
    {}



    template <int dim>
    Base<dim>::~Base()
    {}



    // instantiation
    template class Base<1>;
    template class Base<2>;
    template class Base<3>;



    void declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry ("Type", "Box",
                           Patterns::Selection ("Box|Gmsh"),
                           "Domain geometry type.");
        prm.declare_entry ("Number of initial global refinements", "1",
                           Patterns::Integer(0),
                           "Number of global mesh refinement cycles "
                           "applied to initial grid");
      }
      prm.leave_subsection();

      // Goemetry::X<dim>::declare_parameters() is same for all dims
      Geometry::Box<3>::declare_parameters(prm);
      Geometry::Gmsh<3>::declare_parameters(prm);
    }



    template <int dim>
    std::shared_ptr<const Base<dim> >
    parse_parameters_and_get_geometry (ParameterHandler &prm)
    {
      std::string type;
      std::shared_ptr<Base<dim> > geometry;
      prm.enter_subsection("Geometry");
      {
        type = prm.get("Type");
      }
      prm.leave_subsection();

      if (type == "Box")
        geometry = std::make_shared<Geometry::Box<dim>>();
      else if (type =="Gmsh")
        geometry = std::make_shared<Geometry::Gmsh<dim>>();
      else
        AssertThrow(false, ExcInternalError());

      geometry->parse_parameters(prm);

      return std::const_pointer_cast<const Base<dim> >(geometry);
    }



    // instantiations:
    template std::shared_ptr<const Geometry::Base<1> > parse_parameters_and_get_geometry<1> (ParameterHandler &);
    template std::shared_ptr<const Geometry::Base<2> > parse_parameters_and_get_geometry<2> (ParameterHandler &);
    template std::shared_ptr<const Geometry::Base<3> > parse_parameters_and_get_geometry<3> (ParameterHandler &);



  } // namespace Geometry

} // namespace dealiiqc
