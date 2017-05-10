
#include <dealiiqc/configure/geometry/geometry_box.h>

namespace dealiiqc
{

  using namespace dealii;

  namespace Geometry
  {



    template <int dim>
    Box<dim>::Box()
      :
      Base<dim>()
    {}



    template <int dim>
    Box<dim>::~Box()
    {}



    template <int dim>
    void Box<dim>::create_mesh (parallel::shared::Triangulation<dim> &mesh) const
    {
      std::vector<unsigned int> rep_vec(repetitions, repetitions+dim);
      Point<dim> bottom_left;
      Point<dim> top_right;
      for (unsigned int d=0; d < dim; d++)
        {
          bottom_left[d] = center[d] - extents[d]/2.;
          top_right[d]   = center[d] + extents[d]/2.;
        }
      GridGenerator::subdivided_hyper_rectangle (mesh,
                                                 rep_vec,
                                                 bottom_left,
                                                 top_right,
                                                 false);
      mesh.refine_global(Base<dim>::n_initial_global_refinements);
    }



    template <>
    void Box<1>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        Base<1>::n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
        prm.enter_subsection("Box");
        {
          center[0] = prm.get_double ("X center");
          extents[0] = prm.get_double ("X extent");
          repetitions[0] = prm.get_integer ("X repetitions");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <>
    void Box<2>::parse_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        Base<2>::n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
        prm.enter_subsection("Box");
        {
          center[0] = prm.get_double ("X center");
          extents[0] = prm.get_double ("X extent");
          repetitions[0] = prm.get_integer ("X repetitions");

          center[1] = prm.get_double ("Y center");
          extents[1] = prm.get_double ("Y extent");
          repetitions[1] = prm.get_integer ("Y repetitions");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <>
    void Box<3>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        Base<3>::n_initial_global_refinements = prm.get_integer("Number of initial global refinements");
        prm.enter_subsection("Box");
        {
          center[0] = prm.get_double ("X center");
          extents[0] = prm.get_double ("X extent");
          repetitions[0] = prm.get_integer ("X repetitions");

          center[1] = prm.get_double ("Y center");
          extents[1] = prm.get_double ("Y extent");
          repetitions[1] = prm.get_integer ("Y repetitions");

          center[2] = prm.get_double ("Z center");
          extents[2] = prm.get_double ("Z extent");
          repetitions[2] = prm.get_integer ("Z repetitions");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void Box<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.enter_subsection("Box");
        {
          prm.declare_entry ("X extent", "10",
                             Patterns::Double (0),
                             "Extent of the box in x-direction. Units: Angstrom.");
          prm.declare_entry ("Y extent", "10",
                             Patterns::Double (0),
                             "Extent of the box in y-direction. Units: Angstrom.");
          prm.declare_entry ("Z extent", "10",
                             Patterns::Double (0),
                             "Extent of the box in z-direction. This value is ignored "
                             "if the simulation is in 2d. Units: Angstrom.");

          prm.declare_entry ("X center", "0",
                             Patterns::Double (),
                             "Center of the box in x-direction. Units: Angstrom.");
          prm.declare_entry ("Y center", "0",
                             Patterns::Double (),
                             "Center of the box in y-direction. Units: AnstromBohr.");
          prm.declare_entry ("Z center", "0",
                             Patterns::Double (),
                             "Center of the box in z-direction. This value is ignored "
                             "if the simulation is in 2d. Units: Angstrom.");

          prm.declare_entry ("X repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in X direction.");
          prm.declare_entry ("Y repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in Y direction.");
          prm.declare_entry ("Z repetitions", "1",
                             Patterns::Integer (1),
                             "Number of cells in Z direction. This value is ignored "
                             "if the simulation is in 2d.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    // instantiation
    template class Box<1>;
    template class Box<2>;
    template class Box<3>;

  } // namespace Geometry

} // namespace dealiiqc


