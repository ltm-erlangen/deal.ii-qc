
#ifndef __dealii_qc_geometry_gmsh_h
#define __dealii_qc_geometry_gmsh_h

#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/configure/geometry/geometry_base.h>

namespace dealiiqc
{
  using namespace dealii;

  namespace Geometry
  {
    /**
     * Geometry defined by a Gmsh input file.
     */
    template <int dim>
    class Gmsh : public Base<dim>
    {
    public:

      Gmsh ();

      virtual ~Gmsh ();

      virtual void create_mesh (parallel::shared::Triangulation<dim> &tria) const;

      virtual void parse_parameters (ParameterHandler &prm);

      static void declare_parameters (ParameterHandler &prm);

    private:

      /**
       * Path to the mesh.
       */
      std::string mesh_file;
    };


  } // namesapce Geometry

} //namespace dealiiqc

#endif /* __dealii_qc_geometry_gmsh_h */
