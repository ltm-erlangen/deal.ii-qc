
#ifndef __dealii_qc_geometry_gmsh_h
#define __dealii_qc_geometry_gmsh_h

#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/configure/geometry/geometry_base.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Geometry
{
  /**
   * Geometry defined by a Gmsh input file.
   *
   * @note Here it is assumed that the Gmsh input file specifically sets
   * boundary ids for the mesh.
   */
  template <int dim>
  class Gmsh : public Base<dim>
  {
  public:

    Gmsh ();

    virtual ~Gmsh ();

    virtual void create_mesh (dealii::parallel::shared::Triangulation<dim> &tria) const;

    virtual void parse_parameters (ParameterHandler &prm);

    static void declare_parameters (ParameterHandler &prm);

  private:

    /**
     * Path to the mesh.
     */
    std::string mesh_file;
  };


} // namesapce Geometry


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_geometry_gmsh_h */
