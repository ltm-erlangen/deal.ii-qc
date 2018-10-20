
#ifndef __dealii_qc_geometry_box_h
#define __dealii_qc_geometry_box_h

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II-qc/configure/geometry/geometry_base.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Geometry
{
  /**
   * Geometry defined by a box.
   */
  template <int dim>
  class Box : public Base<dim>
  {
  public:
    Box();

    virtual ~Box();

    virtual void
    create_mesh(dealii::parallel::shared::Triangulation<dim> &tria) const;

    virtual void
    parse_parameters(ParameterHandler &prm);

    static void
    declare_parameters(ParameterHandler &prm);

  private:
    /**
     * Extent of the box in x-, y-, and z-direction (in 3d).
     */
    Point<dim> extents;

    /**
     * Center of the box in x, y, and z (in 3d) coordinates.
     */
    Point<dim> center;

    /**
     * The number of cells in each coordinate direction
     */
    unsigned int repetitions[dim];
  };

} // namespace Geometry


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_geometry_box_h */
