
#ifndef __dealii_qc_geometry_base_h
#define __dealii_qc_geometry_base_h

#include <deal.II/base/parameter_handler.h>
#include <deal.II/distributed/shared_tria.h>

#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

namespace Geometry
{

  /**
   * Declare geometry parameters in @p prm for all classes derived from Base.
   */
  void declare_parameters (ParameterHandler &prm);


  /**
   * Abstract base class to all geometric models.
   */
  template <int dim>
  class Base
  {
  public:
    /**
     * Constructor
     */
    Base();

    /**
     * Destructor.
     */
    virtual ~Base();

    /**
     * Create a parallel::shared::Triangulation @p tria based on the chosen geometry.
     */
    virtual void create_mesh(parallel::shared::Triangulation<dim> &tria) const = 0;

    /**
     * Parse parameter stored in @p prm .
     */
    virtual void parse_parameters(ParameterHandler &prm) = 0;

  protected:

    /**
     * Number of cycles of initial global refinement
     */
    unsigned int n_initial_global_refinements;

  };

  /**
   * Parse parameters stored in @p prm and return a geometry object.
   */
  template <int dim>
  std::shared_ptr<const Base<dim> >
  parse_parameters_and_get_geometry (ParameterHandler &prm);

} // namespace Geometry


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_geometry_base_h */
