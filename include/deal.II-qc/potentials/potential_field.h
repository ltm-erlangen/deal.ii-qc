
#ifndef __dealii_qc_potential_field_h
#define __dealii_qc_potential_field_h

#include <deal.II-qc/utilities.h>

#include <deal.II/base/function_time.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;

/**
 * Abstract class for describing scalar potential fields.
 */
template <int spacedim>
class PotentialField : public FunctionTime<double>
{
public:

  /**
   * Constructor. May take a bool value @p is_electric_field that denotes
   * whether the scalar potential field describes an electric field (which
   * defaults to false) and @p initial_time that defaults to zero.
   *
   */
  PotentialField (const bool   is_electric_field = false,
                  const double initial_time      = 0.   );

  /**
   * Destructor.
   */
  virtual ~PotentialField();

  /**
   * Return the value of the function evaluated at a given point @p p
   * with charge @p q.
   */
  virtual double value (const Point<spacedim> &p,
                        const double           q) const = 0;

  /**
   * Return the gradient of the function evaluated at a given point @p p
   * with charge @p q.
   */
  virtual Tensor<1, spacedim> gradient (const Point<spacedim> &p,
                                        const double           q) const = 0;

  /**
   * Whether the current function describes an electric potential field.
   */
  const bool is_electric_field;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_potential_field_h */
