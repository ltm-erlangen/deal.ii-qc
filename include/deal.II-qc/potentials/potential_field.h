
#ifndef __dealii_qc_potential_field_h
#define __dealii_qc_potential_field_h

#include <deal.II-qc/utilities.h>

#include <deal.II/base/function_parser.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;

/**
 * Class for function objects that describes a potential field.
 */
template <int spacedim>
class PotentialField
{
public:

  /**
   * Constructor.
   */
  PotentialField (bool         is_electric_field = false,
                  const double initial_time      = 0.,
                  const double h                 = 1e-8);

  /**
   * Destructor.
   */
  virtual ~PotentialField();

  /**
   * Initialize the function parser.
   */
  void initialize (const std::string                   &variables,
                   const std::string                   &expressions,
                   const std::map<std::string, double> &constants,
                   const bool                           time_dependent);

  /**
   * Return the value of the function at the given point @p p with charge
   * @p q.
   */
  virtual double value (const Point<spacedim> &p,
                        const double           q) const;

  /**
   * Return the gradient the function at the given point @p with charge @p q.
   */
  virtual Tensor<1, spacedim> gradient (const Point<spacedim> &p,
                                        const double           q) const;

private:

  /**
   * Function object.
   */
  dealii::FunctionParser<spacedim> function_object;

  /**
   * Whether the current function describes an electric potential field.
   */
  bool is_electric_field;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_potential_field_h */
