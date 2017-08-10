
#ifndef __dealii_qc_potential_field_h
#define __dealii_qc_potential_field_h

#include <deal.II-qc/utilities.h>

#include <deal.II/base/function_parser.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;

/**
 * Class for function objects that describes a scalar potential field.
 */
template <int spacedim>
class PotentialField
{
public:

  /**
   * Constructor. May take a bool value @p is_electric_field that denotes
   * whether the scalar potential field describes an electric field (which
   * defaults to false), @p initial_time that defaults to zero and @p h that is
   * used for the computation of gradients using finite differences (which
   * defaults to 1e-8).
   *
   */
  PotentialField (const bool   is_electric_field = false,
                  const double initial_time      = 0.,
                  const double h                 = 1e-8);

  /**
   * Destructor.
   */
  virtual ~PotentialField();

  /**
   * Initialize the function parser. This methods accepts the following
   * parameters:
   * @param variables: a string consisting of comma separated variables that
   * will be used by the expressions to be evaluated.
   * @param expression: a string containing the expression that will be byte
   * compiled by the internal parser of #function_object.
   * @param constants: a map of constants used to pass any necessary constant
   * that we want to specify in @p expression.
   * @param time_dependent: if this is a time dependent function, then the
   * last variable declared in @p variables is assumed to be the time variable.
   */
  void initialize (const std::string                   &variables,
                   const std::string                   &expression,
                   const std::map<std::string, double> &constants,
                   const bool                           time_dependent);

  /**
   * Return the value of the function evaluated at a given point @p p
   * with charge @p q.
   */
  virtual double value (const Point<spacedim> &p,
                        const double           q) const;

  /**
   * Return the gradient of the function evaluated at a given point @p p
   * with charge @p q.
   */
  virtual Tensor<1, spacedim> gradient (const Point<spacedim> &p,
                                        const double           q) const;

  /**
   * Set time for the time dependent #function_object using a given @p time.
   */
  void set_time (const double time);

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
