
#ifndef __dealii_qc_potential_field_parser_h
#define __dealii_qc_potential_field_parser_h

#include <deal.II-qc/potentials/potential_field.h>

#include <deal.II/base/function_parser.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

/**
 * Class for describing scalar potential fields through a FunctionParser object.
 */
template <int spacedim>
class PotentialFieldParser : public PotentialField<spacedim>
{
public:

  /**
   * Constructor. May take a bool value @p is_electric_field that denotes
   * whether the scalar potential field describes an electric field (which
   * defaults to false), @p initial_time that defaults to zero and @p h that is
   * used for the computation of gradients using finite differences (which
   * defaults to 1e-8).
   */
  PotentialFieldParser (const bool   is_electric_field = false,
                        const double initial_time      = 0.,
                        const double h                 = 1e-8);

  /**
   * Destructor.
   */
  virtual ~PotentialFieldParser();

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
  double value (const Point<spacedim> &p,
                const double           q) const;

  /**
   * Return the gradient of the function evaluated at a given point @p p
   * with charge @p q.
   */
  Tensor<1, spacedim> gradient (const Point<spacedim> &p,
                                const double           q) const;

  /**
   * Set time for the time dependent #function_object using @p new_time.
   */
  void set_time (const double new_time);

private:

  /**
   * Function object.
   */
  FunctionParser<spacedim> function_object;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_potential_field_parser_h */
