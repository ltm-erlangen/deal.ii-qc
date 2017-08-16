
#ifndef __dealii_qc_nano_indentor_h
#define __dealii_qc_nano_indentor_h

#include <deal.II-qc/potentials/potential_field.h>

#include <deal.II/base/function_parser.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;


/**
 * Abstract class for describing interaction potential under a nano-indentor
 * present in a <tt>spacedim</tt>-dimensional space with an exponent of
 * <tt>degree</tt> in its empirical formula.
 */
template <int spacedim>
class NanoIndentor : public PotentialField<spacedim>
{
public:

  /**
   * Constructor. Takes parameters @p initial_location and @p dir representing
   * the initial location (usually the center or the tip) and the direction,
   * respectively, and may take @p is_electric_field that denotes whether
   * the indentor induces an electric field (which defaults to false),
   * @p initial_time that defaults to zero.
   */
  NanoIndentor(const Point<spacedim>     &initial_location,
               const Tensor<1, spacedim> &dir,
               const bool                 is_electric_field = false,
               const double               initial_time      = 0.);

  /**
   * Destructor.
   */
  virtual ~NanoIndentor();

  /**
   * Initialize the FunctionParser object #indentor_position_function.
   */
  void initialize (const std::string                   &variables,
                   const std::string                   &expression,
                   const std::map<std::string, double> &constants,
                   const bool                           time_dependent = true);

  /**
   * Set the time to new_time, overwriting the old value.
   */
  void set_time (const double new_time);

protected:

  /**
   * FunctionParser object to describe the position of the indentor along
   * the direction of indentation #direction.
   */
  FunctionParser<spacedim> indentor_position_function;

  /**
   * Initial location of the indentor.
   */
  const Point<spacedim> initial_location;

  /**
   * Current location of the indentor.
   */
  Point<spacedim> current_location;

  /**
   * Unit vector along the direction of indentation.
   */
  const Tensor<1, spacedim> direction;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_nano_indentor_h */
