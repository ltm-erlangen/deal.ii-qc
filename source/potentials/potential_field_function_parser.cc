
#include <deal.II-qc/potentials/potential_field_function_parser.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
PotentialFieldFunctionParser<spacedim>::PotentialFieldFunctionParser(
  const bool   is_electric_field,
  const double initial_time,
  const double h)
  : PotentialField<spacedim>(is_electric_field, initial_time)
  , function_object(1, initial_time, h)
{}



template <int spacedim>
PotentialFieldFunctionParser<spacedim>::~PotentialFieldFunctionParser()
{}



template <int spacedim>
void
PotentialFieldFunctionParser<spacedim>::initialize(
  const std::string &                  variables,
  const std::string &                  expression,
  const std::map<std::string, double> &constants,
  const bool                           time_dependent)
{
  function_object.initialize(variables, expression, constants, time_dependent);
}



template <int spacedim>
double
PotentialFieldFunctionParser<spacedim>::value(const Point<spacedim> &p,
                                              const double           q) const
{
  return PotentialField<spacedim>::is_electric_field ?
           function_object.value(p) * q :
           function_object.value(p);
}



template <int spacedim>
Tensor<1, spacedim>
PotentialFieldFunctionParser<spacedim>::gradient(const Point<spacedim> &p,
                                                 const double           q) const
{
  return PotentialField<spacedim>::is_electric_field ?
           function_object.gradient(p) * q :
           function_object.gradient(p);
}



template <int spacedim>
void
PotentialFieldFunctionParser<spacedim>::set_time(const double new_time)
{
  FunctionTime<double>::set_time(new_time);
  function_object.set_time(new_time);
}



template class PotentialFieldFunctionParser<1>;
template class PotentialFieldFunctionParser<2>;
template class PotentialFieldFunctionParser<3>;


DEAL_II_QC_NAMESPACE_CLOSE
