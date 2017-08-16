
#include <deal.II-qc/potentials/potential_field_parser.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
PotentialFieldParser<spacedim>::PotentialFieldParser (const bool   is_electric_field,
                                                      const double initial_time,
                                                      const double h)
  :
  PotentialField<spacedim>(is_electric_field, initial_time),
  function_object (1, initial_time, h)
{}



template <int spacedim>
PotentialFieldParser<spacedim>::~PotentialFieldParser()
{}



template <int spacedim>
void
PotentialFieldParser<spacedim>::
initialize (const std::string                   &variables,
            const std::string                   &expression,
            const std::map<std::string, double> &constants,
            const bool                           time_dependent)
{
  function_object.initialize (variables,
                              expression,
                              constants,
                              time_dependent);
}



template <int spacedim>
double
PotentialFieldParser<spacedim>::value (const Point<spacedim> &p,
                                       const double           q) const
{
  return PotentialField<spacedim>::is_electric_field ?
         function_object.value(p) * q                :
         function_object.value(p);
}



template <int spacedim>
Tensor<1, spacedim>
PotentialFieldParser<spacedim>::gradient (const Point<spacedim> &p,
                                          const double           q) const
{
  return PotentialField<spacedim>::is_electric_field ?
         function_object.gradient(p) * q             :
         function_object.gradient(p);
}



template <int spacedim>
void PotentialFieldParser<spacedim>::set_time (const double new_time)
{
  function_object.set_time(new_time);
}



template class PotentialFieldParser<1>;
template class PotentialFieldParser<2>;
template class PotentialFieldParser<3>;


DEAL_II_QC_NAMESPACE_CLOSE
