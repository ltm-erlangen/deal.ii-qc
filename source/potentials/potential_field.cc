
#include <deal.II-qc/potentials/potential_field.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
PotentialField<spacedim>::PotentialField (const bool   is_electric_field,
                                          const double initial_time,
                                          const double h)
  :
  function_object (1, initial_time, h),
  is_electric_field (is_electric_field)
{}



template <int spacedim>
PotentialField<spacedim>::~PotentialField()
{}



template <int spacedim>
void PotentialField<spacedim>::initialize (const std::string                   &vars,
                                           const std::string                   &expression,
                                           const std::map<std::string, double> &constants,
                                           const bool                           time_dependent)
{
  function_object.initialize (vars,
                              expression,
                              constants,
                              time_dependent);
}



template <int spacedim>
double
PotentialField<spacedim>::value (const Point<spacedim> &p,
                                 const double           q) const
{
  return is_electric_field            ?
         function_object.value(p) * q :
         function_object.value(p);
}



template <int spacedim>
Tensor<1, spacedim>
PotentialField<spacedim>::gradient (const Point<spacedim> &p,
                                    const double           q) const
{
  return is_electric_field               ?
         function_object.gradient(p) * q :
         function_object.gradient(p);
}



template class PotentialField<1>;
template class PotentialField<2>;
template class PotentialField<3>;


DEAL_II_QC_NAMESPACE_CLOSE
