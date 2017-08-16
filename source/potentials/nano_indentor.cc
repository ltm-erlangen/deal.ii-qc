
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
NanoIndentor<spacedim>::
NanoIndentor(const Point<spacedim>     &initial_location,
             const Tensor<1, spacedim> &dir,
             const bool                 is_electric_field,
             const double               initial_time     )
  :
  PotentialField<spacedim>(is_electric_field, initial_time),
  indentor_position_function(1, initial_time),
  initial_location(initial_location),
  current_location(initial_location),
  direction(dir)
{
  Assert (std::fabs(dir.norm()-1.) < 1e-14,
          ExcMessage("The direction of the indentor should be a unit vector."));
}



template <int spacedim>
NanoIndentor<spacedim>::~NanoIndentor()
{}



template <int spacedim>
void NanoIndentor<spacedim>::initialize
(const std::string                   &variables,
 const std::string                   &expression,
 const std::map<std::string, double> &constants,
 const bool                           time_dependent)
{
  indentor_position_function.initialize (variables,
                                         expression,
                                         constants,
                                         time_dependent);
}



template <int spacedim>
void NanoIndentor<spacedim>::set_time (const double new_time)
{
  FunctionTime<double>::set_time(new_time);
  indentor_position_function.set_time (new_time);
  current_location = initial_location +
                     indentor_position_function.value(initial_location)
                     *
                     direction;
}



#define NANO_INDENTOR(R, X, SAPCEDIM)    \
  template class NanoIndentor<SAPCEDIM>; \
   
BOOST_PP_LIST_FOR_EACH (NANO_INDENTOR, BOOST_PP_NIL, SPACEDIM)


DEAL_II_QC_NAMESPACE_CLOSE
