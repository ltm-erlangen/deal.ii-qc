
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim, int degree>
NanoIndentor<spacedim, degree>::
NanoIndentor(const Point<spacedim>     &point,
             const Tensor<1, spacedim> &dir,
             const double               A,
             const bool                 is_electric_field,
             const double               initial_time     )
  :
  PotentialField<spacedim>(is_electric_field, initial_time),
  indentor_displacement_function(1, initial_time),
  point(point),
  direction(dir),
  A(A)
{
  Assert (std::fabs(dir.norm()-1.) < 1e-14,
          ExcMessage("The direction of the indentor should be a unit vector."));
}



template <int spacedim, int degree>
NanoIndentor<spacedim, degree>::~NanoIndentor()
{}



template <int spacedim, int degree>
void NanoIndentor<spacedim, degree>::initialize
(const std::string                   &variables,
 const std::string                   &expression,
 const std::map<std::string, double> &constants,
 const bool                           time_dependent)
{
  indentor_displacement_function.initialize (variables,
                                             expression,
                                             constants,
                                             time_dependent);
}



template <int spacedim, int degree>
void NanoIndentor<spacedim, degree>::set_time (const double new_time)
{
  FunctionTime<double>::set_time(new_time);
  indentor_displacement_function.set_time (new_time);
  point += indentor_displacement_function.value(point)*direction;
}



#define NANO_INDENTOR(R, X, SAPCEDIM) \
  template class NanoIndentor<SAPCEDIM, 2>; \
  template class NanoIndentor<SAPCEDIM, 3>; \
   
BOOST_PP_LIST_FOR_EACH (NANO_INDENTOR, BOOST_PP_NIL, SPACEDIM)


DEAL_II_QC_NAMESPACE_CLOSE
