
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN

namespace
{
  template <int dim>
  Tensor<1, dim> get_normalized_tensor (const Tensor<1, dim> &in_tensor)
  {
    Tensor<1, dim> unit_tensor = in_tensor;
    unit_tensor *= 1./in_tensor.norm();
    return unit_tensor;
  }
}


template <int dim>
NanoIndentor<dim>::
NanoIndentor(const Point<dim>     &initial_location,
             const Tensor<1, dim> &dir,
             const bool           is_electric_field,
             const double         initial_time     )
  :
  PotentialField<dim>(is_electric_field, initial_time),
  indentor_position_function(1, initial_time),
  initial_location(initial_location),
  current_location(initial_location),
  direction(get_normalized_tensor(dir))
{}



template <int dim>
NanoIndentor<dim>::~NanoIndentor()
{}



template <int dim>
void NanoIndentor<dim>::initialize
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



template <int dim>
void NanoIndentor<dim>::set_time (const double new_time)
{
  FunctionTime<double>::set_time(new_time);
  indentor_position_function.set_time (new_time);
  current_location = initial_location +
                     indentor_position_function.value(Point<1>(new_time))
                     *
                     direction;
}



#define NANO_INDENTOR(R, X,   _SPACE_DIM)  \
  template class NanoIndentor<_SPACE_DIM>;

BOOST_PP_LIST_FOR_EACH (NANO_INDENTOR, BOOST_PP_NIL, _SPACE_DIM_)


DEAL_II_QC_NAMESPACE_CLOSE
