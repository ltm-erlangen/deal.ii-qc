
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/spherical_nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim>
SphericalNanoIndentor<spacedim>::
SphericalNanoIndentor(const Point<spacedim>     &initial_location,
                      const Tensor<1, spacedim> &dir,
                      const double               A,
                      const double               radius,
                      const double               initial_time)
  :
  NanoIndentor<spacedim>(initial_location, dir, false, initial_time),
  radius(radius),
  A(A)
{}



template <int spacedim>
SphericalNanoIndentor<spacedim>::~SphericalNanoIndentor()
{}



template <int spacedim>
double
SphericalNanoIndentor<spacedim>::value (const Point<spacedim> &p,
                                        const double            ) const
{
  const Point<spacedim> &center = NanoIndentor<spacedim>::current_location;

  const double distance = (p-center).norm();

  return distance < radius ?
         A*dealii::Utilities::fixed_power<(spacedim<3) ? 2 : 3>(radius-distance)
         :
         0.;
}



template <int spacedim>
Tensor<1, spacedim>
SphericalNanoIndentor<spacedim>::gradient (const Point<spacedim> &p,
                                           const double            ) const
{
  const Point<spacedim> &center = NanoIndentor<spacedim>::current_location;

  const Tensor<1, spacedim> position_vector = p-center;
  const double distance = position_vector.norm();

  const unsigned int degree = (spacedim < 3) ? 2 : 3;

  const double
  factor = - A*degree*
           dealii::Utilities::fixed_power<degree-1>(radius-distance)
           /
           distance;

  return distance < radius ?
         factor * position_vector
         :
         0. * position_vector;
}



#define SPHERICAL_NANO_INDENTOR(R, X, SAPCEDIM)   \
  template class SphericalNanoIndentor<SAPCEDIM>; \
   
BOOST_PP_LIST_FOR_EACH (SPHERICAL_NANO_INDENTOR, BOOST_PP_NIL, SPACEDIM)


DEAL_II_QC_NAMESPACE_CLOSE
