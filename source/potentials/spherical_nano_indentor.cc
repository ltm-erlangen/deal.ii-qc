
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/spherical_nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int spacedim, int degree>
SphericalNanoIndentor<spacedim, degree>::
SphericalNanoIndentor(const Point<spacedim>     &center,
                      const Tensor<1, spacedim> &dir,
                      const double               A,
                      const double               radius,
                      const double               initial_time)
  :
  NanoIndentor<spacedim, degree>(center, dir, A, false, initial_time),
  radius(radius)
{}



template <int spacedim, int degree>
SphericalNanoIndentor<spacedim, degree>::~SphericalNanoIndentor()
{}



template <int spacedim, int degree>
double
SphericalNanoIndentor<spacedim, degree>::value (const Point<spacedim> &p,
                                                const double            ) const
{
  const Point<spacedim> &center = NanoIndentor<spacedim, degree>::point;
  const double          &A      = NanoIndentor<spacedim, degree>::A;

  const double distance = (p-center).norm();

  return distance < radius ?
         A* dealii::Utilities::fixed_power<degree>(radius-distance)
         :
         0.;
}



template <int spacedim, int degree>
Tensor<1, spacedim>
SphericalNanoIndentor<spacedim, degree>::gradient (const Point<spacedim> &p,
                                                   const double            ) const
{
  const Point<spacedim> &center = NanoIndentor<spacedim, degree>::point;
  const double          &A      = NanoIndentor<spacedim, degree>::A;

  const Tensor<1, spacedim> position_vector = p-center;
  const double distance = position_vector.norm();

  const double
  factor = - A*degree* dealii::Utilities::fixed_power<degree-1>(radius-distance)
           /
           distance;

  return distance < radius ?
         factor * position_vector
         :
         0. * position_vector;
}



#define SPHERICAL_NANO_INDENTOR(R, X, SAPCEDIM) \
  template class SphericalNanoIndentor<SAPCEDIM, 2>; \
  template class SphericalNanoIndentor<SAPCEDIM, 3>; \
   
BOOST_PP_LIST_FOR_EACH (SPHERICAL_NANO_INDENTOR, BOOST_PP_NIL, SPACEDIM)


DEAL_II_QC_NAMESPACE_CLOSE
