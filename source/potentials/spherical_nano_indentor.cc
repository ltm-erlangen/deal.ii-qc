
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/spherical_nano_indentor.h>


DEAL_II_QC_NAMESPACE_OPEN


template <int dim>
SphericalNanoIndentor<dim>::SphericalNanoIndentor(
  const Point<dim> &    initial_location,
  const Tensor<1, dim> &dir,
  const double          radius,
  const double          A,
  const double          initial_time)
  : NanoIndentor<dim>(initial_location, dir, false, initial_time)
  , radius(radius)
  , A(A)
{}



template <int dim>
SphericalNanoIndentor<dim>::~SphericalNanoIndentor()
{}



template <int dim>
double
SphericalNanoIndentor<dim>::value(const Point<dim> &p, const double) const
{
  const Point<dim> &center = NanoIndentor<dim>::current_location;

  const double distance = (p - center).norm();

  return distance < radius ? A * dealii::Utilities::fixed_power < (dim < 3) ?
                             2 :
                             3 > (radius - distance) :
                             0.;
}



template <int dim>
Tensor<1, dim>
SphericalNanoIndentor<dim>::gradient(const Point<dim> &p, const double) const
{
  const Point<dim> &center = NanoIndentor<dim>::current_location;

  const Tensor<1, dim> position_vector = p - center;
  const double         distance        = position_vector.norm();

  const unsigned int degree = (dim < 3) ? 2 : 3;

  const double factor =
    -A * degree *
    dealii::Utilities::fixed_power<degree - 1>(radius - distance) / distance;

  return distance < radius ? factor * position_vector : 0. * position_vector;
}



#define SPHERICAL_NANO_INDENTOR(R, X, _SPACE_DIM) \
  template class SphericalNanoIndentor<_SPACE_DIM>;

BOOST_PP_LIST_FOR_EACH(SPHERICAL_NANO_INDENTOR, BOOST_PP_NIL, _SPACE_DIM_)


DEAL_II_QC_NAMESPACE_CLOSE
