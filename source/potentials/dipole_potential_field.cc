
#include <boost/preprocessor/list/for_each.hpp>

#include <deal.II-qc/potentials/dipole_potential_field.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;


template<int spacedim>
DipolePotentialField<spacedim>::
DipolePotentialField (const Point<spacedim>     &dipole_location,
                      const Tensor<1, spacedim> &orientation,
                      const double               dipole_moment,
                      const double               initial_time)
  :
  PotentialField<spacedim>(true, initial_time),
  dipole_location(dipole_location),
  dipole_orientation(orientation),
  dipole_moment(dipole_moment)
{
  dipole_orientation /= dipole_orientation.norm();
}



template<int spacedim>
double
DipolePotentialField<spacedim>::value (const Point<spacedim> &p,
                                       const double           q) const
{
  Tensor<1, spacedim> position_vector = (p-dipole_location);
  const double distance = position_vector.norm();

  // TODO: Need to setup units
  // The multiplying factor qqrd2e = 14.399645 yields energy in eV
  // and force in eV/Angstrom units
  return 14.399645*q*dipole_moment*(position_vector*dipole_orientation)
         /
         dealii::Utilities::fixed_power<3>(distance);
}



template<int spacedim>
Tensor<1, spacedim>
DipolePotentialField<spacedim>::gradient (const Point<spacedim> &p,
                                          const double           q) const
{
  Tensor<1, spacedim> normalized_position_vector = p-dipole_location;

  const double distance = normalized_position_vector.norm();

  // Normalize position vector now.
  normalized_position_vector /= distance;

  // TODO: Need to setup units
  // The multiplying factor qqrd2e = 14.399645 yields energy in eV
  // and force in eV/Angstrom units
  const double factor = 14.399645 *q *dipole_moment
                        /
                        dealii::Utilities::fixed_power<3>(distance);

  return factor * (dipole_orientation
                   -
                   3 * normalized_position_vector*
                   (dipole_orientation*normalized_position_vector)
                  );
}



#define DIPOLE_POTENTIAL_FIELD(R, X, _SPACE_DIM_)   \
  template class DipolePotentialField<_SPACE_DIM_>; \

BOOST_PP_LIST_FOR_EACH (DIPOLE_POTENTIAL_FIELD, BOOST_PP_NIL, _SPACE_DIM_)


DEAL_II_QC_NAMESPACE_CLOSE
