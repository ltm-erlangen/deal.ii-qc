
#ifndef __dealii_qc_spherical_nano_indentor_h
#define __dealii_qc_spherical_nano_indentor_h

#include <deal.II-qc/potentials/nano_indentor.h>

DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;


/**
 * Class for describing interaction potential under a spherical nano-indentor
 * present in a <tt>spacedim</tt>-dimensional space with an exponent of
 * <tt>degree</tt> in its empirical formula.
 *
 * NanoIndentor::point in this derived class is considered to be the center
 * of the spherical nano-indentor.
 */
template <int spacedim, int degree=2>
class SphericalNanoIndentor : public NanoIndentor<spacedim, degree>
{
public:

  /**
   * Constructor. Takes the following parameters
   * @param center        the center of the indentor,
   * @param dir           the direction of indentation,
   * @param A             the strength of the indentor,
   * @param radius        the radius of the spherical indentor, and
   * @param initial_time  the initial value of the time variable.
   */
  SphericalNanoIndentor(const Point<spacedim>     &center,
                        const Tensor<1, spacedim> &dir,
                        const double               A                 = 0.001,
                        const double               radius            = 100.0,
                        const double               initial_time      = 0.);

  /**
   * Destructor.
   */
  ~SphericalNanoIndentor();

  /**
   * Return the value of the function evaluated at a given point @p p.
   * The value of the function has the following empirical form,
   * \f[
   *     \phi(\boldsymbol{\mathsf x}) := A \,\, \delta \left( R-r \right)
   *                                       \,\,        \left( R-r \right)^n,
   * \f]
   * where \f$ r = |\boldsymbol{\mathsf x} - \boldsymbol{\mathsf c}|\f$,
   * \f$ \boldsymbol{\mathsf c} \f$ is the center of the indentor,
   * \f$\delta \left( R- r\right) \f$ is a Heaviside step function, \f$ n \f$ is the degree and
   * \f$ R \f$ is the radius of the indentor.
   */
  double value (const Point<spacedim> &p,
                const double            ) const;

  /**
   * Return the gradient of the function evaluated at a given point @p p..
   */
  Tensor<1, spacedim> gradient (const Point<spacedim> &p,
                                const double            ) const;

private:

  /**
   * Radius of the spherical indentor.
   */
  const double radius;

};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_spherical_nano_indentor_h */
