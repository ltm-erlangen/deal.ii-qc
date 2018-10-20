
#ifndef __dealii_qc_spherical_nano_indentor_h
#define __dealii_qc_spherical_nano_indentor_h

#include <deal.II-qc/potentials/nano_indentor.h>

DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;


/**
 * Class for describing interaction potential under a spherical nano-indentor
 * present in a <tt>dim</tt>-dimensional space.
 *
 * The potential of spherical nano-indenter has the following empirical form,
 * \f[
 *     \phi(\boldsymbol{\mathsf x}) := A \,\, \delta \left( R-r \right)
 *                                       \,\,        \left( R-r \right)^n,
 * \f]
 * where \f$ r = |\boldsymbol{\mathsf x} - \boldsymbol{\mathsf c}|\f$,
 * \f$ \boldsymbol{\mathsf c} \f$ is the center of the indentor,
 * \f$\delta \left( R- r\right) \f$ is a Heaviside step function; \f$ A \f$ is
 * the strength, \f$ n \f$ is the degree and \f$ R \f$ is the radius
 * of the indentor.
 */
template <int dim>
class SphericalNanoIndentor : public NanoIndentor<dim>
{
public:
  /**
   * Constructor. Takes the following parameters
   * @param initial_location  the initial location of the indentor,
   * @param dir               the direction of indentation,
   * @param radius            the radius of the spherical indentor,
   * @param A                 the strength of the indentor, and
   * @param initial_time      the initial value of the time variable.
   */
  SphericalNanoIndentor(const Point<dim> &    initial_location,
                        const Tensor<1, dim> &dir,
                        const double          radius       = 100.0,
                        const double          A            = 0.001,
                        const double          initial_time = 0.);

  /**
   * Destructor.
   */
  ~SphericalNanoIndentor();

  /**
   * Return the value of the function evaluated at a given point @p p.
   *
   */
  double
  value(const Point<dim> &p, const double) const;

  /**
   * Return the gradient of the function evaluated at a given point @p p.
   */
  Tensor<1, dim>
  gradient(const Point<dim> &p, const double) const;

private:
  /**
   * Radius of the spherical indentor.
   */
  const double radius;

  /**
   * Parameter representing strength of the indentor.
   */
  double A;
};


DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_spherical_nano_indentor_h */
