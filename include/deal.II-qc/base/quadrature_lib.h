
#ifndef __dealii_qc_quadrature_lib_h_
#define __dealii_qc_quadrature_lib_h_

#include <deal.II/base/quadrature.h>

#include <deal.II-qc/utilities.h>


DEAL_II_QC_NAMESPACE_OPEN


/**
 * A quadrature rule that includes quadrature points from QTrapez and QMidpoint.
 */
template <int dim>
class QTrapezWithMidpoint : public dealii::Quadrature<dim>
{
  public:
    /**
     * Constructor.
     */
    QTrapezWithMidpoint();
};



DEAL_II_QC_NAMESPACE_CLOSE


#endif /* __dealii_qc_quadrature_lib_h_ */
