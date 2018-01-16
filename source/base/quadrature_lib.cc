
#include <deal.II/base/utilities.h>
#include <deal.II-qc/base/quadrature_lib.h>

#include <deal.II/base/quadrature_lib.h>


DEAL_II_QC_NAMESPACE_OPEN

using namespace dealii;


template <int dim>
QTrapezWithMidpoint<dim>::QTrapezWithMidpoint()
:
Quadrature<dim>(dealii::Utilities::fixed_power<dim, int>(2)+1)
{
  const int n_points  = dealii::Utilities::fixed_power<dim, int>(2)+1;
  const double weight = 1./static_cast<double>(n_points);

  QTrapez<dim> q_trapez;
  QMidpoint<dim> q_midpoint;
  this->quadrature_points = q_trapez.get_points();

  for (unsigned int i=0; i<q_midpoint.size(); ++i)
   this->quadrature_points.push_back(q_midpoint.point(i));

  Assert (this->quadrature_points.size()==n_points,
          ExcInternalError());

  this->weights = std::vector<double>(n_points, weight);
}



template class QTrapezWithMidpoint<1>;
template class QTrapezWithMidpoint<2>;
template class QTrapezWithMidpoint<3>;


DEAL_II_QC_NAMESPACE_CLOSE
