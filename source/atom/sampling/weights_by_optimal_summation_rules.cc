
#include <deal.II/base/utilities.h>

#include <deal.II-qc/atom/sampling/weights_by_optimal_summation_rules.h>
#include <deal.II-qc/atom/cell_molecule_tools.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{


  template<int dim, int atomicity, int spacedim>
  WeightsByOptimalSummationRules<dim, atomicity, spacedim>::
  WeightsByOptimalSummationRules (const double &cluster_radius,
                                  const double &maximum_cutoff_radius,
                                  const double &rep_distance)
    :
    WeightsByBase<dim, atomicity, spacedim> (cluster_radius,
                                             maximum_cutoff_radius),
    rep_distance(rep_distance)
  {
    AssertThrow (cluster_radius==0, ExcNotImplemented());
  }



  namespace
  {

    /**
     * Return the dim-dimensional volume of the hyper-ball segment, the region
     * of the hyper-ball being cutoff by a hyper-plane that is @p d distance
     * from the center of the hyper-ball of radius @p radius.
     *
     * @note The permissible range of @p d is [0, radius].
     * When the value of @p d is 0 the volume of the half hyper-ball
     * is returned.
     */
    template <int dim>
    double hyperball_segment_volume (const double &radius, const double &d)
    {
      double volume;

      // Height of the segment.
      const double height = radius-d;

      AssertThrow (0 <= height && height <= 2.*radius,
                   ExcMessage("This function is called with invalid parameter: "
                              "d, the distance from the center"
                              "of the hyper-ball to the hyper-plane."
                              "Allowed range of d is [0, radius]."));
      if (dim==1)
        volume = height;
      else if (dim==2)
        {
          // Half of the angle inscribed at the center of the hyper-ball.
          const double alpha = std::acos(d/radius);
          volume = radius* (radius*alpha - d*std::sin(alpha));
        }
      else if (dim==3)
        volume = dealii::numbers::PI * height * height * (3*radius - height)/3.;

      return volume;
    }

  }



  // TODO: Implementation of second order sampling summation rules.
  // FIXME: Following assumptions are made:
  // 1. All cells are hyper-cubes. This assumption can be omitted later if
  //    the extent of cells in each direction is considered or distances between
  //    the vertex-type sampling points is considered.
  // 2. Triangulation is always refined/prepared in a such that cells at
  //    with the finest level can be considered to be in fully atomistic region.
  //    This is primarily to identify fully atomistic region.
  //    If this is not assumed we need lattice constants of the atomistic system
  //    to identify the fully atomistic region.
  template<int dim, int atomicity, int spacedim>
  types::CellMoleculeContainerType<dim, atomicity, spacedim>
  WeightsByOptimalSummationRules<dim, atomicity, spacedim>::
  update_cluster_weights
  (const Triangulation<dim, spacedim>                               &triangulation,
   const types::CellMoleculeContainerType<dim, atomicity, spacedim> &cell_molecules) const
  {
    const double molecule_density = CellMoleculeTools::compute_molecule_density
                                    <dim, atomicity, spacedim> (triangulation,
                                                                cell_molecules);

    const unsigned int &n_sampling_points = this->n_sampling_points();


    // Full value of the sampling weight is
    // 2*rep_distance for dim==1
    // pi*rep_distance^2 for dim==2
    // 4*pi*rep_distance^3/3 for dim==3
    const double full_weight = 2. *
                               hyperball_segment_volume<dim>(rep_distance,
                                                             0);

    const unsigned int n_levels = triangulation.n_global_levels();
    std::vector<bool>  weight_assigned (n_sampling_points, false);

    // Initialize with a full weights for all the sampling points.
    std::vector<float> weight (n_sampling_points, full_weight);

    // For inner-element type sampling points reset the weight to 0.
    for (unsigned int i = 0; i < n_sampling_points; ++i)
      if (this->is_quadrature_type(i))
        weight[i] = 0.;

    for (types::CellIteratorType<dim, spacedim>
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        std::vector<Point<spacedim> >
        this_cell_sampling_points = this->get_sampling_points(cell);

        std::vector<unsigned int>
        this_cell_sampling_indices = this->get_sampling_indices(cell);

        // If the cell is a finest cell of the triangulation (in the fully
        // atomistic region), assign 1. as weight for (vertex-type) sampling
        // points and 0. for the quadrature-type sampling point.
        if (cell->level()==n_levels-1)
          {
            for (const auto &index : this_cell_sampling_indices)
              if (!weight_assigned[index] && !(this->is_quadrature_type(index)))
                {
                  weight[index] = 1.;
                  weight_assigned[index] = true;
                }
          }
      }

    for (types::CellIteratorType<dim, spacedim>
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        const double half_cell_width = cell->extent_in_direction(0)/2.;

        std::vector<Point<spacedim> >
        this_cell_sampling_points = this->get_sampling_points(cell);

        std::vector<unsigned int>
        this_cell_sampling_indices = this->get_sampling_indices(cell);

        // If the cell is in the coarse region, such that none of
        // the representative spheres of the (vertex-type) sampling points
        // are not overlapping, set that the weights are assigned.
        if (half_cell_width > rep_distance)
          {
            for (const auto &index : this_cell_sampling_indices)
              if (!weight_assigned[index] && !(this->is_quadrature_type(index)))
                {
                  // Full weight is already set.
                  weight_assigned[index] = true;
                }
          }
        // If the cell is in the interface region such that
        // the representative spheres of the (vertex-type) sampling point
        // overlap, subtract the representative sphere overlap volume from
        // the full weights that were given to all sampling points except
        // inner-element type.
        else if (cell->level() < n_levels-1)
          {
            for (const auto &index : this_cell_sampling_indices)
              if (!weight_assigned[index] && !(this->is_quadrature_type(index)))
                {
                  weight[index] -= hyperball_segment_volume<dim>(rep_distance,
                                                                 half_cell_width)
                                   /
                                   dealii::Utilities::fixed_power<dim>(2.);

                  // In cases when a sampling point is a hanging node,
                  // we let adjacent cells whose level is lower than the current
                  // cell to reduce the weight of the sampling point and
                  // rely on suitable adjustment to the weight of the inner-element
                  // sampling point.

                  // Also, in this loop we do not touch the weights of
                  // sampling atoms that belong to cells in the finest and coarse
                  // regions of the atomistic system relying again on suitable
                  // adjustment to the weight of the inner-element
                  // sampling point.
                }
          }

      } // End of assigning weights to vertex-type sampling points.

    // Assign weight to inner-element type sampling atoms such that
    // all the weights within the cell add up to at least
    // molecule_density*cell_volume.
    for (types::CellIteratorType<dim, spacedim>
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      {
        // FIXME: Assuming the cell is a hyper-cube get half of the side length.
        const double half_cell_width = cell->extent_in_direction(0)/2.;

        std::vector<Point<spacedim> >
        this_cell_sampling_points = this->get_sampling_points(cell);

        std::vector<unsigned int>
        this_cell_sampling_indices = this->get_sampling_indices(cell);

        double vertex_weights_sum = 0.;

        for (const auto &index : this_cell_sampling_indices)
          if (!this->is_quadrature_type(index))
            vertex_weights_sum += weight[index];

        // For cells at the finest level, inner-element type sampling points
        // aren't considered and their weights are assigned to zero as
        // corresponding sampling atoms/molecules do not exist.
        if (cell->level() < n_levels-1)
          {
            for (const auto &index : this_cell_sampling_indices)
              if (this->is_quadrature_type(index))
                {
                  const double inner_weight = molecule_density*cell->measure()
                                              -
                                              vertex_weights_sum;
                  if (inner_weight > 0.)
                    weight[index] = inner_weight;
                }
          }
      }

    // Now that all the sampling points are assigned weights,
    // associate sampling points to sampling molecules.

    // Prepare energy molecules in this container.
    types::CellMoleculeContainerType<dim, atomicity, spacedim>
    cell_energy_molecules;

    // Get the squared_energy_radius to identify energy molecules.
    const double squared_energy_radius =
      dealii::Utilities::fixed_power<2>(this->maximum_cutoff_radius);

    // FIXME: Choose a suitable distance within which sampling points are to
    // find their sampling molecules.
    const double THICK_EPS = 1e-10;

    // Mask to identify which sampling points found their sampling molecules.
    std::vector<bool>  sampling_molecule_found (n_sampling_points, false);

    types::CellIteratorType<dim, spacedim> unique_cell =
      cell_molecules.begin()->first;

    // Get the sampling points and their indices of the first cell.
    std::vector<unsigned int>
    this_cell_sampling_indices = this->get_sampling_indices(unique_cell);

    std::vector<Point<spacedim> >
    this_cell_sampling_points = this->get_sampling_points(unique_cell);

    // Loop over all molecules and assign weights.
    for (auto &cell_molecule : cell_molecules)
      {
        const auto &cell = cell_molecule.first;
        Molecule<spacedim, atomicity> molecule = cell_molecule.second;

        if (unique_cell != cell)
          {
            unique_cell = cell;

            this_cell_sampling_indices = this->get_sampling_indices(unique_cell);
            this_cell_sampling_points  = this->get_sampling_points(unique_cell);
          }

        // Get the global index of the sampling point (of this cell) closest
        // to the molecule and the squared distance of separation.
        const std::pair<unsigned int, double> closest_sampling_point =
          Utilities::find_closest_point (molecule_initial_location(molecule),
                                         this_cell_sampling_points);

        const unsigned int closest_sampling_index =
          // Advance from begin to location_in_container to get the closest
          // sampling point index.
          *std::next (this_cell_sampling_indices.cbegin(),
                      closest_sampling_point.first);

        // Squared distance to the closest sampling point.
        const double &squared_distance = closest_sampling_point.second;

        if (squared_distance < squared_energy_radius)
          {
            // Squared distance to the closest sampling point is less than THICK_EPS
            if (squared_distance < THICK_EPS)
              {
                molecule.cluster_weight = weight[closest_sampling_index];
                AssertThrow (!sampling_molecule_found[closest_sampling_index],
                             ExcInternalError());
                sampling_molecule_found[closest_sampling_index] = true;
              }
            else
              molecule.cluster_weight = 0.;
            // Insert molecules into cell_energy_molecules if it is within
            // a distance of energy radius to associated cell's vertices.
            cell_energy_molecules.insert(std::make_pair(cell,molecule));
          }
      }

    // All the samplings points with non-zero weights should be able to find
    // their sampling molecules close to them.
    for (unsigned int i = 0; i < n_sampling_points; ++i)
      if (weight[i]>0)
        AssertThrow (sampling_molecule_found[i],
                     ExcMessage("At least one of the sampling points with "
                                "non-zero weight didn't find "
                                "a sampling atom/molecule near it."));

    return cell_energy_molecules;
  }



#define SINGLE_WEIGHTS_BY_OPTIMAL_INSTANTIATION(DIM, ATOMICITY, SPACEDIM)    \
  template class WeightsByOptimalSummationRules< DIM, ATOMICITY, SPACEDIM >; \
   
#define WEIGHTS_BY_OPTIMAL_RULES(R, X)                \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,           \
              SINGLE_WEIGHTS_BY_OPTIMAL_INSTANTIATION,\
              BOOST_PP_TUPLE_EAT(3)) X                \
   
  // WeightsBySamplingPoints class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_OPTIMAL_RULES)

#undef SINGLE_WEIGHTS_BY_OPTIMAL_INSTANTIATION
#undef WEIGHTS_BY_OPTIMAL_RULES


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
