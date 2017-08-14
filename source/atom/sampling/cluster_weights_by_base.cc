
#include <deal.II-qc/atom/sampling/cluster_weights_by_base.h>
#include <deal.II-qc/atom/cell_molecule_tools.h>

#include <deal.II/lac/generic_linear_algebra.h>


DEAL_II_QC_NAMESPACE_OPEN


namespace Cluster
{


  template <int dim, int atomicity, int spacedim>
  WeightsByBase<dim, atomicity, spacedim>::
  WeightsByBase (const double &cluster_radius,
                 const double &maximum_cutoff_radius)
    :
    cluster_radius(cluster_radius),
    maximum_cutoff_radius(maximum_cutoff_radius),
    tria_ptr(NULL)
  {}



  template <int dim, int atomicity, int spacedim>
  WeightsByBase<dim, atomicity, spacedim>::~WeightsByBase()
  {}



  template <int dim, int atomicity, int spacedim>
  void
  WeightsByBase<dim, atomicity, spacedim>::
  initialize (const Triangulation<dim, spacedim> &triangulation,
              const Quadrature<dim>              &quadrature)
  {
    AssertThrow (dynamic_cast<const QTrapez<dim> *> (&quadrature) != NULL,
                 ExcNotImplemented());

    // TODO: Generalize sampling points by adding more sampling points.
    //       Using quadrature get sampling points from cells.
    //       Adjust the following code to work with
    //       generalized sampling points.
    //       But for now:

    // Initialize tria_ptr data member.
    tria_ptr = &triangulation;

    // Get locally relevant ghost cells
    /* TODO: This class is not aware of ghost_cell_layer_thickness
    const auto ghost_cells =
      GridTools::
      compute_ghost_cell_layer_within_distance (triangulation,
                                                ghost_cell_layer_thickness);
    */

    // Initialize cells_to_sampling_indices.
    for (types::CellIteratorType<dim, spacedim>
         cell  = triangulation.begin_active();
         cell != triangulation.end();
         cell++)
      {
        // If the cell is not locally relevant, jump.
        /* TODO:
        if (std::find (ghost_cells.begin(),
                       ghost_cells.end(),
                       cell) == ghost_cells.end() && !cell->is_locally_owned())
          continue;
        */

        std::set<unsigned int> this_cell_sampling_indices_set;

        // First, store vertex indices of all vertices of a given cell.
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
          this_cell_sampling_indices_set.insert(cell->vertex_index(v));

        // We also need to pick up all the hanging nodes, if any, on this cell
        // as the sampling points of this cell.
        // That's because if this cell is coarser than one (or more) of its
        // neighbors, we must gather all the molecules inside the cluster
        // sphere around a sampling point associated to the hanging node of the
        // neighboring cell.

        // So we need to pick up all the hanging nodes on this cell by
        // the following logic:
        // cell -> all faces -> all its sub-faces -> all its vertices.
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
          {
            const auto face = cell->face(f);
            if (face->has_children())
              for (unsigned int sf = 0; sf < face->number_of_children(); ++sf)
                {
                  const auto subface = face->child(sf);
                  for (unsigned int
                       v = 0;
                       v < GeometryInfo<dim>::vertices_per_face;
                       v++)
                    this_cell_sampling_indices_set.insert(subface->vertex_index(v));
                  Assert (!subface->has_children(),
                          ExcInternalError());
                }
          }

        std::vector<unsigned int>
        this_cell_sampling_indices (this_cell_sampling_indices_set.size());

        std::copy(this_cell_sampling_indices_set.cbegin(),
                  this_cell_sampling_indices_set.cend(),
                  this_cell_sampling_indices.begin());

        cells_to_sampling_indices[cell] = this_cell_sampling_indices;
      }

    locally_relevant_sampling_indices.clear();

    locally_relevant_sampling_indices.set_size(this->n_sampling_points());

    // Initialize locally relevant sampling indices.
    for (const auto &entry : cells_to_sampling_indices)
      {
        const std::vector<unsigned int> &sampling_indices = entry.second;

        // The set of sampling indices stores unique elements and
        // is already sorted, they can be directly added to
        // the locally relevant sampling indices.
        locally_relevant_sampling_indices.add_indices(sampling_indices.begin(),
                                                      sampling_indices.end());
      }

  }



  template <int dim, int atomicity, int spacedim>
  template <typename VectorType>
  void
  WeightsByBase<dim, atomicity, spacedim>::
  compute_dof_inverse_masses
  (VectorType                                       &inverse_masses,
   const DoFHandler<dim, spacedim>                  &dof_handler,
   const CellMoleculeData<dim, atomicity, spacedim> &cell_molecule_data) const
  {
    inverse_masses = 0.;

    Assert (inverse_masses.size() == dof_handler.n_dofs(),
            ExcMessage("Invalid inverse_masses provided."
                       "The size of the vector should equal to the total "
                       "number of dofs."))

    // For now support atomicity=1. // TODO: Extend to atomicity>1.
    AssertThrow (atomicity==1, ExcNotImplemented());

    const types::CellMoleculeContainerType<dim, atomicity, spacedim>
    &cell_energy_molecules = cell_molecule_data.cell_energy_molecules;

    for (auto
         cell  = dof_handler.begin_active();
         cell != dof_handler.end();
         cell++)
      {
        // Each process works only on its locally owned cell to sum up masses.
        if (!cell->is_locally_owned())
          continue;

        const types::CellIteratorType<dim, spacedim> tria_cell = cell;

        auto molecules_range_and_size =
          CellMoleculeTools::molecules_range_in_cell<dim, atomicity, spacedim>
          (tria_cell,
           cell_energy_molecules);

        // Get the global indices of the sampling points of this cell.
        const std::vector<unsigned int> this_cell_sampling_indices =
          WeightsByBase<dim, atomicity, spacedim>::get_sampling_indices(cell);

        // Get the sampling points of this cell in this container.
        const std::vector<Point<spacedim> > this_cell_sampling_points =
          WeightsByBase<dim, atomicity, spacedim>::get_sampling_points(cell);

        const unsigned int this_cell_n_sampling_points =
          this_cell_sampling_points.size();

        // Prepare the masses at sampling points in this container.
        std::vector<double>
        masses_at_sampling_points (this_cell_n_sampling_points, 0.);

        for (auto
             cell_molecule_itr  = molecules_range_and_size.first.first;
             cell_molecule_itr != molecules_range_and_size.first.second;
             cell_molecule_itr++)
          {
            const Molecule<spacedim, atomicity> &molecule =
              cell_molecule_itr->second;

            // Get the location of the closest sampling point (of this cell)
            // to the molecule in this_cell_sampling_points.
            const unsigned int location_in_container =
              Utilities::find_closest_point (molecule_initial_location(molecule),
                                             this_cell_sampling_points).first;

            masses_at_sampling_points[location_in_container] +=
              molecule.cluster_weight *
              cell_molecule_data.masses[molecule.atoms[0].type];
          }


        // Mask to identify which of the local sampling indices have been
        // accounted.
        std::vector<bool>
        this_cell_used_sampling_indices (this_cell_n_sampling_points, false);

        for (unsigned int i = 0; i < this_cell_sampling_indices.size(); i++)
          {
            const auto &sampling_index = this_cell_sampling_indices[i];

            // If the sampling_index is one of the vertices of the current cell,
            // get the corresponding dofs at the vertex to add masses.
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; v++)
              if (cell->vertex_index(v) == sampling_index)
                {
                  for (unsigned int
                       c = 0;
                       c < dof_handler.get_fe().n_dofs_per_vertex();
                       c++)
                    inverse_masses[cell->vertex_dof_index (v, c)] +=
                      masses_at_sampling_points[i];

                  this_cell_used_sampling_indices[i] = true;
                  break;
                }

            // If the control reaches here,
            // then this sampling index corresponds to a hanging node.

            // Loop through the faces to check which faces have children.
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
              {
                const auto &face = cell->face(f);

                if (face->has_children())
                  {
                    // Any one of the child of this face will surely have the
                    // hanging node as one of its vertices.

                    // Take the first child and run over all its vertices
                    // to find if the hanging node on this face corresponds
                    // to the current sampling index.
                    const auto &subface = face->child(0);

                    for (unsigned int
                         child_face_v = 0;
                         child_face_v < GeometryInfo<dim>::vertices_per_face;
                         child_face_v++)
                      if (subface->vertex_index(child_face_v) == sampling_index)
                        {
                          for (unsigned int
                               c = 0;
                               c < dof_handler.get_fe().n_dofs_per_vertex();
                               c++)
                            inverse_masses[subface->vertex_dof_index (child_face_v, c)] +=
                              masses_at_sampling_points[i];

                          this_cell_used_sampling_indices[i] = true;
                          break;
                        }
                  }

                if (this_cell_used_sampling_indices[i] == true)
                  break;
              } // for all faces of this cell.

            // We should have accounted for this sampling index when the control
            // reaches here.
            AssertThrow (this_cell_used_sampling_indices[i],
                         ExcInternalError());

          } // for all sampling indices of this cell.

      } // for locally owned cells.

    inverse_masses.compress(VectorOperation::add);

    // Take reciprocal of each locally owned entry to get inverse masses
    // from masses.
    for (typename VectorType::iterator
         entry  = inverse_masses.begin();
         entry != inverse_masses.end();
         entry++)
      *entry = 1./(*entry);
  }


#define SINGLE_WEIGHTS_BY_BASE_INSTANTIATION(DIM, ATOMICITY, SPACEDIM)  \
  template class WeightsByBase< DIM, ATOMICITY, SPACEDIM >;             \
  template void                                                         \
  WeightsByBase< DIM, ATOMICITY, SPACEDIM >::compute_dof_inverse_masses \
  (TrilinosWrappers::MPI::Vector                    &,                  \
   const DoFHandler<DIM, SPACEDIM>                  &,                  \
   const CellMoleculeData<DIM, ATOMICITY, SPACEDIM> &) const;

#define WEIGHTS_BY_BASE(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,         \
              SINGLE_WEIGHTS_BY_BASE_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X              \
   
  // WeightsByBase class Instantiations.
  INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(WEIGHTS_BY_BASE)

#undef SINGLE_WEIGHTS_BY_BASE_INSTANTIATION
#undef WEIGHTS_BY_BASE


} // namespace Cluster


DEAL_II_QC_NAMESPACE_CLOSE
