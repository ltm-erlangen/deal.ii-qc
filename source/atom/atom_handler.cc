
#include <deal.II/distributed/shared_tria.h>

#include <dealiiqc/atom/atom_handler.h>

namespace dealiiqc
{

  template<int dim>
  AtomHandler<dim>::AtomHandler( const ConfigureQC &configure_qc)
    :
    configure_qc(configure_qc)
  {
  }

  template<int dim>
  void AtomHandler<dim>::parse_atoms_and_assign_to_cells( const MeshType &mesh)
  {
    // TODO: Assign atoms to cells as we parse atom data ?
    //       relevant for when we have a large collection of atoms.
    std::vector<Atom<dim>> vector_atoms;
    ParseAtomData<dim> atom_parser;

    if (!(configure_qc.get_atom_data_file()).empty() )
      {
        const std::string atom_data_file = configure_qc.get_atom_data_file();
        std::fstream fin(atom_data_file, std::fstream::in );
        atom_parser.parse( fin, vector_atoms, charges, masses);
      }
    else if ( !(* configure_qc.get_stream()).eof() )
      {
        atom_parser.parse( *configure_qc.get_stream(), vector_atoms, charges, masses);
      }
    else
      AssertThrow(false,
                  ExcMessage("Atom data was not provided neither as an auxiliary "
                             "data file nor at the end of the parameter file!"));

    // In order to speed-up finding an active cell around atoms through
    // find_active_cell_around_point(), we will need to construct a
    // mask for vertices of locally owned cells and ghost cells
    std::vector<bool> locally_active_vertices( mesh.get_triangulation().n_vertices(),
                                               false);

    // Mark (true) all the vertices of the locally owned cells
    for ( typename MeshType::active_cell_iterator
          cell = mesh.begin_active();
          cell != mesh.end(); ++cell)
      if ( cell->is_locally_owned())
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          locally_active_vertices[cell->vertex_index(v)] = true;

    // This MPI process also needs to know certain active ghost cells
    // within a certain distance from locally owned cells.
    // This MPI process will also keep copy of atoms associated to
    // such active ghost cells.
    // ghost_cells vector will contain all such active ghost cells.
    // If the total number of MPI processes is just one,
    // the size of ghost_cells vector is zero.
    const std::vector<typename MeshType::active_cell_iterator> ghost_cells =
      GridTools::compute_ghost_cell_layer_within_distance( mesh,
                                                           configure_qc.get_maximum_search_radius());

    // Mark (true) all the vertices of the active ghost cells within
    // a maximum search radius.
    for ( auto cell : ghost_cells)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        locally_active_vertices[cell->vertex_index(v)] = true;

    // TODO: If/when required collect all non-relevant atoms
    // (those that are not within a maximum search radius
    //  for this MPI process energy computation)
    // For now just add the number of atoms being thrown.
    types::global_atom_index n_thrown_atoms=0;

    for ( auto atom : vector_atoms )
      {
        bool atom_associated_to_cell = false;
        try
          {
            std::pair<typename MeshType::active_cell_iterator, Point<dim> >
            my_pair = GridTools::find_active_cell_around_point( MappingQ1<dim>(),
                                                                mesh,
                                                                atom.position,
                                                                locally_active_vertices);

            atom.reference_position = GeometryInfo<dim>::project_to_unit_cell(my_pair.second);
            // TODO: Remove parent_cell
            atom.parent_cell = my_pair.first;
            if ( Utilities::is_point_within_distance_from_cell_vertices( atom.position, my_pair.first, configure_qc.get_maximum_search_radius() ))
              {
                atom_associated_to_cell = true;
                atoms.insert( std::make_pair( my_pair.first, atom ));
              }
          }
        catch ( dealii::GridTools::ExcPointNotFound<dim> &)
          {
            // The atom is outside the cells that are relevant
            // to this MPI process. Ensuring quiet execution.
          }

        if ( !atom_associated_to_cell )
          n_thrown_atoms++;
      }

    Assert( atoms.size()+n_thrown_atoms==vector_atoms.size(),
            ExcInternalError());

  }

  template<int dim>
  void AtomHandler<dim>::update_neighbor_lists()
  {
    neighbor_lists.clear();

    // cell_neighbor_lists contains all the pairs of cell
    // whose atoms interact with each other.
    std::list< std::pair< CellIteratorType, CellIteratorType> > cell_neighbor_lists;

    const double cutoff_radius = configure_qc.get_maximum_energy_radius();
    const double cluster_radius = configure_qc.get_cluster_radius();

    // For each locally owned cell, identify all the cells
    // whose associated atoms may interact.At this point we do not
    // check if there are indeed some interacting atoms,
    // i.e. those within the cut-off radius. This is done to speedup
    // building of the neighbor list.
    // TODO: this approach strictly holds in the reference
    // (undeformed) configuration only.
    // It may still be ok for small deformations,
    // but for large deformations we may need to
    // use something like MappingQEulerian to work
    // with the deformed mesh.
    // TODO: optimize loop over unique keys ( mulitmap::upper_bound()'s complexity is O(nlogn) )
    for ( auto unique_I = atoms.cbegin(); unique_I != atoms.cend(); unique_I = atoms.upper_bound(unique_I->first))
      // Only locally owned cells have cell neighbors
      if ( unique_I->first->is_locally_owned()  )
        {
          const auto cell_I = unique_I->first;
          const double radius_I = cutoff_radius + Utilities::calculate_cell_radius<dim>(cell_I);
          for ( auto unique_J = atoms.cbegin(); unique_J != atoms.cend(); unique_J = atoms.upper_bound(unique_J->first))
            {
              const auto cell_J = unique_J->first;
              if ( (cell_I->center()-cell_J->center()).norm_square() <
                   dealii::Utilities::fixed_power<2>( radius_I +
                                                      Utilities::calculate_cell_radius<dim>(cell_J)) )
                cell_neighbor_lists.push_back( std::make_pair(cell_I, cell_J) );
            }
        }

    for ( const auto cell_pair_IJ : cell_neighbor_lists )
      {
        const CellIteratorType cell_I = cell_pair_IJ.first;
        const CellIteratorType cell_J = cell_pair_IJ.second;

        const auto range_of_cell_I = atoms.equal_range(cell_I);
        const auto range_of_cell_J = atoms.equal_range(cell_J);

        // for each atom associated to locally owned cell_I
        for ( auto cell_atom_I = range_of_cell_I.first; cell_atom_I != range_of_cell_I.second; ++cell_atom_I)
          {
            const Atom<dim> &atom_I = cell_atom_I->second;

            // TODO: Once functions updating cluster weights of atoms is implemented
            // add
            // bool is_cluster() const
            // {
            //    return cluster_weigth != 0.;
            // }
            // to the atom struct and use it here !!!
            // Check if the atom_I is cluster atom,
            // only cluster atoms get to have neighbor lists
            if ( Utilities::is_point_within_distance_from_cell_vertices( atom_I.position, cell_I, cluster_radius) )
              for ( auto cell_atom_J = range_of_cell_J.first; cell_atom_J != range_of_cell_J.second; ++cell_atom_J )
                {
                  const Atom<dim> &atom_J = cell_atom_J->second;

                  // TODO: Once functions updating cluster weights of atoms is implemented
                  bool atom_J_is_cluster_atom =
                    Utilities::is_point_within_distance_from_cell_vertices( atom_J.position, cell_J, cluster_radius );

                  // If atom_J is not cluster atom,
                  // then add atom_J to atom_i's neighbor list.

                  // If atom_J is also a cluster atom,
                  // then atom_J is only added to atom_I's neighbor list
                  // when atom_I's index is smaller. This ensures
                  // that there is no double counting of energy
                  // contribution due to cluster atoms - atom_I and atom_J

                  if ( ( atom_J_is_cluster_atom && (atom_I.global_index > atom_J.global_index))
                       ||
                       !atom_J_is_cluster_atom )
                    if ( (atom_I.position - atom_J.position).norm_square() < cutoff_radius*cutoff_radius)
                      neighbor_lists.insert( std::make_pair( cell_pair_IJ, std::make_pair( cell_atom_I, cell_atom_J)) );
                }
          }


      }

#ifdef DEBUG
    // For large deformations the cells might
    // distort leading to incorrect neighbor_list
    // using the above hierarchical procedure of
    // updating neighbor lists.
    // In debug mode, check if the number
    // of interactions is the same as computed
    // by the following.
    unsigned int total_number_of_interactions = 0;

    // loop over all atoms
    //   if locally owned cell
    //     loop over all atoms
    //       if within distance
    //         total_number_of_interactions++;
    for ( auto cell_atom_I : atoms )
      if ( cell_atom_I.first->is_locally_owned() )
        {
          const Atom<dim> atom_I = cell_atom_I.second;
          for ( auto cell_atom_J : atoms )
            {
              const Atom<dim> atom_J = cell_atom_J.second;
              // TODO: Once functions updating cluster weights of atoms is implemented
              // use is_cluster() member function in atom struct.
              bool atom_J_is_cluster_atom =
                  Utilities::is_point_within_distance_from_cell_vertices( atom_J.position, cell_atom_J.first, cluster_radius );
              if ( ( atom_J_is_cluster_atom && (atom_I.global_index > atom_J.global_index))
                   ||
                   !atom_J_is_cluster_atom )
                if ( (atom_I.position - atom_J.position).norm_square() < cutoff_radius*cutoff_radius)
                  total_number_of_interactions++;
            }
        }
    Assert( total_number_of_interactions == neighbor_lists.size(),
            ExcMessage("Some of the interactions are not accounted while updating neighbor lists"));
#endif

  }

  template class AtomHandler<1>;
  template class AtomHandler<2>;
  template class AtomHandler<3>;

} // dealiiqc namespace
