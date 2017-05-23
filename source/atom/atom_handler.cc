
#include <deal.II/distributed/shared_tria.h>

#include <deal.II-qc/atom/atom_handler.h>



namespace dealiiqc
{



  template<int dim>
  AtomHandler<dim>::AtomHandler( const ConfigureQC &configure_qc)
    :
    configure_qc(configure_qc)
  {
  }



  template<int dim>
  void AtomHandler<dim>::parse_atoms_and_assign_to_cells( const types::MeshType<dim> &mesh,
                                                          AtomData<dim> &atom_data) const
  {
    // TODO: Assign atoms to cells as we parse atom data ?
    //       relevant for when we have a large collection of atoms.
    std::vector<Atom<dim>> vector_atoms;
    ParseAtomData<dim> atom_parser;

    atom_data.charges = NULL;
    std::vector<types::charge> charges;
    auto &masses  = atom_data.masses;
    auto &atoms = atom_data.atoms;

    const unsigned int n_vertices =  mesh.get_triangulation().n_vertices();

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

    atom_data.charges = std::make_shared<std::vector<types::charge>>(charges);

    // In order to speed-up finding an active cell around atoms through
    // find_active_cell_around_point(), we will need to construct a
    // mask for vertices of locally owned cells and ghost cells
    std::vector<bool> locally_active_vertices( n_vertices,
                                               false);

    // Loop through all the locally owned cells and
    // mark (true) all the vertices of the locally owned cells.
    for (typename types::MeshType<dim>::active_cell_iterator
         cell = mesh.begin_active();
         cell != mesh.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
          locally_active_vertices[cell->vertex_index(v)] = true;

    // This MPI process also needs to know certain active ghost cells
    // within a certain distance from locally owned cells.
    // This MPI process will also keep copy of atoms associated to
    // such active ghost cells.
    // ghost_cells vector will contain all such active ghost cells.
    // If the total number of MPI processes is just one,
    // the size of ghost_cells vector is zero.
    const std::vector<typename types::MeshType<dim>::active_cell_iterator> ghost_cells =
      GridTools::compute_ghost_cell_layer_within_distance( mesh,
                                                           configure_qc.get_ghost_cell_layer_thickness());

    // Loop through all the ghost cells computed above and
    // mark (true) all the vertices of the locally owned and active
    // ghost cells within ConfigureQC::ghost_cell_layer_thickness.
    for (auto cell : ghost_cells)
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        locally_active_vertices[cell->vertex_index(v)] = true;

    // TODO: If/when required collect all non-relevant atoms
    // (those that are not within a ConfigureQC::ghost_cell_layer_thickness
    // for this MPI process energy computation)
    // For now just add the number of atoms being thrown.
    types::global_atom_index n_thrown_atoms=0;

    for ( auto atom : vector_atoms )
      {
        bool atom_associated_to_cell = false;
        try
          {
            std::pair<typename types::MeshType<dim>::active_cell_iterator, Point<dim> >
            my_pair = GridTools::find_active_cell_around_point( MappingQ1<dim>(),
                                                                mesh,
                                                                atom.position,
                                                                locally_active_vertices);

            // Since in locally_active_vertices all the vertices of
            // the ghost cells are marked true, find_active_cell_around_point
            // could take the liberty to find a cell that is not a ghost cell
            // of a current MPI process but has one of it's vertices marked
            // true.
            // In such a case, we need to throw the atom and
            // continue associating remaining atoms.
            if (!my_pair.first->is_locally_owned() &&
                (std::find(ghost_cells.begin(), ghost_cells.end(), my_pair.first)==ghost_cells.end()))
              {
                n_thrown_atoms++;
                continue;
              }

            atom.reference_position = GeometryInfo<dim>::project_to_unit_cell(my_pair.second);
            // TODO: Remove parent_cell
            atom.parent_cell = my_pair.first;
            atoms.insert( std::make_pair( my_pair.first, atom ));
            atom_associated_to_cell = true;
          }
        catch ( dealii::GridTools::ExcPointNotFound<dim> &)
          {
            // The atom is outside the cells that are relevant
            // to this MPI process. Ensuring quiet execution.
          }

        if ( !atom_associated_to_cell )
          n_thrown_atoms++;
      }

    Assert (atoms.size()+n_thrown_atoms==vector_atoms.size(),
            ExcInternalError());

  }



  template<int dim>
  std::multimap< std::pair< types::ConstCellIteratorType<dim>, types::ConstCellIteratorType<dim>>, std::pair< types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> > >
      AtomHandler<dim>::get_neighbor_lists (const types::CellAtomContainerType<dim> &energy_atoms) const
  {
    // Check to see if energy_atoms is updated.
    Assert (energy_atoms.size(),
            ExcInternalError());

    std::multimap< std::pair< types::ConstCellIteratorType<dim>, types::ConstCellIteratorType<dim>>, std::pair< types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> > >
        neighbor_lists;

    // cell_neighbor_lists contains all the pairs of cell
    // whose atoms interact with each other.
    std::list< std::pair< types::CellIteratorType<dim>, types::CellIteratorType<dim>> > cell_neighbor_lists;

    const double cutoff_radius  = configure_qc.get_maximum_cutoff_radius();
    const double squared_cutoff_radius  = dealii::Utilities::fixed_power<2>(cutoff_radius);

    const double squared_cluster_radius =
      dealii::Utilities::fixed_power<2>(configure_qc.get_cluster_radius());

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
    for (types::CellAtomConstIteratorType<dim>
         unique_I  = energy_atoms.cbegin();
         unique_I != energy_atoms.cend();
         unique_I  = energy_atoms.upper_bound(unique_I->first))
      // Only locally owned cells have cell neighbors
      if (unique_I->first->is_locally_owned())
        {
          types::ConstCellIteratorType<dim> cell_I = unique_I->first;

          // Get center and the radius of the enclosing ball of cell_I
          const auto enclosing_ball_I = cell_I->enclosing_ball();

          for ( types::CellAtomConstIteratorType<dim> unique_J = energy_atoms.cbegin(); unique_J != energy_atoms.cend(); unique_J = energy_atoms.upper_bound(unique_J->first))
            {
              types::ConstCellIteratorType<dim> cell_J = unique_J->first;

              // Get center and the radius of the enclosing ball of cell_I
              const auto enclosing_ball_J = cell_J->enclosing_ball();

              // If the shortest distance between the enclosing balls of
              // cell_I and cell_J is less than cutoff_radius, then the
              // cell pair is in the cell_neighbor_lists.
              if (enclosing_ball_I.first.distance_square(enclosing_ball_J.first)
                  < dealii::Utilities::fixed_power<2>(cutoff_radius +
                                                      enclosing_ball_I.second +
                                                      enclosing_ball_J.second) )
                cell_neighbor_lists.push_back( std::make_pair(cell_I, cell_J) );
            }
        }

    for ( const auto cell_pair_IJ : cell_neighbor_lists )
      {
        types::ConstCellIteratorType<dim> cell_I = cell_pair_IJ.first;
        types::ConstCellIteratorType<dim> cell_J = cell_pair_IJ.second;

        std::pair< types::CellAtomConstIteratorType<dim>, types::CellAtomConstIteratorType<dim> >
        range_of_cell_I = energy_atoms.equal_range(cell_I),
        range_of_cell_J = energy_atoms.equal_range(cell_J);

        // for each atom associated to locally owned cell_I
        for (types::CellAtomConstIteratorType<dim>
             cell_atom_I  = range_of_cell_I.first;
             cell_atom_I != range_of_cell_I.second;
             cell_atom_I++)
          {
            const Atom<dim> &atom_I = cell_atom_I->second;

            // Check if the atom_I is cluster atom,
            // only cluster atoms get to have neighbor lists
            if (atom_I.cluster_weight != 0)
              for (types::CellAtomConstIteratorType<dim>
                   cell_atom_J  = range_of_cell_J.first;
                   cell_atom_J != range_of_cell_J.second;
                   cell_atom_J++ )
                {
                  const Atom<dim> &atom_J = cell_atom_J->second;

                  const bool atom_J_is_cluster_atom =
                    atom_J.cluster_weight != 0;

                  // If atom_J is not cluster atom,
                  // then add atom_J to (cluster) atom_I's neighbor list.

                  // If atom_J is also a cluster atom,
                  // then atom_J is only added to atom_I's neighbor list
                  // when atom_I's index is smaller. This ensures
                  // that there is no double counting of energy
                  // contribution due to cluster atoms - atom_I and atom_J

                  if ( (atom_J_is_cluster_atom &&
                        (atom_I.global_index > atom_J.global_index))
                       ||
                       !atom_J_is_cluster_atom )
                    if (atom_I.position.distance_square(atom_J.position) < squared_cutoff_radius)
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
    for (auto cell_atom_I : energy_atoms)
      if (cell_atom_I.first->is_locally_owned())
        {
          const Atom<dim> &atom_I = cell_atom_I.second;

          // We are building neighbor lists for only atom_Is
          // so we can skip atom_I if it's not a cluster atom.
          if (atom_I.cluster_weight != 0)
            for (auto cell_atom_J : energy_atoms)
              {
                const Atom<dim> &atom_J = cell_atom_J.second;

                const bool atom_J_is_cluster_atom =
                  atom_J.cluster_weight != 0;

                if ( (atom_J_is_cluster_atom &&
                       (atom_I.global_index > atom_J.global_index))
                     ||
                     !atom_J_is_cluster_atom )
                  if (atom_I.position.distance_square(atom_J.position) < squared_cutoff_radius)
                    total_number_of_interactions++;
              }
        }
    Assert (total_number_of_interactions == neighbor_lists.size(),
            ExcMessage("Some of the interactions are not accounted while updating neighbor lists"));
#endif

    return neighbor_lists;
  }



  template class AtomHandler<1>;
  template class AtomHandler<2>;
  template class AtomHandler<3>;


} // dealiiqc namespace
