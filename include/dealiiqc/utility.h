
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>
#include <functional>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace dealiiqc
{
  /**
   * Custom types used inside dealiiqc.
   */
  namespace types
  {
    /**
     * The type used for global indices of atoms.
     * In order to have 64-bit unsigned integers (more than 4 billion),
     *  build deal.II with support for 64-bit integers.
     *  The data type always indicates an unsigned integer type.
     */
    typedef  dealii::types::global_dof_index global_atom_index;

    // TODO: Use of correct charge units; Use charge_t for book keeping.
    /**
     * The type used for storing charge of the atom.
     * Computations with charge of atoms don't need high precision.
     */
    typedef float charge;

    /**
     * The type used for identifying atom types. The enumeration starts
     * from 0.
     */
    typedef unsigned char atom_type;

  } //typedefs

  /**
   * Make sure that sscanf doesn't pickup spaces as unsigned char
   * while parsing atom data stream.
   */
#define UC_SCANF_STR "%hhu"

  namespace Utilities
  {
    using namespace dealii;

    /**
     * Extract and return a thick extended halo layer around a subdomain (set of
     * active cells) in the @p mesh (i.e. those cells that are within
     * a specified distance from the cells that share a common set of
     * vertices with the subdomain but are not a part of it). Here, the
     * "subdomain" consists of exactly all of those cells for which the @p
     * predicate returns @p true.
     *
     * @tparam MeshType A type that satisfies the requirements of the
     * @ref ConceptMeshType "MeshType concept".
     * @param[in] mesh A mesh (i.e. objects of type Triangulation, DoFHandler,
     * or hp::DoFHandler).
     * @param[in] predicate A function  (or object of a type with an operator())
     * defining the subdomain around which the halo layer is to be extracted. It
     * is a function that takes in an active cell and returns a boolean.
     * @return A list of active cells within a given distance from vertices of
     * the predicated subdomain.
     *
     * See deal.II documentation of the function compute_active_cell_halo_layer
     *
     */
    template< class MeshType>
    std::vector<typename MeshType::active_cell_iterator>
    compute_active_cell_extended_halo_layer
    (const MeshType                                                           &mesh,
     const std_cxx11::function<bool (const typename MeshType::active_cell_iterator &)> &predicate,
     const double                                                                      &max_extension)
    {
      std::vector<typename MeshType::active_cell_iterator> active_halo_layer;
      std::vector<bool> locally_active_vertices_on_subdomain (mesh.get_triangulation().n_vertices(),
                                                              false);

      // Find the cells for which the predicate is true
      // These are the cells around which we wish to construct
      // the halo layer
      for (typename MeshType::active_cell_iterator
           cell = mesh.begin_active();
           cell != mesh.end(); ++cell)
        if (predicate(cell)) // True predicate --> Part of subdomain
          for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
            locally_active_vertices_on_subdomain[cell->vertex_index(v)] = true;

      // Find the cells that do not conform to the predicate
      // but share a vertex with the selected subdomain
      // These comprise the halo layer
      for (typename MeshType::active_cell_iterator
           cell = mesh.begin_active();
           cell != mesh.end(); ++cell)
        if (!predicate(cell)) // False predicate --> Potential halo cell
          for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
            if (locally_active_vertices_on_subdomain[cell->vertex_index(v)] == true)
              {
                active_halo_layer.push_back(cell);
                break;
              }
      std::vector<typename MeshType::active_cell_iterator> extended_halo_layer(active_halo_layer);

      // touched vertices are initialized in a way that all predicate cells
      // and some of halo cell vertices are touched
      std::vector<bool> touched_vertex (locally_active_vertices_on_subdomain);

      // For all the cells that do no conform to the predicate
      // and that are not in the halo layer,
      // we evaluate the distance of separation from the
      // cells in the halo layer.
      // Find the cells which are closer than a specified distance
      // these comprise the extended halo layer

      for (typename MeshType::active_cell_iterator
           cell = mesh.begin_active();
           cell != mesh.end(); ++cell)
        if (!predicate(cell)) // False predicate --> Not part of subdomain
          if ( std::find(active_halo_layer.begin(), active_halo_layer.end(), cell)
               == active_halo_layer.end() ) // Not found --> Potential extended halo cell
            for ( auto halo_cell : active_halo_layer)
              for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
                if ( touched_vertex[cell->vertex_index(v)] == false ) // vertex not touched
                  for (unsigned int hv=0; hv<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++hv)
                    if ( halo_cell->vertex(hv).distance( cell->vertex(v)) < max_extension)
                      {
                        extended_halo_layer.push_back(cell);
                      }

      return extended_halo_layer;
    }

    /**
     * Extract and return a thick layer of ghost cells which are
     * with a given distance from the locally owned cells.
     *
     * @tparam MeshType A type that satisfies the requirements of the
     * @ref ConceptMeshType "MeshType concept".
     * @param[in] mesh A mesh (i.e. objects of type Triangulation, DoFHandler,
     * or hp::DoFHandler).
     * @param[in] max_extension the specified distance within which the cells are in
     * extended ghost layer
     * @return A list of ghost cells with a given distance from the locally owned
     * subdomain
     *
     * See deal.II documentation of the function compute_ghost_cell_halo_layer
     */
    template< class MeshType>
    std::vector<typename MeshType::active_cell_iterator>
    compute_ghost_cell_extended_halo_layer (const MeshType &mesh,
                                            const double &max_extension)
    {
      std_cxx11::function<bool (const typename MeshType::active_cell_iterator &)> predicate
        = IteratorFilters::LocallyOwnedCell();

      const std::vector<typename MeshType::active_cell_iterator>
      extended_halo_layer = compute_active_cell_extended_halo_layer (mesh, predicate, max_extension);

      // Check that we never return locally owned or artificial cells
      // What is left should only be the ghost cells
      /*
      Assert(contains_locally_owned_cells<MeshType>(extended_halo_layer) == false,
             ExcMessage("Halo layer contains locally owned cells"));
      Assert(contains_artificial_cells<MeshType>(extended_halo_layer) == false,
             ExcMessage("Halo layer contains artificial cells"));
             */

      return extended_halo_layer;
    }


  } // Utilities

}

#endif /* __dealii_qc_utility_h */
