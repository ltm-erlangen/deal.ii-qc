
#ifndef __dealii_qc_utility_h
#define __dealii_qc_utility_h

#include <algorithm>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>

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
     * Function to check if a Point @p p is a within a certain
     * @p distance from the vertices of a given parent cell.
     */
    template<int dim>
    inline
    bool
    is_point_within_distance_from_cell_vertices( const Point<dim> &p,
                                                 const typename Triangulation<dim>::cell_iterator cell,
                                                 const double &distance)
    {
      // Throw exception if the given cell is is not in a valid
      // cell iterator state.
      AssertThrow( cell->state() == IteratorState::valid,
                   ExcMessage( "The given cell iterator is not in a valid iterator state"));

      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        if (  (cell->vertex(v)- p).norm_square()
              < dealii::Utilities::fixed_power<2>( distance ) )
          return true;
      return false;
    }

    /**
     * Utility function that returns true if a point @p p is outside a bounding box.
     * The box is specified by two points @p minp and @p maxp (the order of
     * specifying points is important).
     */
    template<int dim>
    inline
    bool
    is_outside_bounding_box( const Point<dim> &minp,
                             const Point<dim> &maxp,
                             const Point<dim> &p)
    {
      for (unsigned int d=0; d<dim; ++d)
        if ( (minp[d] > p[d]) || (p[d] > maxp[d]) )
          {
            return true;
          }

      return false;
    }

    /**
     * Return radius of a given @p cell.
     * The radius of the cell is defined as the distance from
     * center of the cell to the farthest vertex.
     */
    template <int dim, typename Cell>
    inline
    double calculate_cell_radius(const Cell &cell)
    {
      double res = 0.;
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        res = std::max(res, ( (cell->vertex(v)-cell->center()).norm_square() ));
      return std::sqrt(res);
    }


  } // Utilities

  namespace GridTools
  {
    using namespace dealii;

    template<class MeshType>
    bool
    contains_locally_owned_cells (const std::vector<typename MeshType::active_cell_iterator> &cells)
    {
      for (typename std::vector<typename MeshType::active_cell_iterator>::const_iterator
           it = cells.begin(); it != cells.end(); ++it)
        {
          if ((*it)->is_locally_owned())
            return true;
        }
      return false;
    }

    template<class MeshType>
    bool
    contains_artificial_cells (const std::vector<typename MeshType::active_cell_iterator> &cells)
    {
      for (typename std::vector<typename MeshType::active_cell_iterator>::const_iterator
           it = cells.begin(); it != cells.end(); ++it)
        {
          if ((*it)->is_artificial())
            return true;
        }
      return false;
    }

    template <int dim, typename Cell>
    inline
    double calculate_cell_radius(const Cell &cell)
    {
      double res = 0.;
      for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        res = std::max(res, ( cell->vertex(v)).distance(cell->center()) );
      return res;
    }

    template< class MeshType>
    std::pair< Point<MeshType::space_dimension>, Point<MeshType::space_dimension> >
    compute_bounding_box
    ( const MeshType                                                                    &mesh,
      const std_cxx11::function<bool (const typename MeshType::active_cell_iterator &)> &predicate )
    {
      std::vector<bool> locally_active_vertices_on_subdomain (mesh.get_triangulation().n_vertices(),
                                                              false);

      const unsigned int spacedim = MeshType::space_dimension;
      // Two extreme points can define the proc's extended bounding box
      Point<MeshType::space_dimension> maxp, minp;

      // initialize minp and maxp with the first predicate cell center
      for ( typename MeshType::active_cell_iterator
            cell = mesh.begin_active();
            cell != mesh.end(); ++cell)
        if ( predicate(cell))
          {
            minp = cell->center();
            maxp = cell->center();
            break;
          }

      // Run through all the cells to check if it belongs to predicate domain,
      // if it belongs to the predicate domain, extend the bounding box.
      for ( typename MeshType::active_cell_iterator
            cell = mesh.begin_active();
            cell != mesh.end(); ++cell)
        if (predicate(cell)) // True predicate --> Part of subdomain
          for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
            if (locally_active_vertices_on_subdomain[cell->vertex_index(v)] == false)
              {
                locally_active_vertices_on_subdomain[cell->vertex_index(v)] = true;
                for ( unsigned int d=0; d<spacedim; ++d)
                  {
                    minp[d] = std::min( minp[d], cell->vertex(v)[d]);
                    maxp[d] = std::max( maxp[d], cell->vertex(v)[d]);
                  }
              }

      return std::make_pair( minp, maxp );
    }


    template <int dim, template <int, int> class MeshType, int spacedim>
    unsigned int
    find_closest_vertex (const MeshType<dim,spacedim> &mesh,
                         const Point<spacedim>        &p,
                         const std::vector<bool>      &marked_vertices=std::vector<bool>())
    {
      // first get the underlying
      // triangulation from the
      // mesh and determine vertices
      // and used vertices
      const Triangulation<dim, spacedim> &tria = mesh.get_triangulation();

      const std::vector< Point<spacedim> > &vertices = tria.get_vertices();

      Assert ( tria.get_vertices().size() == marked_vertices.size() || marked_vertices.size() ==0,
               ExcDimensionMismatch(tria.get_vertices().size(), marked_vertices.size()));

      // If p is an element of marked_vertices,
      // and q is that of used_Vertices,
      // the vector marked_vertices does NOT
      // contain unused vertices if p implies q.
      // I.e., if p is true q must be true
      // (if p is false, q could be false or true).
      // p implies q logic is encapsulated in ~p|q.
      Assert( std::equal( marked_vertices.begin(),
                          marked_vertices.end(),
                          tria.get_used_vertices().begin(),
                          [](bool p, bool q)
      {
        return ~p | q;
      })
      || marked_vertices.size()==0,
      ExcMessage("marked_vertices should be a subset of used vertices in the triangulation"
                 "but marked_vertices contains one or more vertices that are not used vertices!") );

      // In addition, if a vector bools
      // is specified (marked_vertices)
      // marking all the vertices which
      // could be the potentially closest
      // vertex to the point, use it instead
      // of used vertices
      const std::vector< bool       > &used     =
        (marked_vertices.size()==0) ? tria.get_used_vertices() : marked_vertices;

      // At the beginning, the first
      // used vertex is the closest one
      std::vector<bool>::const_iterator first =
        std::find(used.begin(), used.end(), true);

      // Assert that at least one vertex
      // is actually used
      Assert(first != used.end(), ExcInternalError());

      unsigned int best_vertex = std::distance(used.begin(), first);
      double       best_dist   = (p - vertices[best_vertex]).norm_square();

      // For all remaining vertices, test
      // whether they are any closer
      for (unsigned int j = best_vertex+1; j < vertices.size(); j++)
        if (used[j])
          {
            double dist = (p - vertices[j]).norm_square();
            if (dist < best_dist)
              {
                best_vertex = j;
                best_dist   = dist;
              }
          }

      return best_vertex;
    }


    template <int dim, template <int, int> class MeshType, int spacedim>
    std::pair<typename MeshType<dim, spacedim>::active_cell_iterator, Point<dim> >
    find_active_cell_around_point (const Mapping<dim,spacedim>  &mapping,
                                   const MeshType<dim,spacedim> &mesh,
                                   const Point<spacedim>        &p,
                                   const std::vector<bool>      &marked_vertices=std::vector<bool>())
    {
      typedef typename MeshType<dim, spacedim>::active_cell_iterator active_cell_iterator;

      // The best distance is set to the
      // maximum allowable distance from
      // the unit cell; we assume a
      // max. deviation of 1e-10
      double best_distance = 1e-10;
      int    best_level = -1;
      std::pair<active_cell_iterator, Point<dim> > best_cell;

      // Find closest vertex and determine
      // all adjacent cells
      std::vector<active_cell_iterator> adjacent_cells_tmp
        = dealii::GridTools::find_cells_adjacent_to_vertex(mesh,
                                                           find_closest_vertex(mesh, p, marked_vertices));

      // Make sure that we have found
      // at least one cell adjacent to vertex.
      Assert(adjacent_cells_tmp.size()>0, ExcInternalError());

      // Copy all the cells into a std::set
      std::set<active_cell_iterator> adjacent_cells (adjacent_cells_tmp.begin(),
                                                     adjacent_cells_tmp.end());
      std::set<active_cell_iterator> searched_cells;

      // Determine the maximal number of cells
      // in the grid.
      // As long as we have not found
      // the cell and have not searched
      // every cell in the triangulation,
      // we keep on looking.
      const unsigned int n_active_cells = mesh.get_triangulation().n_active_cells();
      bool found = false;
      unsigned int cells_searched = 0;
      while (!found && cells_searched < n_active_cells)
        {
          typename std::set<active_cell_iterator>::const_iterator
          cell = adjacent_cells.begin(),
          endc = adjacent_cells.end();
          for (; cell != endc; ++cell)
            {
              try
                {
                  const Point<dim> p_cell = mapping.transform_real_to_unit_cell(*cell, p);

                  // calculate the infinity norm of
                  // the distance vector to the unit cell.
                  const double dist = GeometryInfo<dim>::distance_to_unit_cell(p_cell);

                  // We compare if the point is inside the
                  // unit cell (or at least not too far
                  // outside). If it is, it is also checked
                  // that the cell has a more refined state
                  if ((dist < best_distance)
                      ||
                      ((dist == best_distance)
                       &&
                       ((*cell)->level() > best_level)))
                    {
                      found         = true;
                      best_distance = dist;
                      best_level    = (*cell)->level();
                      best_cell     = std::make_pair(*cell, p_cell);
                    }
                }
              catch (typename MappingQGeneric<dim,spacedim>::ExcTransformationFailed &)
                {
                  // ok, the transformation
                  // failed presumably
                  // because the point we
                  // are looking for lies
                  // outside the current
                  // cell. this means that
                  // the current cell can't
                  // be the cell around the
                  // point, so just ignore
                  // this cell and move on
                  // to the next
                }
            }

          // update the number of cells searched
          cells_searched += adjacent_cells.size();

        }

      AssertThrow (best_cell.first.state() == IteratorState::valid,
                   dealii::GridTools::ExcPointNotFound<spacedim>(p));

      return best_cell;
    }


    template <class MeshType>
    std::vector<typename MeshType::active_cell_iterator>
    compute_active_cell_layer_within_distance
    (const MeshType                                                                    &mesh,
     const std_cxx11::function<bool (const typename MeshType::active_cell_iterator &)> &predicate,
     const double                                                                       layer_thickness)
    {
      std::vector<typename MeshType::active_cell_iterator> subdomain_boundary_cells, active_cell_layer_within_distance;
      std::vector<bool> vertices_outside_subdomain ( mesh.get_triangulation().n_vertices(),
                                                     false);

      const unsigned int spacedim = MeshType::space_dimension;

      // Find the layer of cells for which predicate is true and that
      // are on the boundary with other cells. These are
      // subdomain boundary cells.


      unsigned int n_non_predicate_cells = 0; // Number of non predicate cells

      // Find the layer of cells for which predicate is true and that
      // are on the boundary with other cells. These are
      // subdomain boundary cells.

      // Find the cells for which the predicate is false
      // These are the cells which are around the predicate subdomain
      for ( typename MeshType::active_cell_iterator
            cell = mesh.begin_active();
            cell != mesh.end(); ++cell)
        if ( !predicate(cell)) // Negation of predicate --> Not Part of subdomain
          {
            for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
              vertices_outside_subdomain[cell->vertex_index(v)] = true;
            n_non_predicate_cells++;
          }

      // If all the active cells confirm to the predicate
      // or if none of the active cells confirm to the predicate
      // there is no active cell layer around the predicate
      // subdomain (within any distance)
      if ( n_non_predicate_cells == 0  || n_non_predicate_cells == mesh.get_triangulation().n_active_cells() )
        return std::vector<typename MeshType::active_cell_iterator>();

      // Find the cells that conform to the predicate
      // but share a vertex with the cell not in the predicate subdomain
      for ( typename MeshType::active_cell_iterator
            cell = mesh.begin_active();
            cell != mesh.end(); ++cell)
        if ( predicate(cell)) // True predicate --> Potential boundary cell of the subdomain
          for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
            if (vertices_outside_subdomain[cell->vertex_index(v)] == true)
              {
                subdomain_boundary_cells.push_back(cell);
                break; // No need to go through remaining vertices
              }

      AssertThrow( subdomain_boundary_cells.size() != 0,
                   ExcMessage("None of the active cells confirm to the predicate"));

      // To cheaply filter out some cells located far away from the predicate subdomain,
      // get the bounding box of the predicate subdomain.
      std::pair< Point<spacedim>, Point<spacedim> > bounding_box = compute_bounding_box( mesh,
          predicate );
      // Add layer_thickness to the bounding box
      for ( int d=0; d<spacedim; ++d)
        {
          bounding_box.first[d]  -= layer_thickness; // minp
          bounding_box.second[d] += layer_thickness; // maxp
        }

      std::vector< Point<spacedim> > boundary_cells_centers; // cache all the subdomain boundary cells centers here
      std::vector< double> boundary_cells_radii; // cache all the subdomain boundary cells radii

      // compute cell radius for each boundary cell of the predicate subdomain
      for ( typename std::vector<typename MeshType::active_cell_iterator>::const_iterator
            boundary_cell_iterator  = subdomain_boundary_cells.begin();
            boundary_cell_iterator != subdomain_boundary_cells.end(); ++boundary_cell_iterator )
        {
          boundary_cells_centers.push_back( (*boundary_cell_iterator)->center() );
          boundary_cells_radii.push_back(calculate_cell_radius<MeshType::dimension>( *boundary_cell_iterator ));
        }
      AssertThrow( boundary_cells_radii.size() == boundary_cells_centers.size(),
                   ExcInternalError());

      // Find the cells that are within layer_thickness of predicate subdomain boundary
      // distance but are inside the extended bounding box.
      // Most cells might be outside the extended bounding box, so we could skip them.
      // Those cells that are inside the extended bounding box but are not part of the
      // predicate subdomain are possible candidates to be within the distance to the
      // boundary cells of the predicate subdomain.
      for ( typename MeshType::active_cell_iterator
            cell = mesh.begin_active();
            cell != mesh.end(); ++cell)
        {
          // Ignore all the cells that are in the predicate subdomain
          if ( predicate(cell))
            continue;

          // define cell_radius as the farthest distance of cell vertices to cell center
          const double cell_radius = calculate_cell_radius<MeshType::dimension>(cell);

          bool cell_inside = true; // reset for each cell

          // Faster to check to check with cell->center() instead of all of its vertices
          for (unsigned int d = 0; d < spacedim; ++d)
            cell_inside &= (cell->center()[d] + cell_radius > bounding_box.first[d])
                           && (cell->center()[d] - cell_radius < bounding_box.second[d]);
          // cell_inside is true if any of the cell vertices are inside the extended bounding box

          // Ignore all the cells that are outside the extended bounding box
          if ( cell_inside) // cell is
            for ( unsigned int i =0; i< boundary_cells_radii.size(); ++i)
              if ( (cell->center() - boundary_cells_centers[i]).norm_square()
                   <  dealii::Utilities::fixed_power<2>( cell_radius + boundary_cells_radii[i] + layer_thickness) )
                {
                  active_cell_layer_within_distance.push_back(cell);
                  break; // Exit the loop checking all the remaining subdomain boundary cells
                }

        }
      return active_cell_layer_within_distance;
    }


    template <class MeshType>
    std::vector<typename MeshType::active_cell_iterator>
    compute_ghost_cell_layer_within_distance ( const MeshType &mesh, const double layer_thickness)
    {
      std_cxx11::function<bool (const typename MeshType::active_cell_iterator &)> predicate
        = IteratorFilters::LocallyOwnedCell();

      const std::vector<typename MeshType::active_cell_iterator>
      ghost_cell_layer_within_distance = compute_active_cell_layer_within_distance (mesh, predicate, layer_thickness);

      // Check that we never return locally owned or artificial cells
      // What is left should only be the ghost cells
      Assert(contains_locally_owned_cells<MeshType>(ghost_cell_layer_within_distance) == false,
             ExcMessage("Ghost cells within layer_thickness contains locally owned cells"));
      Assert(contains_artificial_cells<MeshType>(ghost_cell_layer_within_distance) == false,
             ExcMessage("Ghost cells within layer_thickness contains locally owned cells"));

      return ghost_cell_layer_within_distance;
    }


  } // GridTools

}

#endif /* __dealii_qc_utility_h */
