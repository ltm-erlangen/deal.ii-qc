
#include <deal.II-qc/configure/configure_qc.h>

namespace dealiiqc
{
  using namespace dealii;

  // Initialize dimension to a default unusable value
  // Imposes user to `set Dimension`
  ConfigureQC::ConfigureQC( std::shared_ptr<std::istream> is)
    :
    dimension(0),
    input_stream(is)
  {
    AssertThrow( *input_stream, ExcIO() );
    ParameterHandler prm;
    declare_parameters(prm);
    prm.parse_input (*input_stream,"dummy","#end-of-parameter-section");
    parse_parameters(prm);
  }

  unsigned int ConfigureQC::get_dimension() const
  {
    return dimension;
  }

  template<>
  std::shared_ptr<const Geometry::Base<1>> ConfigureQC::get_geometry() const
  {
    AssertThrow (dimension == 1, ExcInternalError());
    Assert(geometry_1d, ExcInternalError());
    return geometry_1d;
  }

  template<>
  std::shared_ptr<const Geometry::Base<2>> ConfigureQC::get_geometry() const
  {
    AssertThrow (dimension == 2, ExcInternalError());
    Assert(geometry_2d, ExcInternalError());
    return geometry_2d;
  }

  template<>
  std::shared_ptr<const Geometry::Base<3>> ConfigureQC::get_geometry() const
  {
    AssertThrow (dimension == 3, ExcInternalError());
    Assert(geometry_3d, ExcInternalError());
    return geometry_3d;
  }

  std::string ConfigureQC::get_atom_data_file () const
  {
    return atom_data_file;
  }

  std::shared_ptr<std::istream> ConfigureQC::get_stream() const
  {
    return input_stream;
  }

  double ConfigureQC::get_ghost_cell_layer_thickness() const
  {
    return ghost_cell_layer_thickness;
  }

  double ConfigureQC::get_maximum_cutoff_radius() const
  {
    return maximum_cutoff_radius;
  }

  double ConfigureQC::get_cluster_radius() const
  {
    return cluster_radius;
  }

  std::shared_ptr<Potential::PairBaseManager>
  ConfigureQC::get_potential() const
  {
    Assert (pair_potential, ExcInternalError());
    return pair_potential;
  }

  template<int dim>
  std::shared_ptr<const Cluster::WeightsByBase<dim> >
  ConfigureQC::get_cluster_weights() const
  {
    AssertDimension(dim, dimension);

    if (cluster_weights_type == "Cell")
      // It would be cleaner to store this object as a member variable, but then
      // we either need to make ConfigureQC templated with dim, or keep around
      // three different shared pointers.
      return std::make_shared<const Cluster::WeightsByCell<dim> >
             (cluster_radius);
    else
      AssertThrow (false, ExcInternalError());

    return NULL;
  }



  void ConfigureQC::declare_parameters( ParameterHandler &prm )
  {
    // TODO: Write intput file name to the screen
    //deallog << std::endl << "Parsing qc input file " << filename << std::endl
    //        << "for a " << dim << " dimensional simulation. " << std::endl;

    prm.declare_entry("Dimension", "2",
                      Patterns::Integer(1,3),
                      "Dimensionality of the problem ");

    Geometry::declare_parameters(prm);

    // TODO: Declare atom information
    // Use LAMMPS-like atom data file
    prm.enter_subsection ("Configure atoms");
    {
      prm.declare_entry("Atom data file", "",
                        Patterns::Anything(),
                        "Name of the atom data file "
                        "that is compatible with LAMMPS");
      prm.declare_entry("Maximum cutoff radius", "5.9",
                        Patterns::Double(0),
                        "Maximum of all the cutoff radii "
                        "plus a skin thickness "
                        "used to update the neighbor lists "
                        "of atoms");
      prm.declare_entry("Pair potential type", "LJ",
                        Patterns::Selection("Coulomb Wolf|LJ"),
                        "Pairwise interactions type of the "
                        "pair potential energy function");
      prm.declare_entry("Pair global coefficients", ".90",
                        Patterns::List(Patterns::Anything(),1),
                        "Comma separated global coefficient values for the "
                        "provided pair potential type."
                        "Coulomb Wolf: alpha and cutoff radius."
                        "LJ: cutoff radius ");
      prm.declare_entry("Pair specific coefficients", "0, 0, .8, 1.1;",
                        Patterns::List(Patterns::List(Patterns::Anything(),
                                                      0,
                                                      std::numeric_limits<unsigned int>::max(),
                                                      ","),
                                       0,
                                       std::numeric_limits<unsigned int>::max(),
                                       ";"),
                        "Additional coefficients for a pair of atoms of "
                        "certain types. Depending on the specific pair "
                        "potential type this input may not be necessary. "
                        "For the pair potential type: Coulomb Wolf, the pair "
                        "specific coefficients are not necessary."
                        "For the pair potential type: LJ, the first two "
                        "entries are the atom types (therefore of type "
                        "unsigned int) and the remaining two are epsilon and "
                        "alpha LJ parameters, respectively.");
    }
    prm.leave_subsection ();
    prm.enter_subsection ("Configure QC");
    {
      prm.declare_entry("Ghost cell layer thickness", "6.0",
                        Patterns::Double(0),
                        "The maximum distance from the locally "
                        "owned cells of each MPI process to to build "
                        "a layer of ghost cells needed for non-local energy "
                        "evaluation.");
      prm.declare_entry("Cluster radius", "2.0",
                        Patterns::Double(0),
                        "Cluster radius used in "
                        "QC simulation");
      prm.declare_entry("Cluster weights by type", "Cell",
                        Patterns::Selection("Cell"),
                        "Select the way how cluster "
                        "weights are computed for "
                        "cluster atoms.");
    }
    prm.leave_subsection ();

    // TODO: Declare Run 0
    //       Compute energy and force at the initial configuration.

  }



  void ConfigureQC::parse_parameters( ParameterHandler &prm )
  {
    dimension = prm.get_integer("Dimension");

    if (dimension==3)
      {
        geometry_3d = Geometry::parse_parameters_and_get_geometry<3>(prm);
        Assert( !geometry_1d, ExcInternalError());
        Assert( !geometry_2d, ExcInternalError());
      }
    else if (dimension==2)
      {
        geometry_2d = Geometry::parse_parameters_and_get_geometry<2>(prm);
        Assert( !geometry_1d, ExcInternalError());
        Assert( !geometry_3d, ExcInternalError());
      }
    else if (dimension==1)
      {
        geometry_1d = Geometry::parse_parameters_and_get_geometry<1>(prm);
        Assert( !geometry_2d, ExcInternalError());
        Assert( !geometry_3d, ExcInternalError());
      }
    else
      AssertThrow (false, ExcNotImplemented());

    prm.enter_subsection("Configure atoms");
    {
      atom_data_file = prm.get("Atom data file");
      maximum_cutoff_radius = prm.get_double("Maximum cutoff radius");

      const std::string pair_potential_type = prm.get("Pair potential type");

      std::vector<double> global_coeffs;
      {
        const std::vector<std::string> tmp =
          dealii::Utilities::split_string_list(prm.get("Pair global coefficients"), ',');
        for ( const auto &c : tmp)
          global_coeffs.push_back(dealii::Utilities::string_to_double(c));
      }

      const std::vector<std::vector<std::string> > list_of_coeffs_per_type =
        Utilities::split_list_of_string_lists(prm.get("Pair specific coefficients"),';',',');

      if (pair_potential_type == "Coulomb Wolf")
        {
          AssertThrow (global_coeffs.size()==2,
                       ExcMessage("Invalid Pair global coefficients provided "
                                  "for the Pair potential type: "
                                  "Coulomb Wolf."));
          AssertThrow (global_coeffs[1] < maximum_cutoff_radius,
                       ExcMessage("Maximum cutoff radius should be more than or "
                                  "equal to the provided cutoff radius."));
          pair_potential =
            std::make_shared<Potential::PairCoulWolfManager> (global_coeffs[0],
                                                              global_coeffs[1]);
        }
      else if (pair_potential_type == "LJ")
        {
          AssertThrow (global_coeffs.size()==1,
                       ExcMessage("Invalid Pair global coefficients provided "
                                  "for the Pair potential type: LJ."));
          AssertThrow (global_coeffs[0] < maximum_cutoff_radius,
                       ExcMessage("Maximum cutoff radius should be more than or "
                                  "equal to the provided cutoff radius."));
          pair_potential =
            std::make_shared<Potential::PairLJCutManager> (global_coeffs[0]);

          for (const auto &specific_coeffs : list_of_coeffs_per_type)
            {
              // Pair specific coefficients = 0, 1, 2.5, 1.0
              // atom type 0 and 1 interact with epsilon = 2.5 and r_m = 1.
              AssertThrow (specific_coeffs.size() == 4,
                           ExcMessage("Only two specific coefficients should be "
                                      "provided the first element being "
                                      "epsilon and second being r_m as "));

              const types::atom_type
              i = dealii::Utilities::string_to_int(specific_coeffs[0]),
              j = dealii::Utilities::string_to_int(specific_coeffs[1]);

              std::vector<double> coeffs(2);
              coeffs[0] = dealii::Utilities::string_to_double(specific_coeffs[2]);
              coeffs[1] = dealii::Utilities::string_to_double(specific_coeffs[3]);

              (std::static_pointer_cast<Potential::PairLJCutManager>
               (pair_potential))->declare_interactions (i,
                                                        j,
                                                        Potential::InteractionTypes::LJ,
                                                        coeffs);
            }
        }
      else
        {
          AssertThrow (false, ExcInternalError());
        }

    }
    prm.leave_subsection();

    prm.enter_subsection("Configure QC");
    {
      ghost_cell_layer_thickness = prm.get_double("Ghost cell layer thickness");

      Assert (maximum_cutoff_radius < ghost_cell_layer_thickness,
              ExcMessage("Ghost cell layer thickness should be more than or "
                         "equal to the Maximum cutoff radius."));

      cluster_radius = prm.get_double( "Cluster radius");
      cluster_weights_type = prm.get("Cluster weights by type");
    }
    prm.leave_subsection();
  }



  template std::shared_ptr<const Geometry::Base<1>> ConfigureQC::get_geometry() const;
  template std::shared_ptr<const Geometry::Base<2>> ConfigureQC::get_geometry() const;
  template std::shared_ptr<const Geometry::Base<3>> ConfigureQC::get_geometry() const;
  template std::shared_ptr<const Cluster::WeightsByBase<1>> ConfigureQC::get_cluster_weights() const;
  template std::shared_ptr<const Cluster::WeightsByBase<2>> ConfigureQC::get_cluster_weights() const;
  template std::shared_ptr<const Cluster::WeightsByBase<3>> ConfigureQC::get_cluster_weights() const;



} // namespace dealiiqc
