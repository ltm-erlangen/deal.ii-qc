
#include <deal.II-qc/configure/configure_qc.h>


DEAL_II_QC_NAMESPACE_OPEN


using namespace dealii;

// Initialize dimension to a default unusable value
// Imposes user to `set Dimension`
ConfigureQC::ConfigureQC (std::shared_ptr<std::istream> is)
  :
  dimension(0),
  input_stream(is)
{
  AssertThrow (*input_stream, ExcIO());
  ParameterHandler prm;
  declare_parameters(prm);
  prm.parse_input (*input_stream, "dummy", "#end-of-parameter-section");
  parse_parameters(prm);
}

unsigned int ConfigureQC::get_dimension() const
{
  return dimension;
}

std::string ConfigureQC::get_pair_potential_type() const
{
  return pair_potential_type;
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

template<int dim, int atomicity, int spacedim>
std::shared_ptr<Cluster::WeightsByBase<dim, atomicity, spacedim> >
ConfigureQC::get_cluster_weights() const
{
  AssertDimension(dim, dimension);

  // It would be cleaner to store this object as a member variable, but then
  // we either need to make ConfigureQC templated with dim, or keep around
  // three different shared pointers.
  if (cluster_weights_type == "Cell")
    return
      std::make_shared<Cluster::WeightsByCell<dim, atomicity, spacedim>>
      (cluster_radius, maximum_cutoff_radius);

  else if (cluster_weights_type == "LumpedVertex")
    return
      std::make_shared<Cluster::WeightsByLumpedVertex<dim, atomicity, spacedim>>
      (cluster_radius, maximum_cutoff_radius);

  else if (cluster_weights_type == "SamplingPoints")
    return
      std::make_shared<Cluster::WeightsBySamplingPoints<dim, atomicity, spacedim>>
      (cluster_radius, maximum_cutoff_radius);

  else
    AssertThrow (false, ExcInternalError());

  return NULL;
}



std::map<unsigned int, std::vector<std::string> >
ConfigureQC::get_boundary_functions() const
{
  return boundary_ids_to_function_expressions;
}



std::map<std::pair<unsigned int, bool>, std::string>
ConfigureQC::get_external_potential_fields() const
{
  return external_potential_field_expressions;
}



std::string ConfigureQC::get_minimizer_name() const
{
  return minimizer;
}



double ConfigureQC::get_time_step() const
{
  return time_step;
}



unsigned int ConfigureQC::get_n_time_steps() const
{
  return n_time_steps;
}



ConfigureQC::SolverControlParameters
ConfigureQC::get_solver_control_parameters () const
{
  return solver_control_parameters;
}



ConfigureQC::FireParameters ConfigureQC::get_fire_parameters () const
{
  return fire_parameters;
}



void ConfigureQC::declare_parameters (ParameterHandler &prm)
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
                      "---"
                      "For the pair potential type: Coulomb Wolf, the pair "
                      "specific coefficients are not necessary."
                      "---"
                      "For the pair potential type: LJ, the first two "
                      "entries are the atom types and the remaining two are"
                      "epsilon and rm LJ parameters, respectively."
                      "Note that the atom data counts the atom types from 1 "
                      "but deal.II-qc from 0. Therefore atom type 2 in the "
                      "atom data is atom type 1 in deal.II-qc.");
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
                      Patterns::Selection("Cell|LumpedVertex|SamplingPoints"),
                      "Select the way how cluster "
                      "weights are computed for "
                      "cluster atoms.");
  }
  prm.leave_subsection ();

  for (unsigned int
       boundary_id = 0;
       boundary_id < max_n_boundaries;
       boundary_id++)
    {
      prm.enter_subsection ("boundary_" +
                            dealii::Utilities::int_to_string(boundary_id));
      {
        prm.declare_entry("Function expressions",
                          ", , ,",
                          Patterns::List(Patterns::Anything(),
                                         0,
                                         std::numeric_limits<unsigned int>::max(),
                                         ","),
                          "Function expressions that describes the boundary "
                          "condition for all the components of the "
                          "vector-valued solution at the current boundary id."
                          "Each expression should end with a comma."
                          "If the function expression of a particular "
                          "component is empty then the corresponding component "
                          "of the vector-valued solution at the boundary "
                          "corresponding to the current boundary id is not "
                          "constrained. This is then equivalent to having the "
                          "corresponding entry in the component mask set to "
                          "false. In this way only certain components of the "
                          "vector-valued solution at the boundary can be "
                          "constrained."
                          "For example:"
                          "If the solution of the problem being solved is "
                          "vector valued displacements in two dimensions, "
                          "the component mask with true false implies that "
                          "only the first component of the displacements is "
                          "being constrained to the given function "
                          "describing the boundary condition."
                          "Non empty function expressions would be passed in "
                          "to initialize valid Function objects using "
                          "FunctionParser."
                          "---"
                          "For example:"
                          "The expression 0 implies that the current "
                          "component of the current boundary id is subjected "
                          "to Homogeneous Dirichlet boundary condition.");
      }
      prm.leave_subsection ();
    }

  for (unsigned int
       material_id = 0;
       material_id < max_n_material_ids;
       material_id++)
    {
      prm.enter_subsection ("ext_potential_material_id_" +
                            dealii::Utilities::int_to_string(material_id));
      {
        prm.declare_entry ("Is electric field",
                           "false",
                           Patterns::Bool(),
                           "Specify whether the potential field is an electric "
                           "potential field.");
        prm.declare_entry("Function expression",
                          "",
                          Patterns::Anything(),
                          "Function expressions that describes the external "
                          "potential field applied to the domain with current "
                          "material id."
                          "If the function expression empty then there "
                          "is no external potential field applied to the "
                          "domain with current material id."
                          "---"
                          "In three dimensions, the function expression "
                          "consists of 4 variables — namely x, y, z and t."
                          "The first three variables denote the spatial "
                          "location of a point where the value is to be "
                          "computed followed by the time variable.");
      }
      prm.leave_subsection ();
    }

  prm.enter_subsection ("Minimizer settings");
  {
    SolverControl::declare_parameters(prm);

    prm.declare_entry ("Minimizer",
                       "FIRE",
                       Patterns::Selection("FIRE"/* TODO Add more minimizers*/),
                       "Choose minimizer.");

    prm.enter_subsection ("FIRE");
    {
      prm.declare_entry ("Initial time step",
                         "0.2",
                         Patterns::Double(1e-16),
                         "FIRE minimizer initial time step.");
      prm.declare_entry ("Maximum time step",
                         "0.5",
                         Patterns::Double(1e-16),
                         "FIRE minimizer maximum time step.");
      prm.declare_entry ("Maximum linfty norm",
                         "0.5",
                         Patterns::Double(1e-16),
                         "FIRE minimizer maximum linfty norm. This refers "
                         "to the maximum allowable change in any degree of "
                         "freedom.");
    }
    prm.leave_subsection ();
  }
  prm.leave_subsection ();

  prm.enter_subsection("Quasi-static loading");
  {
    prm.declare_entry("Number of time steps",
                      "0",
                      Patterns::Integer(0),
                      "The number of load steps to be performed during the "
                      "quasi-static loading process.");
    prm.declare_entry("Time step size",
                      "0.1",
                      Patterns::Double(0),
                      "The time interval between load steps in the "
                      "quasi-static loading process.");
  }
  prm.leave_subsection();

  // TODO: Declare Run 0
  //       Compute energy and force at the initial configuration.

}



void ConfigureQC::parse_parameters (ParameterHandler &prm)
{
  dimension = prm.get_integer("Dimension");

  if (dimension==3)
    {
      geometry_3d = Geometry::parse_parameters_and_get_geometry<3>(prm);
      Assert (!geometry_1d, ExcInternalError());
      Assert (!geometry_2d, ExcInternalError());
    }
  else if (dimension==2)
    {
      geometry_2d = Geometry::parse_parameters_and_get_geometry<2>(prm);
      Assert (!geometry_1d, ExcInternalError());
      Assert (!geometry_3d, ExcInternalError());
    }
  else if (dimension==1)
    {
      geometry_1d = Geometry::parse_parameters_and_get_geometry<1>(prm);
      Assert (!geometry_2d, ExcInternalError());
      Assert (!geometry_3d, ExcInternalError());
    }
  else
    AssertThrow (false, ExcNotImplemented());

  prm.enter_subsection("Configure atoms");
  {
    atom_data_file = prm.get("Atom data file");
    maximum_cutoff_radius = prm.get_double("Maximum cutoff radius");

    pair_potential_type = prm.get("Pair potential type");

    std::vector<double> global_coeffs;
    {
      const std::vector<std::string> tmp =
        dealii::Utilities::
        split_string_list(prm.get("Pair global coefficients"), ',');

      for ( const auto &c : tmp)
        global_coeffs.push_back(dealii::Utilities::string_to_double(c));
    }

    const std::vector<std::vector<std::string> > list_of_coeffs_per_type =
      Utilities::
      split_list_of_string_lists(prm.get("Pair specific coefficients"),';',',');

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
             (pair_potential))->
            declare_interactions (i,
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

  for (unsigned int
       boundary_id = 0;
       boundary_id < max_n_boundaries;
       boundary_id++)
    {
      prm.enter_subsection("boundary_" +
                           dealii::Utilities::int_to_string(boundary_id));
      {
        const std::vector<std::string> function_expressions =
          dealii::Utilities::split_string_list (prm.get("Function expressions"),
                                                ',');

        bool ignore_boundary_id = true;

        for (const auto &expression : function_expressions)
          ignore_boundary_id &= expression.empty();

        // If any of the function expressions are not empty,
        // then consider preparing function expression for this boundary id.
        if (!ignore_boundary_id)
          boundary_ids_to_function_expressions[boundary_id] =
            function_expressions;
      }
      prm.leave_subsection();
    }

  for (unsigned int
       material_id = 0;
       material_id < max_n_material_ids;
       material_id++)
    {
      prm.enter_subsection("ext_potential_material_id_" +
                           dealii::Utilities::int_to_string(material_id));
      {
        const std::pair<unsigned int, bool> unique_key =
          std::make_pair(material_id, prm.get_bool("Is electric field"));

        const std::string function_expression =
          prm.get("Function expression");

        // If the provided function expression is not empty,
        // then prepare potential field function expression for this
        //      material id.
        if (!function_expression.empty())
          external_potential_field_expressions[unique_key] =
            function_expression;
      }
      prm.leave_subsection();
    }

  prm.enter_subsection ("Minimizer settings");
  {
    solver_control_parameters.max_steps     = prm.get_integer("Max steps");
    solver_control_parameters.tolerance     = prm.get_double("Tolerance");
    solver_control_parameters.log_history   = prm.get_bool("Log history");
    solver_control_parameters.log_result    = prm.get_bool("Log result");
    solver_control_parameters.log_frequency = prm.get_integer("Log frequency");

    minimizer              = prm.get("Minimizer");

    prm.enter_subsection("FIRE");
    {
      fire_parameters.initial_time_step = prm.get_double("Initial time step");
      fire_parameters.maximum_time_step = prm.get_double("Maximum time step");
      fire_parameters.maximum_linfty_norm = prm.get_double("Maximum linfty norm");
    }
    prm.leave_subsection();
  }
  prm.leave_subsection ();

  prm.enter_subsection("Quasi-static loading");
  {
    n_time_steps = prm.get_integer("Number of time steps");
    time_step    = prm.get_double("Time step size");
  }
  prm.leave_subsection();
}



template
std::shared_ptr<const Geometry::Base<1>> ConfigureQC::get_geometry() const;
template
std::shared_ptr<const Geometry::Base<2>> ConfigureQC::get_geometry() const;
template
std::shared_ptr<const Geometry::Base<3>> ConfigureQC::get_geometry() const;

#define SINGLE_CONFIGURE_QC_INSTANTIATION(DIM, ATOMICITY, SPACEDIM) \
  template                                                          \
  std::shared_ptr<Cluster::WeightsByBase<DIM, ATOMICITY, SPACEDIM>> \
  ConfigureQC::get_cluster_weights() const;                         \
   
#define CONFIGURE_QC(R, X)                       \
  BOOST_PP_IF(IS_DIM_LESS_EQUAL_SPACEDIM X,      \
              SINGLE_CONFIGURE_QC_INSTANTIATION, \
              BOOST_PP_TUPLE_EAT(3)) X           \
   
// ConfigureQC::get_cluster_weights instantiations
INSTANTIATE_CLASS_WITH_DIM_ATOMICITY_AND_SPACEDIM(CONFIGURE_QC)

#undef SINGLE_CONFIGURE_QC_INSTANTIATION
#undef CONFIGURE_QC


DEAL_II_QC_NAMESPACE_CLOSE
