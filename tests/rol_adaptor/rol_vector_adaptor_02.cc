
#include <cmath>
#include <iostream>
#include <sstream>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/optimization/rol/vector_adaptor.h>

#include <deal.II-qc/potentials/pair_lj_cut.h>

#include <ROL_Objective.hpp>
#include <ROL_Algorithm.hpp>
#include <ROL_LineSearchStep.hpp>
#include <ROL_StatusTest.hpp>
#include <Teuchos_GlobalMPISession.hpp>

using namespace dealii;
using namespace dealiiqc;

using VectorType = typename dealii::Vector<double>;


// Use ROL to minimize the energy of two atoms where one atom is being fixed
// and their interaction is given by a Lennard-Jones potential.

//
// Location of the point `a` is fixed, and epsilon and rmin values are given for
// Lennard-Jones potential. Using ROL and VectorAdaptor find the point b such
// that the energy of atoms at `a` and `b` points is minimum.
//
//    *-----------------*
//    a                 b
//


template<class Real=double, typename Xprim=Rol::VectorAdaptor<VectorType> >
class Objective_LJ : public ROL::Objective<Real>
{

private:
  Real epsilon, rmin;
  const dealii::Point<3> a;

  Teuchos::RCP<const dealii::Vector<Real> >
  get_rcp_to_VectorType (const ROL::Vector<Real> &x)
  {
    return (Teuchos::dyn_cast<const Xprim>(x)).getVector();
  }

  Teuchos::RCP<dealii::Vector<Real> >
  get_rcp_to_VectorType (ROL::Vector<Real> &x)
  {
    return (Teuchos::dyn_cast<Xprim>(x)).getVector();
  }

public:

  Objective_LJ (Real  e=1.,
                Real  rmin=1.,
                const dealii::Point<3> &a=dealii::Point<3>(1.,1.,1.))
    :
    epsilon(e), rmin(rmin), a(a)
  {}

  Real value (const ROL::Vector<Real> &x,
              Real                    &tol)
  {
    Assert (x.dimension()==3,
            ExcInternalError());

    Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
    dealii::Point<3> b((*xp)[0], (*xp)[1], (*xp)[2]);
    dealii::Point<3> rel(b-a);

    const double r_square = rel.square();

    std::vector<Real> lj_params = {epsilon, rmin};

    Potential::PairLJCutManager lj (20);
    lj.declare_interactions (0,
                             1,
                             Potential::InteractionTypes::LJ,
                             lj_params);

    const std::pair<Real, Real> energy =
      lj.energy_and_gradient<false> ( 0, 1, r_square);

    return energy.first;
  }

  void gradient (ROL::Vector<Real>       &g,
                 const ROL::Vector<Real> &x,
                 Real                    &tol)
  {
    Teuchos::RCP<const VectorType> xp = this->get_rcp_to_VectorType(x);
    Teuchos::RCP<VectorType> gp = this->get_rcp_to_VectorType(g);
    //Teuchos::rcp_const_cast< dealii::Vector<Real> >(getVector<Xdual>(g));

    dealii::Point<3> b((*xp)[0], (*xp)[1], (*xp)[2]);
    dealii::Point<3> rel(b-a);

    const double r_square = rel.square();

    std::vector<Real> lj_params = {epsilon, rmin};

    Potential::PairLJCutManager lj (20);
    lj.declare_interactions (0,
                             1,
                             Potential::InteractionTypes::LJ,
                             lj_params);

    const std::pair<Real, Real> energy_and_gradient =
      lj.energy_and_gradient<true> ( 0, 1, r_square);

    const Real gfactor = energy_and_gradient.second;
    (*gp)[0] = gfactor * rel[0];
    (*gp)[1] = gfactor * rel[1];
    (*gp)[2] = gfactor * rel[1];
  }

};


int main (int argc, char **argv)
{
  typedef double RealT;

  try
    {
      Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);

      // Initial guess of the Point b.
      dealii::Point<3> b (2.2, 0.8, 1.4);

      Objective_LJ<RealT> lj(1.,1., dealii::Point<3>(1.,1.,1.) );

      //Teuchos::GlobalMPISession mpiSession(&argc, &argv);
      Teuchos::RCP<std::ostream> outStream = Teuchos::rcp(&std::cout, false);
      Teuchos::RCP<VectorType> x_rcp =
        Teuchos::rcp (new VectorType);

      x_rcp->reinit (3);

      (*x_rcp)[0] = b[0];
      (*x_rcp)[1] = b[1];
      (*x_rcp)[2] = b[2];

      Rol::VectorAdaptor<VectorType> x(x_rcp);

      Teuchos::ParameterList parlist;
      // Set parameters.
      parlist.sublist("Secant").set("Use as Preconditioner", false);
      // Define algorithm.
      ROL::Algorithm<RealT> algo("Line Search", parlist);

      // Run Algorithm
      algo.run(x, lj, true, *outStream);

      Teuchos::RCP<const VectorType> xg = x.getVector();
      std::cout << "The solution to lj minimization problem is: ";
      std::cout << (*xg)[0] << " " << (*xg)[1] << " " << (*xg)[2] << std::endl;

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      throw;
    }

  return 0;
}
