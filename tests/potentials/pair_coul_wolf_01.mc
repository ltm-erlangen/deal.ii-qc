/**
Maxima input script to test PairCoulWolf::energy_and_gradient() 
in pair_coul_wolf_01.cc

Algebraically defining shifted_energy and shifted_force;
Verified the algebraic result with that from [Wolf et al 1999]

Vishal Boddu 28.04.2017

*/

/* erfcc */
erfcc(r,alpha) := erfc(alpha*r)/r;

/* Differentiate erfcc */
derfcc(r,alpha) := diff( erfcc(r,alpha),r);

/* Differentiate ERFCC by hand*/
derfcc_explicit(r,alpha) := -erfc(alpha*r)/r^2 - 2*alpha*(%e^(-alpha^2*r^2))/(sqrt(%pi)*r);

/* shifted_energy and shifted_force */
shifted_energy(p,q,r,rc,alpha) := 14.399645*p*q*( erfcc(r,alpha) - limit( erfcc(r, alpha), r, rc)  );
shifted_force(p,q,r,rc,alpha) := -14.399645*p*q*( derfcc_explicit(r,alpha) - limit( derfcc_explicit(r,alpha), r, rc)  );

print("Energy case 1: ");
at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=.9,rc=.95,alpha=0.25]);
print("Force case 1: ");
at(shifted_force(p,q,r,rc,alpha), [p=1.,q=-1.,r=.9,rc=.95,alpha=0.25]);
float(%);
print("Energy case 2: ");
at(shifted_energy(p,q,r,rc,alpha), [p=1.,q=-1.,r=1.,rc=1.75,alpha=0.25]);
print("Force case 2: ");
(at(shifted_force(p,q,r,rc,alpha), [p=1.,q=-1.,r=1.,rc=1.75,alpha=0.25]));
float(%);

