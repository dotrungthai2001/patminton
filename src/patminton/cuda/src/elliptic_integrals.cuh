#ifndef __ELLIPTIC_INTEGRALS__
#define __ELLIPTIC_INTEGRALS__

#define EPSILON 2.220446049250313e-16
#define ROOT_EPSILON 1.4901161193847656e-8
#define MAX_ITER 10000
#define MAX_VALUE 1.7976931348623157e308
#define MIN_VALUE 2.2250738585072014e-308

#include "math.h"
// #include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// Carlson elementary auxiliary:  Rc(x, y)
//
//   Rc(x, y) = Rf(x, y, y)
//            = (1/2) integral_0^inf  dt / (sqrt(t+x) (t+y))
//
//   Closed form:
//     x < y :  acos(sqrt(x/y)) / sqrt(y-x)   (equivalently atan(sqrt((y-x)/x))/sqrt(y-x))
//     x > y :  acosh(sqrt(x/y)) / sqrt(x-y)  (equivalently atanh(sqrt((x-y)/x))/sqrt(x-y))
//     x == y:  1/sqrt(x)
// ─────────────────────────────────────────────────────────────────────────────

__device__ inline void swap_device(double &x, double &y) {
    double z = x ;
    x = y ;
    y = z ;
}

__device__ double Rc(double x, double y)
{
    if(x < 0)
        return INFINITY ;
    if(y == 0)
        return INFINITY ;

    // for y < 0, the integral is singular, return Cauchy principal value
    double prefix, result;
    if(y < 0) {
        prefix = sqrt(x / (x - y));
        x = x - y;
        y = -y;
    } else {
       prefix = 1;
    }

    if(x == 0) {
       result = M_PI_2 / sqrt(y);
    } else if(x == y) {
       result = 1 / sqrt(x);
    }
    else if(y > x) {
       result = atan(sqrt((y - x) / x)) / sqrt(y - x);
    } else{
       if(y / x > 0.5){
          double arg = sqrt((x - y) / x);
          result = atanh(arg) / sqrt(x - y) ;
       } else {
          result = log((sqrt(x) + sqrt(x - y)) / sqrt(y)) / sqrt(x - y);
       }
    }
    return prefix * result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Rf(x, y, z)  —  Carlson completely symmetric integral of 1st kind
//
//   Rf(x,y,z) = (1/2) integral_0^inf  dt / sqrt((t+x)(t+y)(t+z))
//
// Requires x,y,z >= 0, at most one zero.
// ─────────────────────────────────────────────────────────────────────────────

 __device__ double Rf(double x, double y, double z)
 {
    if(x < 0 || y < 0 || z < 0){
        return INFINITY ;
    }

    if(x + y == 0 || y + z == 0 || z + x == 0) {
        return INFINITY ;

    }
    //
    // Special cases from http://dlmf.nist.gov/19.20#i
    //
    if(x == y)  {
       if(x == z){
          return 1 / sqrt(x);
       } else {
          // 2 equal, x and y:
          if(z == 0)
             return M_PI / (2 * sqrt(x));
          else
             return Rc(z, x);
       }
    }

    if(x == z) {
       if(y == 0)
          return M_PI / (2*sqrt(x)) ;
       else
          return Rc(y, x);
    }

    if(y == z) {
       if(x == 0)
          return M_PI / (2*sqrt(y)) ;
       else
          return Rc(x, y);
    }

    if(x == 0) {
         swap_device(x,z) ;
    } else if(y == 0) {
         swap_device(y,z) ;
    }

    if(z == 0)  {
       //
       // Special case for one value zero:
       //
       double xn = sqrt(x);
       double yn = sqrt(y);

       while(fabs(xn - yn) >= 2.7 * ROOT_EPSILON * fabs(xn))
       {
           double t = sqrt(xn * yn);
           xn = (xn + yn) / 2;
           yn = t;
       }
       return M_PI / (xn + yn);
    }

    double xn = x;
    double yn = y;
    double zn = z;
    double An = (x + y + z) / 3;
    double A0 = An;
    double Q = pow(3 * EPSILON, -1. / 8) * fmax(fmax(fabs(An-xn), fabs(An-yn)), fabs(An-zn)) ;
    double fn = 1;

    // duplication
    unsigned k = 1;
    for(; k < MAX_ITER ; ++k){
        double root_x = sqrt(xn);
        double root_y = sqrt(yn);
        double root_z = sqrt(zn);
        double lambda = root_x * root_y + root_x * root_z + root_y * root_z;
        An = (An + lambda) / 4;
        xn = (xn + lambda) / 4;
        yn = (yn + lambda) / 4;
        zn = (zn + lambda) / 4;
        Q /= 4;
        fn *= 4;
        if(Q < fabs(An))
            break;
    }

    double X = (A0 - x) / (An * fn);
    double Y = (A0 - y) / (An * fn);
    double Z = -X - Y;

    // Taylor series expansion to the 7th order
    double E2 = X * Y - Z * Z;
    double E3 = X * Y * Z;
    return (1 + E3 * (1.0 / 14 + 3 * E3 / 104) + E2 * (-1.0 / 10 + E2 / 24 - (3 * E3) / 44 - 5 * E2 * E2 / 208 + E2 * E3 / 16)) / sqrt(An);
 }


// ─────────────────────────────────────────────────────────────────────────────
// Rd(x, y, z)  —  Carlson degenerate symmetric integral of 2nd kind
//
//   Rd(x,y,z) = (3/2) integral_0^inf  dt / (sqrt((t+x)(t+y)) (t+z)^(3/2))
//
// Requires x,y >= 0 (at most one zero), z > 0.
// ─────────────────────────────────────────────────────────────────────────────


__device__ double Rd(double x, double y, double z) {
   if(x < 0)
      return INFINITY ;
   if(y < 0)
      return INFINITY ;
   if(z <= 0)
       return INFINITY ;
   if(x + y == 0)
       return INFINITY ;

   //
   // Special cases from http://dlmf.nist.gov/19.20#iv
   //
   if(x == z)
        swap_device(x, y);

   if(y == z) {
      if(x == y) {
         return 1 / (x * sqrt(x));
      } else if(x == 0) {
         return 3 * M_PI / (4 * y * sqrt(y));
      } else {
         if( (fmin(x,y) / fmax(x,y)) > 1.3 )
            return 3 * (Rc(x, y) - sqrt(x) / y) / (2 * (y - x));
         // Otherwise fall through to avoid cancellation in the above (RC(x,y) -> 1/x^0.5 as x -> y)
      }
   }

   if(x == y) {
      if( (fmin(x, z) / fmax(x, z)) > 1.3)
         return 3 * (Rc(z, x) - 1 / sqrt(z)) / (z - x);
      // Otherwise fall through to avoid cancellation in the above (RC(x,y) -> 1/x^0.5 as x -> y)
   }

   if(y == 0)
       swap_device(x, y);

   if(x == 0) {
      //
      // Special handling for common case, from
      // Numerical Computation of Real or Complex Elliptic Integrals, eq.47
      //
      double xn = sqrt(y);
      double yn = sqrt(z);
      double x0 = xn;
      double y0 = yn;
      double sum = 0;
      double sum_pow = 0.25f;

      while(fabs(xn - yn) >= 2.7 * ROOT_EPSILON * fabs(xn)) {
          double t = sqrt(xn * yn);
          xn = (xn + yn) / 2;
          yn = t;
          sum_pow *= 2;
          sum += sum_pow * (xn-yn)*(xn-yn) ;
      }
      double RF = M_PI / (xn + yn);
      //
      // This following calculation suffers from serious cancellation when y ~ z
      // unless we combine terms.  We have:
      //
      // ( ((x0 + y0)/2)^2 - z ) / (z(y-z))
      //
      // Substituting y = x0^2 and z = y0^2 and simplifying we get the following:
      //
      double pt = (x0 + 3 * y0) / (4 * z * (x0 + y0));
      //
      // Since we've moved the denominator from eq.47 inside the expression, we
      // need to also scale "sum" by the same value:
      //
      pt -= sum / (z * (y - z));
      return pt * RF * 3;
   }

   double xn = x;
   double yn = y;
   double zn = z;
   double An = (x + y + 3 * z) / 5;
   double A0 = An;
   // This has an extra 1.2 fudge factor which is really only needed when x, y and z are close in magnitude:
   double Q = pow( EPSILON / 4, -1. / 8) * fmax(fmax(An-x, An-y), An - z ) * 1.2 ;

   double lambda, rx, ry, rz;
   unsigned k = 0;
   double fn = 1;
   double RD_sum = 0;

   for(; k < MAX_ITER ; ++k)
   {
      rx = sqrt(xn);
      ry = sqrt(yn);
      rz = sqrt(zn);
      lambda = rx * ry + rx * rz + ry * rz;
      RD_sum += fn / (rz * (zn + lambda));
      An = (An + lambda) / 4;
      xn = (xn + lambda) / 4;
      yn = (yn + lambda) / 4;
      zn = (zn + lambda) / 4;
      fn /= 4;
      Q /= 4;
      if(Q < An)
         break;
   }

   double X = fn * (A0 - x) / An;
   double Y = fn * (A0 - y) / An;
   double Z = -(X + Y) / 3;
   double E2 = X * Y - 6 * Z * Z;
   double E3 = (3 * X * Y - 8 * Z * Z) * Z;
   double E4 = 3 * (X * Y - Z * Z) * Z * Z;
   double E5 = X * Y * Z * Z * Z;

   double result = fn * pow(An, -3. / 2) *
      (1 - 3 * E2 / 14 + E3 / 6 + 9 * E2 * E2 / 88 - 3 * E4 / 22 - 9 * E2 * E3 / 52 + 3 * E5 / 26 - E2 * E2 * E2 / 16
      + 3 * E3 * E3 / 40 + 3 * E2 * E4 / 20 + 45 * E2 * E2 * E3 / 272 - 9 * (E3 * E4 + E2 * E5) / 68);
   result += 3 * RD_sum;

   return result;
}

// // ─────────────────────────────────────────────────────────────────────────────
// // Rj(x, y, z, p)  —  Carlson symmetric integral of 3rd kind
// //
// //   Rj(x,y,z,p) = (3/2) integral_0^inf  dt / (sqrt((t+x)(t+y)(t+z)) (t+p))
// //
// // Requires x,y,z >= 0 (at most one zero), p > 0.
// // ─────────────────────────────────────────────────────────────────────────────
// __device__   double Rj(double x, double y, double z, double p)
// {
//     constexpr double tol = 2e-16;
//     constexpr double c1  = 3.0 / 14.0;
//     constexpr double c2  = 1.0 / 3.0;
//     constexpr double c3  = 3.0 / 22.0;
//     constexpr double c4  = 3.0 / 26.0;

//     double sum = 0.0;
//     double fac = 1.0;
//     double dx, dy, dz, dp, A;
//     for (;;) {
//         double lx  = sqrt(x);
//         double ly  = sqrt(y);
//         double lz  = sqrt(z);
//         double lp  = sqrt(p);
//         double lam = lx*ly + ly*lz + lz*lx;
//         // Carlson (1995) eq. (2.17): accumulate Rc term
//         double alpha = p*(lx + ly + lz) + lx*ly*lz;
//         alpha *= alpha;
//         double beta  = p * (p + lam) * (p + lam);
//         sum += fac * Rc(alpha, beta);
//         fac *= 0.25;
//         x = (x + lam) * 0.25;
//         y = (y + lam) * 0.25;
//         z = (z + lam) * 0.25;
//         p = (p + lam) * 0.25;
//         A  = (x + y + z + 2.0*p) / 5.0;
//         dx = 1.0 - x/A;
//         dy = 1.0 - y/A;
//         dz = 1.0 - z/A;
//         dp = 1.0 - p/A;
//         if (fmax(fmax(fabs(dx), fabs(dy)), fabs(dz)) < tol)
//                 break;
//     }
//     double E2 = dx*dy + dx*dz + dy*dz - 3.0*dp*dp;
//     double E3 = dx*dy*dz + 2.0*E2*dp + 3.0*dp*(dx*dp + dy*dp + dz*dp
//                     - dx*dy - dx*dz - dy*dz);
//     double E4 = (2.0*dx*dy*dz + E2*dp + 3.0*(dx+dy+dz)*dp*dp) * dp;
//     double E5 = dx*dy*dz*dp*dp;
//     double poly = 1.0 - 3.0*c1*E2 + c2*E3 + (3.0/16.0)*E2*E2
//                       - 3.0*c3*E4 + 3.0*c4*E5 - (3.0/44.0)*E2*E3;
//     return 3.0*sum + fac*poly / (A * sqrt(A));
// }


// =============================================================================
//  Public API
// =============================================================================

// ── First kind  F(phi, k) ─────────────────────────────────────────────────────

/** Complete elliptic integral of the first kind  K(k) = F(pi/2, k) */
__device__   double ellint_1(double k) {
    if (abs(k) >= 1.0)
        return INFINITY ;
    return Rf(0.0, 1.0 - k*k, 1.0);
}

/**
 * Incomplete elliptic integral of the first kind
 *   F(phi, k) = integral_0^phi  dt / sqrt(1 - k² sin²t)
 *
 * @param k    modulus   |k| <= 1
 * @param phi  amplitude (radians, any real)
 */

// Elliptic integral (Legendre form) of the first kind
__device__ double ellint_1(double k, double phi) {
    bool invert = false;
    if(phi < 0) {
       phi = fabs(phi);
       invert = true;
    }

    double result;

    if(phi >= MAX_VALUE) {
        return INFINITY ;
    } else if(phi > 1 / EPSILON) {
       // Phi is so large that phi%pi is necessarily zero (or garbage),
       // just return the second part of the duplication formula:
       result = 2 * phi * ellint_1(k) / M_PI ;
    } else {
       // Carlson's algorithm works only for |phi| <= pi/2,
       // use the integrand's periodicity to normalize phi
       //
       // Xiaogang's original code used a cast to long long here
       // but that fails if T has more digits than a long long,
       // so rewritten to use fmod instead:
       //
       double rphi = fmod(phi, M_PI_2) ;
       double m = round((phi - rphi)/M_PI_2) ;

       int s = 1;
       if(fmod(m, 2.0) > 0.5 ) {
          m += 1;
          s = -1;
          rphi = M_PI_2 - rphi;
       }
       double sinp = sin(rphi);
       sinp *= sinp;
       if (sinp * k * k >= 1) {
           return INFINITY ;
       }

       double cosp = cos(rphi);
       cosp *= cosp;
       if(sinp > MIN_VALUE) {
          //
          // Use http://dlmf.nist.gov/19.25#E5, note that
          // c-1 simplifies to cot^2(rphi) which avoid cancellation:
          //
          double c = 1 / sinp;
          result = s * Rf(cosp / sinp, c - k * k, c);
       } else {
          result = s * sin(rphi);
       }

       if(m != 0) {
          result += m * ellint_1(k);
       }
    }
    return invert ? (-result) : result;
}


// ── Second kind  E(phi, k) ────────────────────────────────────────────────────

/**
 * Incomplete elliptic integral of the second kind
 *   E(phi, k) = integral_0^phi  sqrt(1 - k² sin²t)  dt
 *
 * @param k    modulus   |k| <= 1
 * @param phi  amplitude (radians, any real)
 */

/** Complete elliptic integral of the second kind  E(k) = E(pi/2, k) */
 __device__  double ellint_2(double k)
 {
     if (abs(k) > 1.0)
         return INFINITY ;
     if (abs(k) == 1.0)
         return 1 ;
     double k2 = k * k;

     return Rf(0.0, 1.0-k2, 1.0) - (k2/3.0) * Rd(0.0, 1.0-k2, 1.0);
 }

// Elliptic integral (Legendre form) of the second kind
__device__ double ellint_2(double k, double phi)
{

    bool invert = false;
    if (phi == 0)
       return 0;

    if(phi < 0) {
       phi = fabs(phi);
       invert = true;
    }

    double result;

    if(phi >= MAX_VALUE) {
        result = INFINITY ;
    } else if(phi > 1 / EPSILON) {
       // Phi is so large that phi%pi is necessarily zero (or garbage),
       // just return the second part of the duplication formula:
       result = 2 * phi * ellint_2(k) / M_PI ;
    } else if(k == 0) {
       return invert ? (-phi) : phi;
    } else if(fabs(k) == 1) {
       //
       // For k = 1 ellipse actually turns to a line and every pi/2 in phi is exactly 1 in arc length
       // Periodicity though is in pi, curve follows sin(pi) for 0 <= phi <= pi/2 and then
       // 2 - sin(pi- phi) = 2 + sin(phi - pi) for pi/2 <= phi <= pi, so general form is:
       //
       // 2n + sin(phi - n * pi) ; |phi - n * pi| <= pi / 2
       //
       int m = round(phi / M_PI) ;
       double remains = phi - m*M_PI ;
       double value = 2 * m + sin(remains);
       // negative arc length for negative phi
       return invert ? -value : value;
    } else {
       // Carlson's algorithm works only for |phi| <= pi/2,
       // use the integrand's periodicity to normalize phi
       //
       // Xiaogang's original code used a cast to long long here
       // but that fails if T has more digits than a long long,
       // so rewritten to use fmod instead:
       //
       double rphi = fmod(phi,M_PI_2) ;
       double m = round((phi - rphi)/M_PI_2) ;

       int s = 1;
       if(fmod(m, 2.0) > 0.5 ) {
          m += 1;
          s = -1;
          rphi = M_PI_2 - rphi;
       }

       double k2 = k * k;
       if( (rphi*rphi*rphi * k2 / 6) < (EPSILON * fabs(rphi) ) ) {
          // See http://functions.wolfram.com/EllipticIntegrals/EllipticE2/06/01/03/0001/
          result = s * rphi;
       } else {
          // http://dlmf.nist.gov/19.25#E10
          double sinp = sin(rphi);
          if (k2 * sinp * sinp >= 1){
              return INFINITY ;
          }
          double cosp = cos(rphi);
          double c = 1 / (sinp * sinp);
          double cm1 = cosp * cosp / (sinp * sinp);  // c - 1
          result = s * (Rf(cm1, c - k2, c) - (k2/3.0) * Rd(cm1, c - k2, c));
       }
       if(m != 0)
          result += m * ellint_2(k);
    }
    return invert ? (-result) : result;
}


// ── Third kind  Π(phi, n, k) ──────────────────────────────────────────────────

/**
 * Incomplete elliptic integral of the third kind
 *   Π(phi, n, k) = integral_0^phi  dt / ((1 - n sin²t) sqrt(1 - k² sin²t))
 *
 * @param k    modulus          |k| <= 1
 * @param n    characteristic   n*sin²(phi) must not equal 1
 * @param phi  amplitude        (radians, any real)
//  */
// __device__   double ellint_3(double k, double n, double phi)
// {
//     if (abs(k) > 1.0)
//         return INFINITY ;

//     if (phi < 0.0) return -ellint_3(k, n, -phi);

//     int  periods;
//     double rphi = period_reduce(phi, M_PI, periods);
//     bool   flip = (rphi > M_PI_2);
//     if (flip) rphi = M_PI - rphi;

//     double k2  = k * k;
//     double sp  = sin(rphi);
//     double cp  = cos(rphi);
//     double sp2 = sp * sp;
//     double p   = 1.0 - n * sp2;

//     if (abs(p) < 10.0 * 1e-16)
//         return INFINITY ;

//     // Π(phi,n,k) = sin(phi)*Rf(cos²φ, 1-k²sin²φ, 1)
//     //             +(n/3)*sin³(phi)*Rj(cos²φ, 1-k²sin²φ, 1, 1-n*sin²φ)
//     double core = sp * Rf(cp*cp, 1.0-k2*sp2, 1.0)
//                 + (n/3.0) * sp*sp2 * Rj(cp*cp, 1.0-k2*sp2, 1.0, p);

//     double Pic = Rf(0.0, 1.0-k2, 1.0)
//                + (n/3.0) * Rj(0.0, 1.0-k2, 1.0, 1.0-n);   // Π(n,k)

//     double base = flip ? 2.0*Pic - core : core;
//     return base + 2.0 * periods * Pic;
// }

// /** Complete elliptic integral of the third kind  Π(n, k) = Π(pi/2, n, k) */
// __device__   double ellint_3(double k, double n)
// {
//     if (abs(k) > 1.0)
//         return INFINITY ;

//     double k2 = k * k ;
//     return Rf(0.0, 1.0-k2, 1.0)
//          + (n/3.0) * Rj(0.0, 1.0-k2, 1.0, 1.0-n);
// }


// ── D integral  D(phi, k) ─────────────────────────────────────────────────────

/**
 * Incomplete elliptic integral D
 *   D(phi, k) = integral_0^phi  sin²t / sqrt(1 - k² sin²t)  dt
 *             = (F(phi,k) - E(phi,k)) / k²
 *
 * Uses the direct Rd form — numerically stable even near k = 0.
 *
 * @param k    modulus   |k| <= 1
 * @param phi  amplitude (radians, any real)
 */
// __device__   double ellint_d(double k, double phi)
// {
//     if (std::abs(k) > 1.0)
//         return INFINITY ;

//     if (phi < 0.0) return -ellint_d(k, -phi);

//     int   periods;
//     double rphi = period_reduce(phi, M_PI, periods);
//     bool   flip = (rphi > M_PI_2);
//     if (flip) rphi = M_PI - rphi;

//     double k2  = k * k;
//     double sp  = sin(rphi);
//     double cp  = cos(rphi);
//     double sp2 = sp * sp;

//     // D(phi,k) = (sin³phi / 3) * Rd(cos²phi, 1-k²sin²phi, 1)
//     double core = (sp * sp2 / 3.0) * Rd(cp*cp, 1.0-k2*sp2, 1.0);

//     double Dc = Rd(0.0, 1.0-k2, 1.0) / 3.0;   // D(k)

//     double base = flip ? 2.0*Dc - core : core;
//     return base + 2.0 * periods * Dc;
// }

// /** Complete elliptic integral D(k) = D(pi/2, k) */
// __device__   double ellint_d(double k)
// {
//     if (std::abs(k) > 1.0)
//         return INFINITY ;
//     return Rd(0.0, 1.0 - k*k, 1.0) / 3.0;
// }

#endif
