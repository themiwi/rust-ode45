//                    c2     c3     c4       c5  c6
const C: [f64; 5] = [ 1./4., 3./8., 12./13., 1., 1./2. ];

const B: [f64; 6] = [
//  b1        b2  b3            b4             b5       b6
    16./135., 0., 6656./12825., 28561./56430., -9./50., 2./55. ];

const BS: [f64; 6] = [
//  b*1       b*2 b*3          b*4          b*5     b*6
    25./216., 0., 1408./2565., 2197./4104., -1./5., 0. ];

const A: [f64; 15] = [
    1./4.,        // a21:  0

    3./32.,       // a31:  1
    9./32.,       // a32:  2

    1932./2197.,  // a41:  3
    -7200./2197., // a42:  4
    7296./2197.,  // a43:  5

    439./216.,    // a51:  6
    -8.,          // a52:  7
    3680./513.,   // a53:  8
    -845./4104.,  // a54:  9

    -8./27.,      // a61: 10
    2.,           // a62: 11
    -3544./2565., // a63: 12
    1859./4104.,  // a64: 13
    -11./40.,     // a65: 14
];

/// Naïve implementation of a RKF45 ODE integrator.
///
/// Based on [this Wikipedia article](https://en.wikipedia.org/wiki/Runge-Kutta_methods),
/// specifically the Runge-Kutta-Fehlberg method.
pub struct Ode45
{
    f: fn(f64, f64) -> f64,
    y: f64,
    t: f64,
    h: f64,
    hmin: f64,
    hmax: f64,
    delta0: f64,
}

impl Ode45
{
    /// Returns a new `Ode45` object.
    ///
    /// * `f`: The right-hand side function of dy/dt = f(t, y).
    /// * `y`: The initial value of the integral.
    /// * `t`: The initial time.
    /// * `h`: The initial step size.
    ///
    /// The following defaults apply:
    ///
    /// * `hmin = 1e-14`: The minimum time step size
    /// * `hmax = 1e-3`: The maximum time step size
    /// * `delta0 = 1e-6`: The error criterion
    pub fn new(f: fn(f64, f64)->f64, y: f64, t: f64, h: f64) -> Ode45
    {
        Ode45
        {
            f: f,
            y: y,
            t: t,
            h: h,
            hmin: 1e-14,
            hmax: 1e-3,
            delta0: 1e-6,
        }
    }

    // GETTERS

    /// Returns the current function value.
    pub fn y(&self) -> &f64 { return &self.y; }
    /// Returns the current time.
    pub fn t(&self) -> &f64 { return &self.t; }
    /// Returns the current time step size
    pub fn h(&self) -> &f64 { return &self.h; }
    /// Returns the minimum time step size
    pub fn hmin(&self) -> &f64 { return &self.hmin; }
    /// Returns the maximum time step size
    pub fn hmax(&self) -> &f64 { return &self.hmax; }
    /// Returns the error criterion
    pub fn delta0(&self) -> &f64 { return &self.delta0; }

    // SETTERS

    /// Sets the minimum time step size
    pub fn set_hmin(&mut self, hmin: f64) { self.hmin = hmin; }
    /// Sets the maximum time step size
    pub fn set_hmax(&mut self, hmax: f64) { self.hmax = hmax; }
    /// Sets the error criterion
    pub fn set_delta0(&mut self, delta0: f64) { self.delta0 = delta0; }

    /// Performs a single time step.
    ///
    /// Returns `true` if `tmax` has been reached, `false` otherwise. `self.t`
    /// is updated to the new time value, and `self.h` is set to the new
    /// adaptive time-step.
    pub fn step(&mut self, tmax: f64) -> bool
    {
        let mut done = self.clip_h(tmax);

        // get coefficients
        let k = self.calc_k();

        // error estimate
        let delta1 =
              (B[0] - BS[0])*k.0
            + (B[1] - BS[1])*k.1
            + (B[2] - BS[2])*k.2
            + (B[3] - BS[3])*k.3
            + (B[4] - BS[4])*k.4
            + (B[5] - BS[5])*k.5;

        // new, adaptive timestep
        let h1 = (self.delta0/delta1).abs().powf(0.2);
        let h1 = h1.max(self.hmin);
        let h1 = h1.min(self.hmax);

        if delta1 < self.delta0
        {
            // we met criterium
            self.y += self.h*(
                  B[0]*k.0
                + B[1]*k.1
                + B[2]*k.2
                + B[3]*k.3
                + B[4]*k.4
                + B[5]*k.5);
            self.t += self.h;
            // for next time step use adapted time step
            self.h = h1;
        }
        else
        {
            // we exceeded criterium, get coefficients with adapted time step
            self.h = h1;
            done = self.clip_h(tmax);

            let k = self.calc_k();
            self.y += self.h*(
                  B[0]*k.0
                + B[1]*k.1
                + B[2]*k.2
                + B[3]*k.3
                + B[4]*k.4
                + B[5]*k.5);
            self.t += self.h;
        }
        done
    }

    // Calculates RK-coefficients
    fn calc_k(&self) -> (f64, f64, f64, f64, f64, f64)
    {
        let k1 = (self.f)(
            self.t,
            self.y);
        let k2 = (self.f)(
            self.t + C[0]*self.h,
            self.y + self.h*(A[ 0]*k1));
        let k3 = (self.f)(
            self.t + C[1]*self.h,
            self.y + self.h*(A[ 1]*k1+A[ 2]*k2));
        let k4 = (self.f)(
            self.t + C[2]*self.h,
            self.y + self.h*(A[ 3]*k1+A[ 4]*k2+A[ 5]*k3));
        let k5 = (self.f)(
            self.t + C[3]*self.h,
            self.y + self.h*(A[ 6]*k1+A[ 7]*k2+A[ 8]*k3+A[ 9]*k4));
        let k6 = (self.f)(
            self.t + C[4]*self.h,
            self.y + self.h*(A[10]*k1+A[11]*k2+A[12]*k3+A[13]*k4+A[14]*k5));

        (k1, k2, k3, k4, k5, k6)
    }

    // Clips the time step to `tmax-self.t`
    fn clip_h(&mut self, tmax: f64) -> bool
    {
        if self.t + self.h >= tmax
        {
            self.h = tmax - self.t;
            return true;
        }
        false
    }
}


#[allow(unused_imports)]
#[macro_use(assert_approx_eq)]
extern crate assert_approx_eq;

#[cfg(test)]
mod tests {
    use std;
    fn f_zero(_t: f64, _y: f64) -> f64 { 0. }
    fn f_unit(_t: f64, _y: f64) -> f64 { 1. }
    fn f_sin(t: f64, _y: f64) -> f64 {
        println!("DEBUG: t = {}", t);
        t.sin()
    }

    #[test]
    fn it_returns_true_when_done()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        assert!(ode.step(h))
    }

    #[test]
    fn it_returns_false_when_not_done()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        assert!(!ode.step(2.*h))
    }

    #[test]
    fn it_increases_time_step()
    {
        let h = 1e-5;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        ode.step(h);

        assert_eq!(ode.h, ode.hmax);
    }

    #[test]
    fn it_clips_time_step_to_hmax()
    {
        let h = 1.;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        ode.step(h);

        assert_eq!(ode.h, ode.hmax);
    }

    #[test]
    fn it_clips_time_step_to_hmin()
    {
        let h = 1e-15;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);
        // prevent it from going above hmin too
        ode.set_hmax(1e-14);

        ode.step(h);

        assert_eq!(ode.h, ode.hmin);
    }

    #[test]
    fn it_clips_time_step_to_tmax()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        ode.step(h/2.);

        assert_eq!(ode.t, h/2.);
    }

    #[test]
    fn it_integrates_0_from_0_to_1_to_0()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_zero, 0., 0., h);

        while !ode.step(1.) {}
        assert!(ode.y == 0.);
    }

    #[test]
    fn it_integrates_1_from_0_to_1_to_1()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_unit, 0., 0., h);

        while !ode.step(1.) {}

        assert_approx_eq!(ode.y, 1.);
    }

    #[test]
    fn it_integrates_sin_from_0_to_pi_to_2()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_sin, 0., 0., h);

        while !ode.step(std::f64::consts::PI) {}

        assert_approx_eq!(ode.y, 2.);
    }

    #[test]
    fn it_integrates_sin_from_0_to_2pi_to_0()
    {
        let h = 1e-3;

        let mut ode = super::Ode45::new(f_sin, 0., 0., h);

        while !ode.step(2.*std::f64::consts::PI) {}

        assert_approx_eq!(ode.y, 0.);
    }
}
