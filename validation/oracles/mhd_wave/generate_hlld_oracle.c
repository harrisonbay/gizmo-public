/*
 * Standalone differential-oracle driver for GIZMO's actual HLLD routine.
 *
 * This compatibility shell intentionally defines only the types and constants
 * needed to compile hydro/reimann.h. The numerical implementation remains the
 * repository implementation; it is not copied into this driver.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGNETIC
#define DIVBCLEANING_DEDNER
#define HYDRO_MESHLESS_FINITE_MASS

#define GAMMA_DEFAULT (5.0 / 3.0)
#define MIN_REAL_NUMBER 1e-56
#define DMAX(a, b) (((a) > (b)) ? (a) : (b))
#define DMIN(a, b) (((a) < (b)) ? (a) : (b))
#define MHD_CONSTRAINED_GRADIENT_FAC_MINMAX 0.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MEDDEV 0.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MAX_PM 0.0
#define MHD_CONSTRAINED_GRADIENT_FAC_MED_PM 0.0

typedef double MyDouble;
typedef double MyFloat;

struct Conserved_var_Riemann {
    MyDouble rho;
    MyDouble p;
    MyDouble v[3];
    MyDouble u;
    MyDouble cs;
    MyDouble B[3];
    MyDouble B_normal_corrected;
    MyDouble phi;
};

struct kernel_hydra {
    double dp[3];
    double r;
};

struct {
    double cf_a2inv;
    double cf_a3inv;
    double cf_afac1;
    double cf_afac3;
    double cf_atime;
    double DivBcleanHyperbolicSigma;
    int ComovingIntegrationOn;
} All;

static void endrun(int status) {
    exit(status);
}

#include "../../../hydro/reimann.h"

struct oracle_case {
    const char *name;
    struct Conserved_var_Riemann left;
    struct Conserved_var_Riemann right;
};

static struct Conserved_var_Riemann state(double rho, double pressure,
                                           double vx, double vy, double vz,
                                           double bx, double by, double bz,
                                           double phi) {
    struct Conserved_var_Riemann value;
    memset(&value, 0, sizeof(value));
    value.rho = rho;
    value.p = pressure;
    value.v[0] = vx;
    value.v[1] = vy;
    value.v[2] = vz;
    value.u = pressure / ((GAMMA_DEFAULT - 1.0) * rho);
    value.cs = sqrt(GAMMA_DEFAULT * pressure / rho);
    value.B[0] = bx;
    value.B[1] = by;
    value.B[2] = bz;
    value.phi = phi;
    return value;
}

static void emit_case(const struct oracle_case *test) {
    struct Input_vec_Riemann input;
    struct Riemann_outputs output;
    memset(&input, 0, sizeof(input));
    memset(&output, 0, sizeof(output));
    input.L = test->left;
    input.R = test->right;

    HLLD_Riemann_solver(input, &output, 1e100);

    printf(
        "%s,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
        "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
        test->name,
        test->left.rho, test->left.p,
        test->left.v[0], test->left.v[1], test->left.v[2],
        test->left.B[0], test->left.B[1], test->left.B[2], test->left.phi,
        test->right.rho, test->right.p,
        test->right.v[0], test->right.v[1], test->right.v[2],
        test->right.B[0], test->right.B[1], test->right.B[2], test->right.phi,
        output.Fluxes.rho,
        output.Fluxes.v[0], output.Fluxes.v[1], output.Fluxes.v[2],
        output.Fluxes.p,
        output.Fluxes.B[0], output.Fluxes.B[1], output.Fluxes.B[2],
        output.S_M, output.P_M, output.B_normal_corrected,
        output.phi_normal_mean, output.phi_normal_db,
        output.cfast_L, output.cfast_R);
}

int main(void) {
    const struct oracle_case cases[] = {
        {
            "constant_fast_wave_base",
            state(1.0, 0.6, 0.0, 0.0, 0.0, 1.0, sqrt(2.0), 0.5, 0.0),
            state(1.0, 0.6, 0.0, 0.0, 0.0, 1.0, sqrt(2.0), 0.5, 0.0),
        },
        {
            "dedner_phi_and_bnormal_jump",
            state(1.1, 0.7, 0.2, -0.1, 0.05, 0.8, 1.2, -0.3, 0.18),
            state(0.9, 0.5, -0.15, 0.08, -0.04, 1.1, 0.7, 0.2, -0.12),
        },
        {
            "zero_normal_field_degeneracy",
            state(1.0, 1.0, 0.15, 0.6, -0.2, 0.0, 1.0, 0.5, 0.0),
            state(0.7, 0.4, -0.25, -0.3, 0.4, 0.0, -0.8, 0.2, 0.0),
        },
        {
            "brio_wu_strong_discontinuity",
            state(1.0, 1.0, 0.0, 0.0, 0.0, 0.75, 1.0, 0.0, 0.0),
            state(0.125, 0.1, 0.0, 0.0, 0.0, 0.75, -1.0, 0.0, 0.0),
        },
    };
    size_t index;

    puts(
        "case,"
        "l_rho,l_p,l_vx,l_vy,l_vz,l_bx,l_by,l_bz,l_phi,"
        "r_rho,r_p,r_vx,r_vy,r_vz,r_bx,r_by,r_bz,r_phi,"
        "flux_mass,flux_momentum_x,flux_momentum_y,flux_momentum_z,"
        "flux_total_energy,flux_bx,flux_by,flux_bz,"
        "contact_speed,star_total_pressure,corrected_normal_b,"
        "phi_mean,phi_db,cfast_l,cfast_r");
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        emit_case(&cases[index]);
    }
    return EXIT_SUCCESS;
}
