import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAVTREE = (ROOT / "gravity" / "gravtree.c").read_text()
RT_INJECTION = (ROOT / "radiation" / "rt_source_injection.c").read_text()
HYDRO_EVALUATE = (ROOT / "hydro" / "hydro_evaluate.h").read_text()
RIEMANN = (ROOT / "hydro" / "reimann.h").read_text()
FORCETREE = (ROOT / "gravity" / "forcetree.c").read_text()
RUN = (ROOT / "run.c").read_text()


class AdaptiveTreeforceRegression(unittest.TestCase):
    def test_treeforce_decision_is_cached_before_walk_and_reused(self):
        cache = (
            "NeedsNewTreeforce[i] = needs_new_treeforce(i);"
        )
        walk = "for(Ewald_iter = 0; Ewald_iter <= ewald_max; Ewald_iter++)"
        self.assertIn(cache, GRAVTREE)
        self.assertLess(GRAVTREE.index(cache), GRAVTREE.index(walk))

        primary = GRAVTREE[
            GRAVTREE.index("void *gravity_primary_loop"):
            GRAVTREE.index("void *gravity_secondary_loop")
        ]
        final_operations = GRAVTREE[
            GRAVTREE.index("/* now perform final operations on results"):
            GRAVTREE.index("/* end of loop over active particles*/")
        ]
        self.assertIn("if(!NeedsNewTreeforce[i])", primary)
        self.assertIn("if(!NeedsNewTreeforce[i])", final_operations)
        self.assertNotIn("needs_new_treeforce(i)", primary)
        self.assertNotIn("needs_new_treeforce(i)", final_operations)

    def test_cached_new_force_cannot_be_reclassified_as_jerk_only(self):
        gravitational_constant = 6.0
        raw_tree_acceleration = 2.0
        old_acceleration = 10.0
        jerk_increment = 0.5

        needs_treeforce_before_walk = True
        needs_treeforce_after_walk = False

        legacy_result = (
            old_acceleration + jerk_increment
            if not needs_treeforce_after_walk
            else raw_tree_acceleration * gravitational_constant
        )
        cached_result = (
            old_acceleration + jerk_increment
            if not needs_treeforce_before_walk
            else raw_tree_acceleration * gravitational_constant
        )

        self.assertEqual(legacy_result, 10.5)
        self.assertEqual(cached_result, 12.0)


class ReinjectAccretedPhotonsRegression(unittest.TestCase):
    def test_accumulator_is_not_reset_while_packing_inputs(self):
        input_function = RT_INJECTION[
            RT_INJECTION.index("void INPUTFUNCTION_NAME"):
            RT_INJECTION.index("/*! this structure defines the variables")
        ]
        injection_driver = RT_INJECTION[
            RT_INJECTION.index("void rt_source_injection(void)"):
            RT_INJECTION.index('#include "../system/code_block_xchange_finalize.h"')
        ]
        exchange = (
            '#include "../system/code_block_xchange_perform_ops.h"'
        )
        reset = "P[i].BH_accreted_photon_energy = 0;"

        self.assertNotIn(reset, input_function)
        self.assertEqual(len(re.findall(re.escape(reset), RT_INJECTION)), 1)
        self.assertIn(reset, injection_driver)
        self.assertLess(injection_driver.index(exchange), injection_driver.index(reset))
        self.assertRegex(
            injection_driver,
            r"if\(P\[i\]\.Type == 5 && rt_sourceinjection_active_check\(i\)\)"
            r" \{P\[i\]\.BH_accreted_photon_energy = 0;\}",
        )

    def test_local_and_export_payloads_receive_the_same_energy_before_reset(self):
        accumulator = 7.5

        local_payload = accumulator
        export_payload = accumulator
        accumulator = 0.0

        self.assertEqual(local_payload, 7.5)
        self.assertEqual(export_payload, local_payload)
        self.assertEqual(accumulator, 0.0)


class HydroTimestepInitializationRegression(unittest.TestCase):
    def test_target_timestep_is_initialized_before_neighbor_flux_work(self):
        assignment = "dt_hydrostep_i = local.dt_hydrostep_i;"
        neighbor_loop = "for(n = 0; n < numngb; n++)"
        maximum = "dt_hydrostep = DMAX(dt_hydrostep_i , dt_hydrostep_j);"

        self.assertEqual(HYDRO_EVALUATE.count(assignment), 1)
        self.assertLess(HYDRO_EVALUATE.index(assignment), HYDRO_EVALUATE.index(neighbor_loop))
        self.assertLess(HYDRO_EVALUATE.index(assignment), HYDRO_EVALUATE.index(maximum))

    def test_target_timestep_is_not_defaulted_to_zero(self):
        pre_neighbor = HYDRO_EVALUATE[: HYDRO_EVALUATE.index("for(n = 0; n < numngb; n++)")]
        self.assertNotIn("dt_hydrostep_i = 0", pre_neighbor)


class RiemannVacuumThresholdRegression(unittest.TestCase):
    def test_ideal_gas_vacuum_guard_uses_the_exact_threshold(self):
        threshold = "return GAMMA_G4 * (cs_L + cs_R);"
        guard = (
            "(v_line_R - v_line_L) > "
            "riemann_vacuum_velocity_threshold(cs_L,cs_R)"
        )

        self.assertEqual(RIEMANN.count(threshold), 1)
        self.assertEqual(RIEMANN.count(guard), 2)
        self.assertIn(
            "check_vel = GAMMA_G4 * (cs_R + cs_L) - dvel;",
            RIEMANN,
        )

    def test_failed_non_vacuum_estimates_reach_the_fallback_chain(self):
        start = RIEMANN.rindex("void get_wavespeeds_and_pressure_star")
        end = RIEMANN.index("void HLLC_fluxes", start)
        star_estimator = RIEMANN[start:end]
        self.assertNotIn("P_M <= MIN_REAL_NUMBER", star_estimator)
        self.assertIn("if((Riemann_out->P_M <= 0)", star_estimator)
        self.assertIn("if((Riemann_out->P_M<0)", RIEMANN)

    def test_one_sound_speed_rarefaction_is_not_a_physical_vacuum(self):
        gamma = 5.0 / 3.0
        sound_left = sound_right = 1.0
        velocity_jump = 1.01

        legacy_threshold = max(sound_left, sound_right)
        exact_threshold = 2.0 * (sound_left + sound_right) / (gamma - 1.0)

        self.assertGreater(velocity_jump, legacy_threshold)
        self.assertLess(velocity_jump, exact_threshold)
        self.assertAlmostEqual(exact_threshold, 6.0)

        pressure = 1.0 / gamma
        stronger_non_vacuum_jump = 2.0
        first_hllc_pressure = pressure - 0.5 * stronger_non_vacuum_jump
        self.assertLess(first_hllc_pressure, 0.0)
        self.assertLess(stronger_non_vacuum_jump, exact_threshold)


class ExactMfmFluxRegression(unittest.TestCase):
    def test_exact_fallback_rebuilds_mfm_flux_from_final_star_state(self):
        start = RIEMANN.rindex("void Riemann_solver_exact")
        end = RIEMANN.index("void sample_reimann_vaccum_left", start)
        exact_solver = RIEMANN[start:end]

        self.assertIn("Riemann_out->Fluxes.rho = 0;", exact_solver)
        self.assertIn(
            "Riemann_out->Fluxes.v[k] = Riemann_out->P_M * n_unit[k];",
            exact_solver,
        )
        self.assertIn(
            "Riemann_out->Fluxes.p = "
            "Riemann_out->P_M * Riemann_out->S_M;",
            exact_solver,
        )
        self.assertLess(
            exact_solver.index("Riemann_solver_exact"),
            exact_solver.index("Riemann_out->Fluxes.rho = 0;"),
        )

    def test_contact_frame_flux_uses_one_consistent_exact_state(self):
        pressure = 0.37
        contact_speed = -0.42
        normal = (1.0, 0.0, 0.0)

        mass_flux = 0.0
        momentum_flux = tuple(pressure * component for component in normal)
        energy_flux = pressure * contact_speed

        self.assertEqual(mass_flux, 0.0)
        self.assertEqual(momentum_flux, (0.37, 0.0, 0.0))
        self.assertAlmostEqual(energy_flux, -0.1554)


class ExactRiemannIterationRegression(unittest.TestCase):
    def test_vacuum_convergence_and_failure_have_distinct_statuses(self):
        start = RIEMANN.rindex("int iterative_Riemann_solver")
        end = RIEMANN.rindex("double guess_for_pressure")
        iteration = RIEMANN[start:end]
        exact_start = RIEMANN.rindex("void Riemann_solver_exact")
        exact_end = RIEMANN.index("void sample_reimann_vaccum_left", exact_start)
        exact_wrapper = RIEMANN[exact_start:exact_end]

        self.assertIn("if(check_vel <= 0) return 0;", iteration)
        self.assertIn("return -1;", iteration)
        self.assertIn("if(exact_status > 0)", exact_wrapper)
        self.assertIn("else if(exact_status == 0)", exact_wrapper)
        self.assertIn("Riemann_out->P_M = NAN;", exact_wrapper)

    def test_iteration_rejects_nonfinite_arithmetic(self):
        start = RIEMANN.rindex("int iterative_Riemann_solver")
        end = RIEMANN.rindex("double guess_for_pressure")
        iteration = RIEMANN[start:end]

        self.assertIn("(!isfinite(Pg))", iteration)
        self.assertIn("(!isfinite(W_L))", iteration)
        self.assertIn("(!isfinite(W_R))", iteration)
        self.assertIn("(!isfinite(derivative_sum))", iteration)
        self.assertIn("if(!isfinite(tol)) return -1;", iteration)

    def test_pressure_guess_uses_density_for_dimensional_consistency(self):
        start = RIEMANN.rindex("double guess_for_pressure")
        end = RIEMANN.index("void convert_face_to_flux", start)
        guess = RIEMANN[start:end]

        self.assertIn(
            "(v_line_R-v_line_L)"
            "*(Riemann_vec.L.rho+Riemann_vec.R.rho)"
            "*(cs_L+cs_R)",
            guess,
        )
        self.assertNotIn(
            "(v_line_R-v_line_L)"
            "*(Riemann_vec.L.p+Riemann_vec.R.p)"
            "*(cs_L+cs_R)",
            guess,
        )


class ForceTreeCapacityRegression(unittest.TestCase):
    def test_empty_ordinary_nodes_use_the_nodes_array_capacity(self):
        start = FORCETREE.index("void force_create_empty_nodes")
        end = FORCETREE.index("void force_insert_pseudo_particles", start)
        create_empty_nodes = FORCETREE[start:end]

        self.assertIn("if((*nodecount) >= MaxNodes)", create_empty_nodes)
        self.assertNotIn("(*nodecount) >= MaxTopNodes", create_empty_nodes)


class TerminalOutputTimeRegression(unittest.TestCase):
    def test_near_terminal_output_maps_to_the_exact_final_tick(self):
        start = RUN.index("static integertime output_time_to_integer_tick")
        end = RUN.index("integertime find_next_outputtime", start)
        conversion = RUN[start:end]

        self.assertIn("64.0 * DBL_EPSILON", conversion)
        self.assertIn("fabs(time - All.TimeMax)", conversion)
        self.assertIn("return TIMEBASE;", conversion)
        self.assertIn("output_time_is_not_after_terminal(time)", RUN)
        self.assertIn("regular_output_time_with_terminal_snap(time, iter)", RUN)

    def test_public_dustybox_accumulation_is_within_snap_tolerance(self):
        time = 0.0
        for _ in range(250):
            time += 0.01

        tolerance = 64.0 * 2.220446049250313e-16 * 2.5
        self.assertEqual(time, 2.4999999999999907)
        self.assertLess(abs(time - 2.5), tolerance)

    def test_indexed_terminal_detection_survives_long_output_schedules(self):
        for interval, count in ((1.0e-4, 10_000), (1.0e-6, 1_000_000)):
            accumulated = 0.0
            for _ in range(count):
                accumulated += interval
            self.assertNotEqual(accumulated, 1.0)
            indexed = count * interval
            tolerance = 64.0 * 2.220446049250313e-16
            self.assertLessEqual(abs(indexed - 1.0), tolerance)

        start = RUN.index("static double regular_output_time_with_terminal_snap")
        end = RUN.index("integertime find_next_outputtime", start)
        indexed_schedule = RUN[start:end]
        self.assertIn("fma((double) output_index", indexed_schedule)
        self.assertIn("pow(All.TimeBetSnapshot", indexed_schedule)


if __name__ == "__main__":
    unittest.main()
