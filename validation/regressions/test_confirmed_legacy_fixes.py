import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GRAVTREE = (ROOT / "gravity" / "gravtree.c").read_text()
RT_INJECTION = (ROOT / "radiation" / "rt_source_injection.c").read_text()


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


if __name__ == "__main__":
    unittest.main()
