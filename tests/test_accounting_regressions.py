import unittest
from unittest.mock import patch

import numpy as np

from env.env_core import EconomicSociety
from utils.config import load_config


class AccountingRegressions(unittest.TestCase):
    def make_env(self, scene):
        np.random.seed(17)
        return EconomicSociety(load_config(scene).Environment)

    def step(self, env, t=0, procurement=0.1, policy_rate=None):
        actions = {kind: np.array(gov.initial_action[:gov.action_dim], copy=True)
                   for kind, gov in env.government.items()}
        if "tax" in actions:
            actions["tax"][4] = procurement
        if policy_rate is not None:
            actions["central_bank"][0] = policy_rate
        env.step({"government": actions,
                  "households": np.tile([0.25, 0.6], (env.households.households_n, 1)),
                  "market": np.empty(0), "bank": np.empty(0)}, t)

    def test_debt_rate_on_first_step_and_after_reset(self):
        env = self.make_env("work_hard")
        for debt in (1e6, -1e6):
            env.reset()
            env.main_gov.Bt_next = debt
            self.step(env)
            gov, bank = env.main_gov, env.bank
            rate, lending = bank.base_interest_rate, bank.lending_rate
            self.assertNotAlmostEqual(rate, lending)
            spending = float(np.sum(gov.gov_spending * env.market.price))
            self.assertAlmostEqual(gov.Bt_next, debt * (1 + rate) + spending - gov.tax_array.sum())
            firm_interest = float(np.sum(lending * env.market.Kt_next))
            deposits = float(np.sum(env.households.at_next))
            self.assertAlmostEqual(bank.profit, firm_interest + rate * gov.Bt_next - bank.deposit_rate * deposits)
            settlement = (-bank.deposit_rate * env.households.at.sum()
                          + (lending - bank.depreciation_rate) * env.market.Kt.sum()
                          + rate * debt + deposits - env.market.Kt_next.sum() - gov.Bt_next)
            self.assertAlmostEqual(bank.current_account, settlement, places=6)

    def test_central_bank_rate_is_used_in_the_same_period(self):
        env = self.make_env("dbl_government")
        fiscal = env.government["tax"]
        fiscal.base_interest_rate = 0.02
        for t, policy_rate in enumerate((0.07, 0.01)):
            self.assertNotAlmostEqual(env.bank.base_interest_rate, policy_rate)
            self.step(env, t=t, policy_rate=policy_rate)
            bank = env.bank
            self.assertAlmostEqual(bank.base_interest_rate, policy_rate)
            for gov in env.government.values():
                spending = float(np.sum(gov.gov_spending * env.market.price))
                self.assertAlmostEqual(
                    gov.Bt_next, gov.Bt * (1 + policy_rate) + spending - gov.tax_array.sum()
                )
            firm_interest = float(np.sum(bank.lending_rate * env.market.Kt_next))
            deposit_interest = bank.deposit_rate * env.households.at_next.sum()
            government_interest = policy_rate * env.main_gov.Bt_next
            self.assertAlmostEqual(bank.profit, firm_interest + government_interest - deposit_interest)

    def test_work_hard_actual_procurement_in_all_three_ledgers(self):
        for family_units in (40., 85., 100.):
            with self.subTest(family_units=family_units):
                env = self.make_env("work_hard")
                household_step = env.households.step

                def settle_household_goods(society, t):
                    household_step(society, t)
                    society.households.final_consumption.fill(0)
                    society.households.final_consumption[0, 0] = family_units
                    society.households.goods_consumed_before_turnover = np.array([[family_units]])

                with patch.object(env.market, "production_output", return_value=np.array([[100.]])), \
                     patch.object(env.households, "step", side_effect=settle_household_goods):
                    env.market.price[:] = 2.
                    self.step(env, procurement=0.2)
                gov, bank = env.main_gov, env.bank
                bought = min(20., 100. - family_units)
                self.assertAlmostEqual(gov.gov_spending.item(), bought)
                self.assertAlmostEqual(env.real_deals.item(), family_units + bought)
                delta = gov.Bt_next - (1 + bank.base_interest_rate) * gov.Bt + gov.tax_array.sum()
                self.assertAlmostEqual(delta, 2 * bought)
                k = bank.compute_next_kt(env, 1e12)
                investment = (k - (1 - bank.depreciation_rate) * env.market.Kt).item()
                self.assertAlmostEqual(investment, 2 * (100. - family_units - bought))

    def test_retirement_masks_hours_before_production(self):
        env = self.make_env("delayed_retirement")
        original = env.market.step
        def check(society):
            retired = society.households.age >= society.main_gov.retire_age
            self.assertTrue(retired.any())
            self.assertTrue((society.households.ht[retired] == 0).all())
            self.assertTrue((society.households.ht[~retired] > 0).all())
            return original(society)
        with patch.object(env.market, "step", side_effect=check) as market:
            self.step(env)
        market.assert_called_once()

    def test_pension_ledger_includes_deceased_current_period(self):
        env = self.make_env("delayed_retirement")
        households, gov = env.households, env.main_gov
        pension = np.arange(households.households_n, dtype=float).reshape(-1, 1) * -10
        deceased = np.array([1, 3])
        previous_fund = gov.pension_fund
        with patch.object(households, "sample_deaths_by_probability", return_value=(2, deceased)), \
             patch.object(gov, "calculate_pension", return_value=pension.copy()):
            self.step(env, t=1)
        self.assertAlmostEqual(gov.current_net_households_pension, pension.sum())
        self.assertNotEqual(households.pension.sum(), pension.sum())
        self.assertAlmostEqual(gov.pension_fund, (1 + gov.pension_growth_rate) * previous_fund - pension.sum())

    def test_estate_uses_deceased_indices_not_count(self):
        env = self.make_env("delayed_retirement")
        households = env.households
        households.at_next[:4, 0] = [11., 222., 3333., 44444.]
        env.estate_tax_exemption = 1000.
        env.estate_tax_rate = 0.20
        with patch.object(households, "sample_deaths_by_probability",
                          return_value=(2, np.array([1, 3]))), \
             patch.object(households, "compute_estate_tax", wraps=households.compute_estate_tax) as tax:
            self.step(env, t=1)
        tax.assert_called_once()
        self.assertAlmostEqual(households.estate_tax, 8688.80)
