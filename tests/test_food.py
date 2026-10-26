"""T14 food energy: unit conversions against hand-worked numbers; the cited yields as recorded."""
import pytest

from src import food


def test_energy_per_hectare_is_hand_worked():
    # 1,000 kg of a food at 100 kcal / 100 g = 1,000,000 g x 1 kcal/g = 1,000,000 kcal = 4,184,000 kJ = 4.184 GJ
    assert food.energy_per_ha_gj(1000.0, 100.0) == pytest.approx(4.184)
    assert food.energy_per_ha_gj(2485.0, 365.0) == pytest.approx(2485 * 365 * 10 * 4.184 / 1e6)


def test_food_energy_adds_rice_land_and_garden_land():
    assert food.food_energy_gj(10.0, 2.0, rice_gj_ha=30.0, veg_gj_ha=5.0) == pytest.approx(10 * 30 + 2 * 5)
    assert food.food_energy_gj(0.0, 0.0, 30.0, 5.0) == 0.0


def test_recorded_yields_match_the_source_tables_and_derive_correctly():
    assert food.RICE_YIELD_KG_HA == 2485.0
    assert food.RICE_YIELD_3YR_KG_HA == pytest.approx((2030 + 2081 + 2485) / 3)
    assert food.VEG_YIELD_T_HA == pytest.approx(9704.19 / 704.18) and 13.7 < food.VEG_YIELD_T_HA < 13.9


def test_vegetable_proxy_is_the_unweighted_mean_of_the_declared_foods():
    assert len(food.VEG_FOODS) == 6 and food.mean_energy([20.0, 30.0, 40.0]) == pytest.approx(30.0)
