"""T14 food: food energy produced by cropland and community gardens from cited yields (docs/results_log.md T14 addendum). Yields come from
the Ministry of Agriculture's 'Agricultural Statistics at a Glance 2024-25' (Odisha), energy densities from USDA FoodData Central."""
import json

RICE_YIELD_KG_HA = 2485.0               # Odisha, all seasons, milled-rice terms, 2024-25 (Table 2.31)
RICE_YIELD_3YR_KG_HA = (2030.0 + 2081.0 + 2485.0) / 3.0
VEG_YIELD_T_HA = 9704.19 / 704.18       # Odisha, 2024-25 3rd advance estimate (Table 2.48): production / area
RICE_FOOD = "Rice, white, long-grain, regular, raw, unenriched"
VEG_FOODS = ["Eggplant, raw", "Tomatoes, red, ripe, raw, year round average", "Cabbage, raw", "Cauliflower, raw", "Okra, raw", "Pumpkin, raw"]


def energy_per_ha_gj(yield_kg_ha, kcal_per_100g):
    """Food energy per hectare in gigajoules: kg x kcal/100 g x 10 = kcal; 1 kcal = 4.184 kJ."""
    return float(yield_kg_ha) * float(kcal_per_100g) * 10.0 * 4.184 / 1e6


def food_energy_gj(crop_ha, garden_ha, rice_gj_ha, veg_gj_ha):
    return float(crop_ha) * rice_gj_ha + float(garden_ha) * veg_gj_ha


def mean_energy(kcal):
    return sum(kcal) / len(kcal)


# ---------------------------------------------------------------- USDA FoodData Central
def fetch_energy(cache=None):
    """kcal per 100 g for each food, exact-description match in SR Legacy; cached so the numbers are reproducible offline."""
    import requests

    from src.config import RAW
    cache = cache or RAW / "external" / "usda_fdc.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for name in [RICE_FOOD] + VEG_FOODS:
        r = requests.get("https://api.nal.usda.gov/fdc/v1/foods/search", timeout=120,
                         params=dict(query=name, dataType="SR Legacy", pageSize=25, api_key="DEMO_KEY"))
        r.raise_for_status()
        hits = [f for f in r.json()["foods"] if f["description"].strip().lower() == name.lower()]
        if len(hits) != 1:
            raise LookupError(f"{name!r}: {len(hits)} exact matches; candidates {[f['description'] for f in r.json()['foods'][:8]]}")
        kc = [n["value"] for n in hits[0]["foodNutrients"] if n.get("nutrientNumber") == "208" or n.get("nutrientName") == "Energy" and n.get("unitName") == "KCAL"]
        if len(kc) < 1:
            raise LookupError(f"{name!r}: no energy value")
        out[name] = dict(fdc_id=hits[0]["fdcId"], kcal_per_100g=float(kc[0]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def factors(rice_yield=RICE_YIELD_KG_HA):
    e = fetch_energy()
    rice = e[RICE_FOOD]["kcal_per_100g"]
    veg = mean_energy([e[v]["kcal_per_100g"] for v in VEG_FOODS])
    return dict(rice_kcal_100g=rice, veg_kcal_100g=veg, rice_gj_ha=energy_per_ha_gj(rice_yield, rice),
                veg_gj_ha=energy_per_ha_gj(VEG_YIELD_T_HA * 1000.0, veg), rice_yield_kg_ha=rice_yield, veg_yield_t_ha=VEG_YIELD_T_HA)


if __name__ == "__main__":
    print(json.dumps(factors(), indent=1))
