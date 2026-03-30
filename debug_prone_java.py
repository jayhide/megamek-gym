"""Play games against Java, find prone steps, compare move sets."""

import json
import random
import sys

import gymnasium

from megamek_gym.config import MegaMekConfig
from megamek_gym.sim.board import BOARD
from megamek_gym.sim.movement import enumerate_moves
from tests.sim_validation.reconstruct import extract_unit_state, reconstruct_unit
from tests.sim_validation.test_legal_moves import diagnose_prone_extras


def main():
    megamek_dir = sys.argv[1] if len(sys.argv) > 1 else "../megamek"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    prone_found = 0
    prone_with_extras = 0
    games = 0

    for attempt in range(50):
        config = MegaMekConfig(
            megamek_dir=megamek_dir,
            rl_port=port,
            rl_unit="Trebuchet TBT-5S",
            opponent_unit="Trebuchet TBT-5S",
            board="Map Set 6/16x17 Woodland",
            max_game_rounds=50,
            max_rotating_round_saves=0,
            auto_wake_pilot=True,
            firing_strategy="naive",
        )

        env = gymnasium.make("MegaMekGym/MegaMek-v0", config=config)
        inner = env.unwrapped
        obs, info = env.reset()
        games += 1

        for step in range(200):
            raw = inner._last_raw_obs
            if raw.get("terminated") or raw.get("truncated"):
                break
            if not raw.get("legal_moves"):
                break

            java_unit = extract_unit_state(raw, inner._rl_owner_id)
            if java_unit and java_unit.get("prone", False):
                prone_found += 1

                # Reconstruct and enumerate sim moves
                sim_unit = reconstruct_unit(java_unit, "Trebuchet TBT-5S")
                sim_enemy = None
                for u in raw.get("units", []):
                    if u.get("owner") != inner._rl_owner_id:
                        sim_enemy = reconstruct_unit(u, "Trebuchet TBT-5S")
                        break

                sim_moves = enumerate_moves(sim_unit, BOARD, enemy=sim_enemy,
                                            algorithm="deque")

                java_moves = raw.get("legal_moves", [])
                java_hexes = {(m["dest_x"], m["dest_y"]) for m in java_moves}
                sim_hexes = {(m["dest_x"], m["dest_y"]) for m in sim_moves}

                sim_only = sim_hexes - java_hexes
                java_only = java_hexes - sim_hexes

                print(f"\nGame {games}, step {step}: PRONE at ({sim_unit.x},{sim_unit.y},f={sim_unit.facing})")
                print(f"  walk_mp={sim_unit.walk_mp} run_mp={sim_unit.run_mp} heat={sim_unit.heat}")
                print(f"  gyro_hits={sim_unit.gyro_hits} engine_hits={sim_unit.engine_hits}")
                from megamek_gym.sim.unit import Location
                destroyed_locs = [loc.name for loc in Location if sim_unit.loc_destroyed[loc]]
                print(f"  destroyed_locs={destroyed_locs}")
                print(f"  hip_hits={sim_unit.hip_hits} leg_actuator_hits={sim_unit.leg_actuator_hits}")
                # Raw Java entity fields
                print(f"  java: mp_walk={java_unit.get('mp_walk')} mp_run={java_unit.get('mp_run')} "
                      f"mp_used={java_unit.get('mp_used', '?')} delta_dist={java_unit.get('delta_distance', '?')} "
                      f"moved={java_unit.get('moved', '?')} shutdown={java_unit.get('shutdown', '?')}")
                print(f"  Java: {len(java_moves)} moves, {len(java_hexes)} hexes")
                print(f"  Sim:  {len(sim_moves)} moves, {len(sim_hexes)} hexes")
                print(f"  Shared: {len(java_hexes & sim_hexes)} hexes")

                # Show Java mp_used distribution
                java_mp_vals = sorted(set(m["mp_used"] for m in java_moves))
                sim_mp_vals = sorted(set(m["mp_used"] for m in sim_moves))
                print(f"  Java mp_used values: {java_mp_vals}")
                print(f"  Sim  mp_used values: {sim_mp_vals}")

                # Check max mp_used
                java_max_mp = max(m["mp_used"] for m in java_moves) if java_moves else 0
                sim_max_mp = max(m["mp_used"] for m in sim_moves) if sim_moves else 0
                print(f"  Java max_mp={java_max_mp}, Sim max_mp={sim_max_mp}")

                # Check if enemy blocks any paths
                if sim_enemy:
                    print(f"  enemy at ({sim_enemy.x},{sim_enemy.y},f={sim_enemy.facing}) "
                          f"destroyed={sim_enemy.destroyed}")

                if sim_only:
                    prone_with_extras += 1
                    # Classify sim-only moves
                    walk_extras = []
                    run_extras = []
                    for m in sim_moves:
                        if (m["dest_x"], m["dest_y"]) in sim_only:
                            if m["mp_used"] <= sim_unit.walk_mp:
                                walk_extras.append(m)
                            else:
                                run_extras.append(m)

                    print(f"  SIM-ONLY: {len(sim_only)} hexes "
                          f"({len(walk_extras)} walk-speed, {len(run_extras)} run-speed entries)")

                    # Show a few sim-only moves with details
                    for m in sorted(run_extras, key=lambda m: m["mp_used"])[:5]:
                        print(f"    ({m['dest_x']},{m['dest_y']},f={m['facing']}) "
                              f"mp={m['mp_used']} hm={m['hexes_moved']} "
                              f"psr={m.get('success_probability', 1.0):.3f}")

                if java_only:
                    print(f"  JAVA-ONLY: {len(java_only)} hexes: {sorted(java_only)[:10]}")

                if prone_with_extras >= 5:
                    env.close()
                    print(f"\n=== Summary: {prone_found} prone steps, "
                          f"{prone_with_extras} with sim extras ===")
                    return

            # Random action
            legal = raw.get("legal_moves", [])
            action = random.randint(0, len(legal) - 1) if legal else 0
            obs, reward, term, trunc, info = env.step(action)

        env.close()

    print(f"\n=== Summary: {prone_found} prone steps, "
          f"{prone_with_extras} with sim extras across {games} games ===")


if __name__ == "__main__":
    main()
