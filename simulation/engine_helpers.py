# engine_helpers.py
"""Shared particle-batch helpers used by both the ray and EM simulation engines."""
import numpy as np

def _empty_particle_batch():
    return {
        "origins": np.empty((0, 3), dtype=np.float64),
        "directions": np.empty((0, 3), dtype=np.float64),
        "powers": np.empty(0, dtype=np.float64),
        "energies_ev": np.empty(0, dtype=np.float64),
        "currents_a": np.empty(0, dtype=np.float64),
        "masses_kg": np.empty(0, dtype=np.float64),
        "charge_states": np.empty(0, dtype=np.int32),
        "source_indices": np.empty(0, dtype=np.int32),
    }


def _generate_source_particle_data(source, count):
    origins, dirs, powers, energies, currents, charge_states = source.generate(num_particles=count)
    count = len(origins)
    return {
        "origins": np.asarray(origins, dtype=np.float64),
        "directions": np.asarray(dirs, dtype=np.float64),
        "powers": np.asarray(powers, dtype=np.float64),
        "energies_ev": np.asarray(energies, dtype=np.float64),
        "currents_a": np.asarray(currents, dtype=np.float64),
        "masses_kg": np.full(count, source.mass, dtype=np.float64),
        "charge_states": np.asarray(charge_states, dtype=np.int32),
        "source_indices": np.full(count, source.source_index, dtype=np.int32),
    }


def _iter_particle_batches(particle_sources_list, particle_batch_size):
    particle_batch_size = max(1, int(particle_batch_size))

    batch_parts = {key: [] for key in _empty_particle_batch().keys()}
    batch_count = 0

    for source in particle_sources_list:
        remaining = int(source.num_particles)
        if remaining <= 0:
            continue

        while remaining > 0:
            space_left = particle_batch_size - batch_count
            take_count = min(space_left, remaining)
            generated = _generate_source_particle_data(source, take_count)
            if generated["origins"].size == 0:
                break
            for key, value in generated.items():
                batch_parts[key].append(value)
            produced = int(generated["origins"].shape[0])
            batch_count += produced
            remaining -= produced

            if batch_count >= particle_batch_size:
                yield {
                    key: np.concatenate(parts, axis=0)
                    for key, parts in batch_parts.items()
                }
                batch_parts = {key: [] for key in _empty_particle_batch().keys()}
                batch_count = 0

    if batch_count > 0:
        yield {
            key: np.concatenate(parts, axis=0)
            for key, parts in batch_parts.items()
        }


def _empty_impact_data(num_objects):
    return [{"total_hits": 0, "stored_hits": 0, "records": []} for _ in range(num_objects)]


def _merge_impact_records(impact_data, chunk_impacts, save_impact_flags, max_impact_records):
    num_objects = len(impact_data)
    for obj_idx in range(num_objects):
        if not save_impact_flags[obj_idx]:
            continue
        cap = max_impact_records[obj_idx]
        new_records = chunk_impacts[obj_idx]["data"]
        reservoir = impact_data[obj_idx]["records"]
        total_seen = impact_data[obj_idx]["total_hits"]

        for record in new_records:
            total_seen += 1
            if cap is None or len(reservoir) < cap:
                reservoir.append(record)
            else:
                j = np.random.randint(0, total_seen)
                if j < cap:
                    reservoir[j] = record

        impact_data[obj_idx]["total_hits"] = total_seen
        impact_data[obj_idx]["stored_hits"] = len(reservoir)


