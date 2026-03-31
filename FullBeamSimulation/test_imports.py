import sys
sys.path.insert(0, '.')
try:
    import config
    print("config OK:", config.TRACKING_ENGINE, config.BORIS_RELATIVISTIC, config.BORIS_NULLCOLL_ENABLED)
except Exception as e:
    print("config FAIL:", e)
try:
    import nullcoll
    print("nullcoll OK:", nullcoll.NullCollResult._fields[:3])
except Exception as e:
    print("nullcoll FAIL:", e)
try:
    import engine_boris
    print("engine_boris OK")
except Exception as e:
    print("engine_boris FAIL:", e)
try:
    import engine_raytrace
    print("engine_raytrace OK")
except Exception as e:
    print("engine_raytrace FAIL:", e)
try:
    import run_simulation
    print("run_simulation OK")
except Exception as e:
    print("run_simulation FAIL:", e)
