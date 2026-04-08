# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

datas = []
binaries = []
hiddenimports = ['open3d', 'pyvista', 'pandas', 'numpy', 'trimesh', 'scipy', 'PIL', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'threadpoolctl', 'joblib', 'fast_simplification', 'tqdm', 'config', 'geometry', 'particles', 'engine', 'output', 'batch_smoother', 'generate_report', 'run_simulation', 'embreex.rtcore']
tmp_ret = collect_all('open3d')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyvista')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('embreex')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('trimesh')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('scipy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('rtree')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas += [
    (os.path.join(PROJECT_DIR, 'BOT_logo.png'), '.'),
    (os.path.join(PROJECT_DIR, 'BOT_icon.bmp'), '.'),
    (os.path.join(PROJECT_DIR, 'BOT_icon.ico'), '.'),
]


a = Analysis(
    [os.path.join(PROJECT_DIR, 'sim_gui.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BeamOnTarget',
    icon=os.path.join(PROJECT_DIR, 'BOT_icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BeamOnTarget',
)
