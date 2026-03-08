#!/usr/bin/env python3
"""
Model G 2D Vortex Finder GUI

A Tkinter GUI wrapper for Brendan's solver:
    model_g_2d_xy_safe_plus_vort__1a.py

Run with: python3 model_g_2d_vortex_finder_gui__1k_step_commented.py

Tested for: Ubuntu 24.04.3 LTS on i7-4790 (Optiplex 7020/9020), Python 3.10+

.........
- Written by Brendan Darrer aided by ChatGPT 5.4 --Date: 8th March 2026 15:24 GMT
- adapted from: @ https://github.com/blue-science/subquantum_kinetics/blob/master/particle.nb and
https://github.com/frostburn/tf2-model-g with https://github.com/frostburn/tf2-model-g/blob/master/docs/overview.pdf
- with ChatGPT 5.4 writing it and Brendan guiding it to produce a clean code.
..........

Main features
-------------
- Edit the main CLI parameters from the original solver.
- SAFE vs VORT mode switching.
- Good support for vortex investigation: rotseed, omega_drive, quiver, vorticity limits.
- Command preview, live log, stop button.
- Optional MP4 export.
- Auto-patches a sidecar runner with --no_mp4 support when needed.
- Vortex diagnostics panel that reads the latest checkpoint and estimates vortex strength/center.

--------------
Brendan: Thank you very much! This is very helpful. Can you produce the code again, with step by step commentary, describing how the code works?
Thought for 5m 11s

ChatGPT 5.4:

Yes — I made a commented edition of the GUI code with STEP 0, STEP 1, STEP 2... style commentary throughout.

Here it is:
model_g_2d_vortex_finder_gui__1k_step_commented.py

What’s inside:

section-by-section comments at the top

comments before the main classes and methods

extra inline commentary in the key parts:

GUI layout

parameter handling

command building

solver launching

checkpoint reading

vorticity calculation

live plot updates

preset save/load

CSV export

It still compiles cleanly, so it should remain runnable as a normal Python file.

At a high level, the code works like this:

defines defaults and preset dictionaries

creates scrollable Tkinter panels

builds the main GUI window

stores all field values in Tk variables

converts GUI settings into a solver command

launches the solver as a background subprocess

reads solver output into the GUI log

watches checkpoint files during the run

computes ωz diagnostics from ux, uy

updates the live readout and live plot

lets you open output folders, save/load presets, and export CSV

If you want, I can also produce a second version with even more detailed line-by-line comments inside the most important functions, especially:
_compute_vortex_diagnostics(), build_command(), and run_simulation().
--------------
"""

from __future__ import annotations

import os
import re
import sys
import json
import csv
import shlex
import queue
import threading
import subprocess
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =============================================================================
# STEP-BY-STEP OVERVIEW
# =============================================================================
# 0. Store default solver arguments and preset definitions.
# 1. Build a reusable scrollable frame helper.
# 2. Create the main GUI object and application state.
# 3. Turn defaults into Tkinter variables.
# 4. Build the top bar, left parameter editor, right diagnostics area, and
#    bottom status/run bar.
# 5. Keep the command preview synchronized with GUI edits.
# 6. Detect the solver and optionally patch a no-MP4 sidecar runner.
# 7. Load checkpoints and compute vortex diagnostics / live plot history.
# 8. Validate parameters and support preset save/load.
# 9. Assemble the final subprocess command.
# 10. Launch, monitor, and stop the solver process.
# 11. Open output folders, export CSV history, and support clipboard copy.
# 12. Start the Tk application.
# =============================================================================


# -----------------------------------------------------------------------------
# Defaults copied from Brendan's attached solver
# -----------------------------------------------------------------------------
# STEP 0: Module-level configuration.
# These dictionaries define the default solver arguments, the layout of
# the editable fields, and a few useful built-in presets.

DEFAULTS = {
    # domain/grid
    "Lx": 20.0,
    "Ly": 20.0,
    "nx": 160,
    "ny": 160,
    # time
    "Tfinal": 40.0,
    "segment_dt": 0.5,
    "nt_anim": 480,
    # solver
    "method": "RK23",
    "max_step": 0.01,
    "atol": 1e-6,
    "rtol": 1e-6,
    # model G
    "a": 14.0,
    "b": 29.0,
    "Dg": 1.0,
    "Dx": 1.0,
    "Dy": 12.0,
    "pcoef": 1.0,
    "qcoef": 1.0,
    "gcoef": 0.1,
    "scoef": 0.0,
    "ucross": 0.0,
    # forcing
    "Tseed": 10.0,
    "seed_sigma_space": 2.0,
    "seed_sigma_time": 3.0,
    "seed_center_x": "",
    "seed_center_y": "",
    # vortical motion
    "no_vort": False,
    "alphaG": 0.02,
    "alphaX": 0.02,
    "alphaY": 0.02,
    "cs2": 1.0,
    "nu": 0.25,
    # rotational seed / drive
    "rotseed": False,
    "swirl_amp": 1.0,
    "swirl_sigma": 6.0,
    "swirl_cx": "",
    "swirl_cy": "",
    "omega_drive": 0.0,
    # viz
    "zlim": 1.0,
    "vortlim": "",
    "quiver": True,
    "quiver_stride": 8,
    # GUI-only
    "export_mp4": True,
}

FIELD_SPECS = [
    ("Domain / grid", [
        ("Lx", "float"), ("Ly", "float"), ("nx", "int"), ("ny", "int"),
    ]),
    ("Time", [
        ("Tfinal", "float"), ("segment_dt", "float"), ("nt_anim", "int"),
    ]),
    ("Solver", [
        ("method", "str"), ("max_step", "float"), ("atol", "float"), ("rtol", "float"),
    ]),
    ("Model G", [
        ("a", "float"), ("b", "float"), ("Dg", "float"), ("Dx", "float"), ("Dy", "float"),
        ("pcoef", "float"), ("qcoef", "float"), ("gcoef", "float"), ("scoef", "float"), ("ucross", "float"),
    ]),
    ("Forcing / seed", [
        ("Tseed", "float"), ("seed_sigma_space", "float"), ("seed_sigma_time", "float"),
        ("seed_center_x", "optfloat"), ("seed_center_y", "optfloat"),
    ]),
    ("Vortical motion", [
        ("alphaG", "float"), ("alphaX", "float"), ("alphaY", "float"), ("cs2", "float"), ("nu", "float"),
    ]),
    ("Rotational seed / drive", [
        ("swirl_amp", "float"), ("swirl_sigma", "float"), ("swirl_cx", "optfloat"), ("swirl_cy", "optfloat"), ("omega_drive", "float"),
    ]),
    ("Visualisation", [
        ("zlim", "float"), ("vortlim", "optfloat"), ("quiver_stride", "int"),
    ]),
]

PRESETS = {
    "SAFE baseline": {
        "no_vort": True,
        "rotseed": False,
        "omega_drive": 0.0,
        "quiver": False,
        "vortlim": "",
    },
    "VORT seed test": {
        "no_vort": False,
        "rotseed": True,
        "swirl_amp": 1.0,
        "swirl_sigma": 6.0,
        "omega_drive": 0.0,
        "quiver": True,
        "quiver_stride": 8,
    },
    "Driven vortex": {
        "no_vort": False,
        "rotseed": True,
        "swirl_amp": 1.0,
        "swirl_sigma": 6.0,
        "omega_drive": 0.5,
        "quiver": True,
        "quiver_stride": 8,
        "vortlim": 0.5,
    },
}


class ScrollableFrame(ttk.Frame):
        # STEP 1A: Create a canvas + vertical scrollbar + inner frame.
        # Widgets are packed into the inner frame, while the canvas provides
        # the actual scrolling behavior.
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

        # STEP 1B: Keep the inner frame width matched to the canvas width.
    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window_id, width=event.width)

        # STEP 1C: Windows/macOS style mouse-wheel scrolling.
    def _on_mousewheel(self, event):
        if self.winfo_ismapped():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # STEP 1D: Linux X11 mouse-wheel scrolling uses Button-4 / Button-5.
    def _on_mousewheel_linux(self, event):
        if not self.winfo_ismapped():
            return
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# STEP 2: Define the main application class.
# This class owns the Tk variables, builds the interface, launches the
# solver process, watches the checkpoint file, computes vortex diagnostics,
# and keeps the live plot/history updated.
class VortexFinderGUI:
        # STEP 2A: Store the root window and initialise application state.
        # This includes subprocess handles, log queues, diagnostics timers,
        # and the in-memory history used by the live |ωz| vs time plot.
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Model G 2D Vortex Finder GUI")
        self._set_initial_geometry()
        self.root.minsize(900, 700)

        self.process: subprocess.Popen | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.generated_runner: Path | None = None
        self.run_buttons: list[ttk.Button] = []
        self.stop_buttons: list[ttk.Button] = []
        self.last_run_context: dict[str, object] | None = None
        self.live_diag_after_id: str | None = None
        self.live_diag_interval_ms = 1500
        self.live_plot_times: list[float] = []
        self.live_plot_peaks: list[float] = []

        self.vars: dict[str, tk.Variable] = {}
        self._build_vars()
        self._build_ui()
        self._auto_detect_solver()
        self._update_mode_state()
        self._update_command_preview()
        self._set_diagnostics_empty("No diagnostics yet")
        # Start the recurring queue-poll loop that feeds solver output into the log window.
        self.root.after(100, self._poll_log_queue)

        # STEP 2B: Choose a startup size that fits the current screen.
    def _set_initial_geometry(self):
        """Choose a startup size that fits on the current screen."""
        try:
            sw = max(800, int(self.root.winfo_screenwidth()))
            sh = max(700, int(self.root.winfo_screenheight()))
        except Exception:
            sw, sh = 1200, 900

        width = min(1160, max(900, int(sw * 0.88)))
        height = min(820, max(640, int(sh * 0.76)))
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
        # STEP 3: Convert defaults into Tkinter variables.
        # These variables back the entry boxes, checkboxes, status labels,
        # command preview, and diagnostics readout.
    def _build_vars(self):
        for key, value in DEFAULTS.items():
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, int):
                self.vars[key] = tk.IntVar(value=value)
            elif isinstance(value, float):
                self.vars[key] = tk.StringVar(value=str(value))
            else:
                self.vars[key] = tk.StringVar(value=value)

        self.vars["solver_path"] = tk.StringVar(value="")
        self.vars["python_exec"] = tk.StringVar(value=sys.executable or "python3")
        self.vars["preset"] = tk.StringVar(value="VORT seed test")
        self.vars["command_preview"] = tk.StringVar(value="")
        self.vars["status"] = tk.StringVar(value="Ready")

        self.vars["diag_status"] = tk.StringVar(value="No diagnostics yet")
        self.vars["diag_mode"] = tk.StringVar(value="—")
        self.vars["diag_time"] = tk.StringVar(value="—")
        self.vars["diag_peak_abs"] = tk.StringVar(value="—")
        self.vars["diag_max"] = tk.StringVar(value="—")
        self.vars["diag_min"] = tk.StringVar(value="—")
        self.vars["diag_peak_loc"] = tk.StringVar(value="—")
        self.vars["diag_center"] = tk.StringVar(value="—")
        self.vars["diag_ckpt"] = tk.StringVar(value="—")
        self.vars["diag_live"] = tk.StringVar(value="—")
        self.vars["diag_points"] = tk.StringVar(value="0")

        # STEP 4: Build the visible GUI layout.
        # The window is split into:
        #   - a top run/setup bar
        #   - a left scrollable parameter area
        #   - a right scrollable diagnostics/log area
        #   - a bottom run/status bar
    def _build_ui(self):
        # Outer container for the whole window.
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="Solver / run setup", padding=10)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Python").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.vars["python_exec"], width=35).grid(row=0, column=1, sticky="ew", padx=(5, 12))

        ttk.Label(top, text="Solver script").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.vars["solver_path"], width=70).grid(row=0, column=3, sticky="ew", padx=(5, 5))
        ttk.Button(top, text="Browse…", command=self._choose_solver).grid(row=0, column=4, sticky="ew")

        ttk.Label(top, text="Preset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        preset_box = ttk.Combobox(top, textvariable=self.vars["preset"], values=list(PRESETS.keys()), state="readonly", width=22)
        preset_box.grid(row=1, column=1, sticky="w", padx=(5, 8), pady=(8, 0))
        preset_box.bind("<<ComboboxSelected>>", lambda _e: self.apply_preset())
        ttk.Button(top, text="Save preset…", command=self.save_preset_to_file).grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Load preset…", command=self.load_preset_from_file).grid(row=1, column=3, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(top, text="Export MP4", variable=self.vars["export_mp4"], command=self._update_command_preview).grid(row=1, column=4, sticky="w", pady=(8, 0), padx=(12, 0))
        ttk.Button(top, text="Restore defaults", command=self.restore_defaults).grid(row=1, column=5, sticky="w", pady=(8, 0))
        ttk.Button(top, text="Copy command", command=self.copy_command).grid(row=1, column=6, sticky="ew", pady=(8, 0))
        ttk.Button(top, text="Open output folder", command=self.open_latest_output_folder).grid(row=1, column=7, sticky="ew", padx=(8, 0), pady=(8, 0))

        toolbar = ttk.Frame(top)
        toolbar.grid(row=2, column=0, columnspan=8, sticky="w", pady=(10, 0))
        top_run_btn = ttk.Button(toolbar, text="Run simulation", command=self.run_simulation)
        top_run_btn.pack(side="left")
        top_stop_btn = ttk.Button(toolbar, text="Stop", command=self.stop_simulation, state="disabled")
        top_stop_btn.pack(side="left", padx=(8, 0))
        ttk.Label(toolbar, text="A second Run/Stop pair is also shown at the bottom.").pack(side="left", padx=(12, 0))
        self.run_buttons.append(top_run_btn)
        self.stop_buttons.append(top_stop_btn)

        top.columnconfigure(3, weight=1)

        # Split the main working area into left and right panes.
        center = ttk.PanedWindow(outer, orient="horizontal")
        center.pack(fill="both", expand=True)

        left_wrap = ttk.Frame(center)
        right_wrap = ttk.Frame(center)
        center.add(left_wrap, weight=2)
        center.add(right_wrap, weight=1)

        self.left_scroll = ScrollableFrame(left_wrap)
        self.left_scroll.pack(fill="both", expand=True)
        self._build_parameter_panels(self.left_scroll.inner)

        self.right_scroll = ScrollableFrame(right_wrap)
        self.right_scroll.pack(fill="both", expand=True)
        self._build_right_panel(self.right_scroll.inner)

        bottom = ttk.Frame(outer)
        bottom.pack(fill="x", pady=(10, 0))
        bottom.columnconfigure(2, weight=1)

        bottom_run_btn = ttk.Button(bottom, text="Run simulation", command=self.run_simulation)
        bottom_run_btn.grid(row=0, column=0, sticky="w")
        bottom_stop_btn = ttk.Button(bottom, text="Stop", command=self.stop_simulation, state="disabled")
        bottom_stop_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.run_buttons.append(bottom_run_btn)
        self.stop_buttons.append(bottom_stop_btn)

        status_frame = ttk.Frame(bottom)
        status_frame.grid(row=0, column=2, sticky="e")
        ttk.Label(status_frame, text="Status:").pack(side="left", padx=(12, 6))
        ttk.Label(status_frame, textvariable=self.vars["status"], anchor="e", justify="right").pack(side="left")

        for key, var in self.vars.items():
            if key not in {"status", "command_preview", "diag_status", "diag_mode", "diag_time", "diag_peak_abs", "diag_max", "diag_min", "diag_peak_loc", "diag_center", "diag_ckpt", "diag_live", "diag_points"}:
                try:
                    var.trace_add("write", lambda *_args: self._on_any_change())
                except Exception:
                    pass

        # STEP 4A: Build the left-hand parameter editor.
        # This includes SAFE/VORT mode buttons, quick vortex toggles, and
        # grouped entry panels for the solver arguments.
    def _build_parameter_panels(self, parent):
        mode_frame = ttk.LabelFrame(parent, text="Mode / quick vortex controls", padding=10)
        mode_frame.pack(fill="x", pady=(0, 10))

        self.mode_var = tk.StringVar(value="vort")
        ttk.Radiobutton(mode_frame, text="SAFE (no vortical motion)", value="safe", variable=self.mode_var, command=self._mode_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(mode_frame, text="VORT (vortical motion enabled)", value="vort", variable=self.mode_var, command=self._mode_changed).grid(row=0, column=1, sticky="w", padx=(15, 0))

        self.chk_rotseed = ttk.Checkbutton(mode_frame, text="rotseed", variable=self.vars["rotseed"], command=self._update_command_preview)
        self.chk_rotseed.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.chk_quiver = ttk.Checkbutton(mode_frame, text="quiver", variable=self.vars["quiver"], command=self._update_command_preview)
        self.chk_quiver.grid(row=1, column=1, sticky="w", pady=(8, 0))

        hint = ttk.Label(
            mode_frame,
            text="Typical vortex search: VORT + rotseed + quiver, then vary omega_drive, swirl_amp, swirl_sigma, nu and vortlim.",
            foreground="#444444",
            wraplength=780,
        )
        hint.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        highlight_names = {"swirl_amp", "swirl_sigma", "swirl_cx", "swirl_cy", "omega_drive", "alphaG", "alphaX", "alphaY", "cs2", "nu", "quiver_stride", "vortlim"}

        for section_title, fields in FIELD_SPECS:
            frame = ttk.LabelFrame(parent, text=section_title, padding=10)
            frame.pack(fill="x", pady=(0, 10))
            for i, (name, _kind) in enumerate(fields):
                r = i // 3
                c = (i % 3) * 2
                ttk.Label(frame, text=name).grid(row=r, column=c, sticky="w", padx=(0, 6), pady=4)
                entry = ttk.Entry(frame, textvariable=self.vars[name], width=18)
                entry.grid(row=r, column=c + 1, sticky="w", padx=(0, 18), pady=4)
                if name in highlight_names:
                    entry.configure(style="Vortex.TEntry")
            for cc in range(0, 6, 2):
                frame.columnconfigure(cc + 1, weight=1)

        # STEP 4B: Build the right-hand panel.
        # This panel shows the command preview, explanatory notes, vortex
        # diagnostics, the live |ωz| trend plot, and the run log.
    def _build_right_panel(self, parent):
        cmd_frame = ttk.LabelFrame(parent, text="Command preview", padding=10)
        cmd_frame.pack(fill="x")
        self.cmd_text = tk.Text(cmd_frame, height=7, wrap="word")
        self.cmd_text.pack(fill="both", expand=True)

        notes_frame = ttk.LabelFrame(parent, text="Notes", padding=10)
        notes_frame.pack(fill="x", pady=(10, 10))
        notes = (
            "SAFE mode sends --no_vort.\n\n"
            "VORT mode enables the velocity fields; rotseed gives an initial swirl and omega_drive adds a persistent tangential drive.\n\n"
            "If Export MP4 is unticked, the GUI auto-builds a patched runner with --no_mp4 and uses that instead of the original solver."
        )
        ttk.Label(notes_frame, text=notes, wraplength=330, justify="left").pack(fill="x")

        diag_frame = ttk.LabelFrame(parent, text="Vortex diagnostics", padding=10)
        diag_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(diag_frame, textvariable=self.vars["diag_status"], wraplength=330, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        diag_rows = [
            ("Detected mode", "diag_mode"),
            ("Checkpoint time", "diag_time"),
            ("Live readout", "diag_live"),
            ("History points", "diag_points"),
            ("|ωz| peak", "diag_peak_abs"),
            ("ωz max", "diag_max"),
            ("ωz min", "diag_min"),
            ("Peak ωz location", "diag_peak_loc"),
            ("Weighted center", "diag_center"),
            ("Checkpoint", "diag_ckpt"),
        ]
        for i, (label, key) in enumerate(diag_rows, start=1):
            ttk.Label(diag_frame, text=label).grid(row=i, column=0, sticky="nw", padx=(0, 8), pady=2)
            ttk.Label(diag_frame, textvariable=self.vars[key], wraplength=235, justify="left").grid(row=i, column=1, sticky="w", pady=2)
        diag_btns = ttk.Frame(diag_frame)
        diag_btns.grid(row=len(diag_rows) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Button(diag_btns, text="Refresh diagnostics", command=self.refresh_diagnostics).pack(side="left")
        ttk.Button(diag_btns, text="Open latest output folder", command=self.open_latest_output_folder).pack(side="left", padx=(8, 0))
        ttk.Label(diag_frame, text="Live diagnostics auto-refresh during a run. Tip: the whole right panel now scrolls if the window is too short.", wraplength=330, justify="left").grid(row=len(diag_rows) + 2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        plot_frame = ttk.LabelFrame(parent, text="Live |ωz| peak vs time", padding=8)
        plot_frame.pack(fill="x", pady=(0, 10))
        self.plot_canvas = tk.Canvas(plot_frame, height=190, highlightthickness=1)
        self.plot_canvas.pack(fill="x", expand=True)
        self.plot_canvas.bind("<Configure>", lambda _e: self._draw_live_plot())
        plot_btns = ttk.Frame(plot_frame)
        plot_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(plot_btns, text="Export |ωz| history to CSV…", command=self.export_live_history_csv).pack(side="left")
        ttk.Label(plot_frame, text="This updates as new checkpoints arrive during the run.", wraplength=330, justify="left").pack(anchor="w", pady=(6, 0))
        diag_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(parent, text="Run log", padding=10)
        log_frame.pack(fill="both", expand=True, pady=(0, 6))
        self.log_text = tk.Text(log_frame, wrap="word", height=12)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ------------------------------------------------------------------
    # State / preview helpers
    # ------------------------------------------------------------------
        # STEP 5: Any parameter edit refreshes mode-dependent widgets and
        # rebuilds the command preview.
    def _on_any_change(self):
        self._update_mode_state()
        self._update_command_preview()

        # STEP 5A: Sync the SAFE/VORT radio buttons into the internal
        # --no_vort boolean.
    def _mode_changed(self):
        self.vars["no_vort"].set(self.mode_var.get() == "safe")
        self._update_mode_state()
        self._update_command_preview()

        # STEP 5B: Enable or disable vortex-only controls depending on whether
        # SAFE or VORT mode is active.
    def _update_mode_state(self):
        safe = bool(self.vars["no_vort"].get())
        self.mode_var.set("safe" if safe else "vort")
        if safe:
            self.chk_rotseed.state(["disabled"])
            self.chk_quiver.state(["disabled"])
        else:
            self.chk_rotseed.state(["!disabled"])
            self.chk_quiver.state(["!disabled"])

        # STEP 5C: Rebuild the exact command-line that would be run.
        # This gives the user a transparent view of how GUI settings map
        # back onto the solver script arguments.
    def _update_command_preview(self):
        try:
            cmd = self.build_command(validate=False)
            preview = shlex.join(cmd)
        except Exception as e:
            preview = f"Command preview unavailable: {e}"
        self.vars["command_preview"].set(preview)
        self.cmd_text.delete("1.0", "end")
        self.cmd_text.insert("1.0", preview)

        # STEP 5D: Append text to the run log widget and keep it scrolled
        # to the newest output.
    def _append_log(self, text: str):
        self.log_text.insert("end", text)
        self.log_text.see("end")

        # STEP 5E: Periodically move lines from the background reader thread
        # into the Tk text widget. Tk updates must stay on the main thread.
    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log(msg)
        self.root.after(100, self._poll_log_queue)

    # ------------------------------------------------------------------
    # Solver detection / patching
    # ------------------------------------------------------------------
        # STEP 6: Try to find Brendan's base solver automatically in likely
        # locations so the GUI works with minimal setup.
    def _auto_detect_solver(self):
        candidates = [
            Path.cwd() / "model_g_2d_xy_safe_plus_vort__1a.py",
            Path(__file__).resolve().parent / "model_g_2d_xy_safe_plus_vort__1a.py",
            Path("/mnt/data/model_g_2d_xy_safe_plus_vort__1a.py"),
        ]
        for p in candidates:
            if p.exists():
                self.vars["solver_path"].set(str(p))
                return

        # STEP 6A: Let the user browse for the base solver manually.
    def _choose_solver(self):
        path = filedialog.askopenfilename(
            title="Choose the base solver script",
            filetypes=[("Python files", "*.py"), ("All files", "*")],
        )
        if path:
            self.vars["solver_path"].set(path)
            self._update_command_preview()

        # STEP 6B: If MP4 export is unticked, make sure we have a runnable
        # solver that supports --no_mp4.
        #
        # The original solver always writes an MP4. This helper creates a
        # sidecar patched runner so the GUI can make MP4 export optional.
    def _ensure_no_mp4_runner(self, source_path: Path) -> Path:
        text = source_path.read_text(encoding="utf-8")
        if "--no_mp4" in text:
            return source_path

        marker = 'ap.add_argument("--quiver_stride", type=int, default=8)'
        if marker not in text:
            raise RuntimeError("Could not find quiver_stride argument in the solver, so auto-patching failed.")

        text = text.replace(
            marker,
            marker + '\nap.add_argument("--no_mp4", action="store_true", help="Skip MP4 assembly at end")',
            1,
        )

        pattern = re.compile(
            r'\n\s*# assemble MP4\n\s*print\("\[Video\] Writing MP4:", mp4_path\)\n\s*fps = .*?\n\s*with imageio\.get_writer\(mp4_path, fps=fps\) as writer:\n\s*for i in range\(args\.nt_anim\):\n\s*img = imageio\.imread\(os\.path\.join\(frames_dir, f"frame_\{i:04d\}\.png"\)\)\n\s*writer\.append_data\(img\)\n\s*print\("\[Done\] MP4 saved:", mp4_path\)',
            re.DOTALL,
        )

        replacement = '''
    # assemble MP4
    if args.no_mp4:
        print("[Video] Skipped MP4 assembly because --no_mp4 was requested")
    else:
        print("[Video] Writing MP4:", mp4_path)
        fps = max(8, int(args.nt_anim / max(1.0, args.Tfinal / 2.0)))
        with imageio.get_writer(mp4_path, fps=fps) as writer:
            for i in range(args.nt_anim):
                img = imageio.imread(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                writer.append_data(img)
        print("[Done] MP4 saved:", mp4_path)'''

        new_text, nsubs = pattern.subn(replacement, text, count=1)
        if nsubs != 1:
            raise RuntimeError("Could not patch the MP4 assembly block cleanly.")

        # Also fix the checkpoint temp-file naming so live diagnostics can follow runs
        # cleanly when the patched runner is used.
        new_text = new_text.replace(
            'tmp = ckpt_path + ".tmp"',
            'tmp = ckpt_path + ".tmp.npz"',
            1,
        )

        runner_path = source_path.with_name(source_path.stem + "__gui_runner.py")
        runner_path.write_text(new_text, encoding="utf-8")
        self.generated_runner = runner_path
        return runner_path

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
        # STEP 7: Reset the diagnostics panel to a clean "waiting/empty" state.
    def _set_diagnostics_empty(self, message: str = "No diagnostics yet"):
        self.vars["diag_status"].set(message)
        for key in ["diag_mode", "diag_time", "diag_live", "diag_peak_abs", "diag_max", "diag_min", "diag_peak_loc", "diag_center", "diag_ckpt"]:
            self.vars[key].set("—")
        self.vars["diag_points"].set(str(len(self.live_plot_times)))

        # STEP 7A: Clear the in-memory history used by the live trend plot.
    def _reset_live_history(self):
        self.live_plot_times = []
        self.live_plot_peaks = []
        self.vars["diag_points"].set("0")
        self._draw_live_plot()

        # STEP 7B: Add one (time, peak |ωz|) point to the live history.
        # If the time is repeated, overwrite the previous point; if the run
        # appears to restart, reset the history automatically.
    def _record_live_point(self, t_curr: float, peak_abs: float):
        if not (np.isfinite(t_curr) and np.isfinite(peak_abs)):
            return
        if not self.live_plot_times:
            self.live_plot_times = [float(t_curr)]
            self.live_plot_peaks = [float(peak_abs)]
        else:
            last_t = float(self.live_plot_times[-1])
            if abs(float(t_curr) - last_t) < 1e-12:
                self.live_plot_peaks[-1] = float(peak_abs)
            elif float(t_curr) > last_t:
                self.live_plot_times.append(float(t_curr))
                self.live_plot_peaks.append(float(peak_abs))
            else:
                # Restart or out-of-order checkpoint: keep a clean history for the new run.
                self.live_plot_times = [float(t_curr)]
                self.live_plot_peaks = [float(peak_abs)]
        self.vars["diag_points"].set(str(len(self.live_plot_times)))
        self._draw_live_plot()

        # STEP 7C: Draw the live |ωz| peak vs time plot on a Tk canvas.
        # The plot is intentionally lightweight, so it updates quickly without
        # requiring matplotlib.
    def _draw_live_plot(self):
        canvas = getattr(self, "plot_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
            w = max(260, int(canvas.winfo_width()))
            h = max(150, int(canvas.winfo_height()))

            left = 58
            right = max(left + 80, w - 18)
            top = 24
            bottom = max(top + 60, h - 38)

            # Axis labels are kept outside the plotting box so they stay clear of
            # the numeric tick labels on smaller windows.
            canvas.create_text(left, 6, text="peak |ωz|", anchor="nw")
            canvas.create_text((left + right) / 2, h - 6, text="time", anchor="s")
            canvas.create_rectangle(left, top, right, bottom, outline="#666666")

            if not self.live_plot_times:
                canvas.create_text((left + right) / 2, (top + bottom) / 2, text="No live points yet", anchor="center")
                return

            tvals = np.asarray(self.live_plot_times, dtype=float)
            pvals = np.asarray(self.live_plot_peaks, dtype=float)

            tmin = float(np.min(tvals))
            tmax = float(np.max(tvals))
            pmin = 0.0
            pmax = float(np.max(pvals))
            if not np.isfinite(pmax) or pmax <= 0:
                pmax = 1.0
            pmax *= 1.05

            def sx(tv):
                if tmax <= tmin:
                    return (left + right) / 2
                return left + (tv - tmin) * (right - left) / (tmax - tmin)

            def sy(pv):
                return bottom - (pv - pmin) * (bottom - top) / max(1e-12, (pmax - pmin))

            for frac in (0.25, 0.5, 0.75):
                yy = top + frac * (bottom - top)
                canvas.create_line(left, yy, right, yy, dash=(2, 3), fill="#cccccc")

            pts = []
            for tt, pp in zip(tvals, pvals):
                pts.extend([sx(float(tt)), sy(float(pp))])

            if len(pts) >= 4:
                canvas.create_line(*pts, width=2, smooth=False)
            if pts:
                x_last, y_last = pts[-2], pts[-1]
                r = 3
                canvas.create_oval(x_last - r, y_last - r, x_last + r, y_last + r, outline="", fill="black")

            canvas.create_text(left - 6, bottom, text=f"{pmin:.0f}", anchor="e")
            canvas.create_text(left - 6, top, text=f"{pmax:.3g}", anchor="e")
            canvas.create_text(left, bottom + 8, text=f"{tmin:.3g}", anchor="n")
            canvas.create_text(right, bottom + 8, text=f"{tmax:.3g}", anchor="n")
        except Exception:
            pass
        # STEP 7D: Format a numeric diagnostic value for the labels.
    def _format_num(self, value: float, digits: int = 6) -> str:
        if value is None or not np.isfinite(value):
            return "—"
        return f"{float(value):.{digits}g}"

        # STEP 7E: Format an (x, y) diagnostic location for display.
    def _format_xy(self, xy: tuple[float, float] | None) -> str:
        if xy is None:
            return "—"
        x, y = xy
        if not (np.isfinite(x) and np.isfinite(y)):
            return "—"
        return f"({x:.4g}, {y:.4g})"

        # STEP 7F: Predict where the solver should save its checkpoint file.
    def _expected_checkpoint_path(self, vals: dict[str, object], base_dir: Path) -> Path:
        base_name = "model_g_2d_xy_safe_plus_vort__1a"
        run_name = base_name + ("_novort" if bool(vals["no_vort"]) else "_vort")
        return base_dir / f"out_{run_name}" / "checkpoint_2d_plus_vort.npz"

        # STEP 7G: Account for the solver's temporary checkpoint naming too.
        # This lets live diagnostics read either the final .npz or the
        # temporary .tmp/.tmp.npz variant while a run is in progress.
    def _candidate_checkpoint_paths(self, ckpt_path: Path) -> list[Path]:
        # The current solver writes via:
        #   tmp = ckpt_path + ".tmp"
        #   np.savez_compressed(tmp, ...)
        # which actually creates "...tmp.npz" because NumPy appends ".npz" when needed.
        # So live diagnostics should look for both the intended checkpoint name and the
        # temporary filename that really appears on disk.
        candidates = [
            ckpt_path,
            Path(str(ckpt_path) + ".tmp"),
            Path(str(ckpt_path) + ".tmp.npz"),
        ]
        seen: set[str] = set()
        out: list[Path] = []
        for p in candidates:
            s = str(p)
            if s not in seen:
                out.append(p)
                seen.add(s)
        return out

        # STEP 7H: Choose the newest existing checkpoint candidate.
    def _resolve_checkpoint_path(self, ckpt_path: Path) -> Path:
        existing = []
        for p in self._candidate_checkpoint_paths(ckpt_path):
            try:
                if p.exists():
                    existing.append(p)
            except Exception:
                pass
        if not existing:
            return ckpt_path
        try:
            return max(existing, key=lambda p: p.stat().st_mtime)
        except Exception:
            return existing[0]

        # STEP 7I: Load the checkpoint and compute vortex diagnostics.
        #
        # Main idea:
        #   1. Load y_curr and t_curr from the checkpoint
        #   2. Rebuild ux and uy if present
        #   3. Compute ωz = d(uy)/dx - d(ux)/dy
        #   4. Find max/min/peak |ωz|
        #   5. Estimate a weighted vortex center
    def _compute_vortex_diagnostics(self, ckpt_path: Path, vals: dict[str, object]) -> dict[str, object]:
        # Load the checkpoint written by the solver.
        d = np.load(ckpt_path, allow_pickle=True)
        y_curr = np.asarray(d["y_curr"]).ravel()
        t_curr = float(d["t_curr"])

        nx = int(vals["nx"])
        ny = int(vals["ny"])
        Lx = float(vals["Lx"])
        Ly = float(vals["Ly"])
        N = nx * ny

        if y_curr.size == 3 * N:
            with_vort = False
            ux = np.zeros((ny, nx), dtype=float)
            uy = np.zeros((ny, nx), dtype=float)
        elif y_curr.size >= 5 * N:
            with_vort = True
            ux = y_curr[3 * N:4 * N].reshape(ny, nx)
            uy = y_curr[4 * N:5 * N].reshape(ny, nx)
        else:
            raise ValueError(
                f"Checkpoint size {y_curr.size} does not match nx*ny={N}. "
                "Try refreshing with the same grid settings that were used for the run."
            )

        x = np.linspace(-Lx / 2.0, Lx / 2.0, nx)
        y = np.linspace(-Ly / 2.0, Ly / 2.0, ny)
        dx = float(x[1] - x[0]) if nx > 1 else 1.0
        dy = float(y[1] - y[0]) if ny > 1 else 1.0

        def gradx(u):
            gx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
            gx[:, 0] = 0.0
            gx[:, -1] = 0.0
            return gx

        def grady(u):
            gy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
            gy[0, :] = 0.0
            gy[-1, :] = 0.0
            return gy

        # Compute the z-component of vorticity: ωz = d(uy)/dx - d(ux)/dy.
        vort = gradx(uy) - grady(ux)
        vmax = float(np.max(vort))
        vmin = float(np.min(vort))
        abs_vort = np.abs(vort)
        peak_abs = float(np.max(abs_vort))
        peak_idx = np.unravel_index(int(np.argmax(abs_vort)), abs_vort.shape)
        peak_loc = (float(x[peak_idx[1]]), float(y[peak_idx[0]]))

        center = None
        if peak_abs > 0.0:
            weights = np.where(abs_vort >= 0.5 * peak_abs, abs_vort, 0.0)
            wsum = float(np.sum(weights))
            if wsum > 0.0:
                Xg, Yg = np.meshgrid(x, y, indexing="xy")
                center = (
                    float(np.sum(weights * Xg) / wsum),
                    float(np.sum(weights * Yg) / wsum),
                )

        return {
            "mode": "VORT" if with_vort else "SAFE",
            "t_curr": t_curr,
            "peak_abs": peak_abs,
            "vmax": vmax,
            "vmin": vmin,
            "peak_loc": peak_loc,
            "center": center if center is not None else peak_loc,
            "ckpt_path": str(ckpt_path),
        }

        # STEP 7J: Build a compact one-line live summary from the diagnostic labels.
    def _update_live_readout(self):
        mode = self.vars["diag_mode"].get()
        tval = self.vars["diag_time"].get()
        peak = self.vars["diag_peak_abs"].get()
        center = self.vars["diag_center"].get()
        if mode == "—" or tval == "—" or peak == "—":
            self.vars["diag_live"].set("Waiting for checkpoint…")
            return
        summary = f"{mode} | t={tval} | |ωz|={peak}"
        if center not in ("—", "", None):
            summary += f" | center {center}"
        self.vars["diag_live"].set(summary)

        # STEP 7K: Start the repeating timer that refreshes diagnostics during a run.
    def _schedule_live_diagnostics(self):
        self._cancel_live_diagnostics()
        self.live_diag_after_id = self.root.after(self.live_diag_interval_ms, self._live_diag_tick)

        # STEP 7L: Stop any pending live-diagnostics timer.
    def _cancel_live_diagnostics(self):
        if self.live_diag_after_id is not None:
            try:
                self.root.after_cancel(self.live_diag_after_id)
            except Exception:
                pass
            self.live_diag_after_id = None

        # STEP 7M: One live-diagnostics tick: refresh diagnostics and then
        # reschedule the next tick if the process is still running.
    def _live_diag_tick(self):
        self.live_diag_after_id = None
        try:
            if self.process is not None and self.process.poll() is None:
                self.refresh_diagnostics(silent=True)
                self._schedule_live_diagnostics()
        except Exception:
            # Keep the GUI responsive even if a checkpoint is temporarily unreadable.
            if self.process is not None and self.process.poll() is None:
                self._schedule_live_diagnostics()

        # STEP 7N: Refresh the diagnostics panel from the latest checkpoint.
        # This is used both when the user clicks the button and automatically
        # during a run.
    def refresh_diagnostics(self, silent: bool = False):
        try:
            if self.last_run_context is not None:
                vals = dict(self.last_run_context["vals"])
                ckpt_path = Path(self.last_run_context["ckpt_path"])
            else:
                vals = self.collect_values(validate=True)
                solver_path_raw = self.vars["solver_path"].get().strip()
                if not solver_path_raw:
                    raise ValueError("Please choose the solver script first.")
                solver_dir = Path(solver_path_raw).expanduser().resolve().parent
                ckpt_path = self._expected_checkpoint_path(vals, solver_dir)

            resolved_ckpt = self._resolve_checkpoint_path(ckpt_path)
            if not resolved_ckpt.exists():
                candidates = "\n".join(str(p) for p in self._candidate_checkpoint_paths(ckpt_path))
                self._set_diagnostics_empty(f"Checkpoint not found yet. Looked for:\n{candidates}")
                self.vars["diag_live"].set("Waiting for checkpoint…")
                return

            diag = self._compute_vortex_diagnostics(resolved_ckpt, vals)
            if resolved_ckpt == ckpt_path:
                self.vars["diag_status"].set("Diagnostics loaded from latest checkpoint")
            else:
                self.vars["diag_status"].set(f"Diagnostics loaded from temporary checkpoint: {resolved_ckpt.name}")
            self.vars["diag_mode"].set(diag["mode"])
            self.vars["diag_time"].set(self._format_num(diag["t_curr"]))
            self.vars["diag_peak_abs"].set(self._format_num(diag["peak_abs"]))
            self.vars["diag_max"].set(self._format_num(diag["vmax"]))
            self.vars["diag_min"].set(self._format_num(diag["vmin"]))
            self.vars["diag_peak_loc"].set(self._format_xy(diag["peak_loc"]))
            self.vars["diag_center"].set(self._format_xy(diag["center"]))
            self.vars["diag_ckpt"].set(str(diag["ckpt_path"]))
            self._record_live_point(float(diag["t_curr"]), float(diag["peak_abs"]))
            self._update_live_readout()
        except Exception as e:
            if silent:
                self.vars["diag_status"].set(f"Live refresh waiting: {e}")
                self.vars["diag_live"].set("Live refresh waiting for checkpoint…")
            else:
                self._set_diagnostics_empty(f"Diagnostics unavailable: {e}")

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
        # STEP 8: Put all GUI parameters back to their original defaults.
    def restore_defaults(self):
        for key, default in DEFAULTS.items():
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(default))
            elif isinstance(var, tk.IntVar):
                var.set(int(default))
            else:
                var.set(str(default))
        self.vars["preset"].set("VORT seed test")
        self._update_mode_state()
        self._update_command_preview()

        # STEP 8A: Apply one of the built-in presets from the drop-down box.
    def apply_preset(self):
        preset = PRESETS.get(self.vars["preset"].get(), {})
        for key, value in preset.items():
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.IntVar):
                var.set(int(value))
            else:
                var.set(str(value))
        self._update_mode_state()
        self._update_command_preview()


        # STEP 8B: Apply a preset that was loaded from disk.
    def _apply_loaded_preset_dict(self, data: dict[str, object], *, preset_label: str = "Loaded preset"):
        for key in DEFAULTS.keys():
            if key not in data or key not in self.vars:
                continue
            value = data[key]
            var = self.vars[key]
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.IntVar):
                try:
                    var.set(int(value))
                except Exception:
                    var.set(int(DEFAULTS[key]))
            else:
                if value is None:
                    var.set("")
                else:
                    var.set(str(value))
        self.vars["preset"].set(preset_label)
        self._update_mode_state()
        self._update_command_preview()

        # STEP 8C: Save the current GUI settings to a JSON preset file.
    def save_preset_to_file(self):
        try:
            data = self.collect_values(validate=True)
        except Exception as e:
            messagebox.showerror("Save preset", f"Cannot save preset because some values are invalid.\n\n{e}")
            return

        data["no_vort"] = bool(self.vars["no_vort"].get())
        data["rotseed"] = bool(self.vars["rotseed"].get())
        data["quiver"] = bool(self.vars["quiver"].get())
        data["export_mp4"] = bool(self.vars["export_mp4"].get())
        data["_meta"] = {
            "kind": "model_g_2d_vortex_finder_preset",
            "version": 1,
        }

        initial_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.vars["preset"].get().strip() or "vortex_preset") + ".json"
        path = filedialog.asksaveasfilename(
            title="Save preset",
            defaultextension=".json",
            initialfile=initial_name,
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            self.vars["status"].set(f"Preset saved: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Save preset", f"Could not save preset.\n\n{e}")

        # STEP 8D: Load a JSON preset file and apply it to the GUI.
    def load_preset_from_file(self):
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Preset file does not contain a JSON object.")
            self._apply_loaded_preset_dict(data, preset_label=Path(path).stem)
            self.vars["status"].set(f"Preset loaded: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Load preset", f"Could not load preset.\n\n{e}")

        # STEP 8E: Convert one Tk variable into the requested Python type.
        # This method also handles optional blank floats cleanly.
    def _parse_number(self, name: str, kind: str):
        raw_value = self.vars[name].get()
        raw = raw_value.strip() if isinstance(raw_value, str) else raw_value
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
        if kind == "optfloat":
            return None if raw in ("", None) else float(raw)
        if kind == "str":
            return raw if isinstance(raw, str) else str(raw)
        raise ValueError(f"Unknown kind: {kind}")

        # STEP 8F: Read all GUI fields into one validated Python dictionary.
        # This is the central parameter bundle used for command-building,
        # diagnostics, presets, and output-folder guessing.
    def collect_values(self, validate: bool = True) -> dict[str, object]:
        vals: dict[str, object] = {}
        for _section, fields in FIELD_SPECS:
            for name, kind in fields:
                try:
                    vals[name] = self._parse_number(name, kind)
                except Exception as e:
                    if validate:
                        raise ValueError(f"Invalid value for {name}: {e}")
                    vals[name] = self.vars[name].get()

        vals["no_vort"] = bool(self.vars["no_vort"].get())
        vals["rotseed"] = bool(self.vars["rotseed"].get())
        vals["quiver"] = bool(self.vars["quiver"].get())
        vals["export_mp4"] = bool(self.vars["export_mp4"].get())

        if vals["seed_center_x"] is None or vals["seed_center_y"] is None:
            if (vals["seed_center_x"] is None) != (vals["seed_center_y"] is None):
                raise ValueError("seed_center_x and seed_center_y must either both be blank or both be set.")
        if vals["swirl_cx"] is None or vals["swirl_cy"] is None:
            if (vals["swirl_cx"] is None) != (vals["swirl_cy"] is None):
                raise ValueError("swirl_cx and swirl_cy must either both be blank or both be set.")

        if int(vals["nx"]) < 8 or int(vals["ny"]) < 8:
            raise ValueError("nx and ny should both be at least 8.")
        if float(vals["Tfinal"]) <= 0:
            raise ValueError("Tfinal must be > 0.")
        if int(vals["nt_anim"]) < 2:
            raise ValueError("nt_anim must be at least 2.")
        if float(vals["segment_dt"]) <= 0 or float(vals["max_step"]) <= 0:
            raise ValueError("segment_dt and max_step must be > 0.")
        if int(vals["quiver_stride"]) < 1:
            raise ValueError("quiver_stride must be >= 1.")
        return vals

        # STEP 9: Convert the validated GUI settings into the exact command
        # line that will be given to subprocess.Popen(...).
    def build_command(self, validate: bool = True) -> list[str]:
        solver_path_raw = self.vars["solver_path"].get().strip()
        if not solver_path_raw:
            raise ValueError("Please choose the solver script first.")
        source_path = Path(solver_path_raw).expanduser().resolve()
        if validate and not source_path.exists():
            raise ValueError(f"Solver script not found: {source_path}")

        vals = self.collect_values(validate=validate)
        python_exec = self.vars["python_exec"].get().strip() or sys.executable or "python3"

        # By default run the original solver. If MP4 export is off, switch to the
        # patched sidecar runner that understands --no_mp4.
        solver_to_run = source_path
        if not vals["export_mp4"]:
            solver_to_run = self._ensure_no_mp4_runner(source_path)

        # Start the command with the Python executable and the solver path.
        cmd = [python_exec, str(solver_to_run)]
        ordered_fields = [
            "Lx", "Ly", "nx", "ny",
            "Tfinal", "segment_dt", "nt_anim",
            "method", "max_step", "atol", "rtol",
            "a", "b", "Dg", "Dx", "Dy", "pcoef", "qcoef", "gcoef", "scoef", "ucross",
            "Tseed", "seed_sigma_space", "seed_sigma_time",
            "alphaG", "alphaX", "alphaY", "cs2", "nu",
            "swirl_amp", "swirl_sigma", "omega_drive",
            "zlim", "quiver_stride",
        ]
        for key in ordered_fields:
            cmd.extend([f"--{key}", str(vals[key])])

        if vals["seed_center_x"] is not None and vals["seed_center_y"] is not None:
            cmd.extend(["--seed_center", str(vals["seed_center_x"]), str(vals["seed_center_y"])])
        if vals["swirl_cx"] is not None and vals["swirl_cy"] is not None:
            cmd.extend(["--swirl_cx", str(vals["swirl_cx"]), "--swirl_cy", str(vals["swirl_cy"])])
        if vals["vortlim"] is not None:
            cmd.extend(["--vortlim", str(vals["vortlim"])])

        if vals["no_vort"]:
            cmd.append("--no_vort")
        if vals["rotseed"] and not vals["no_vort"]:
            cmd.append("--rotseed")
        if vals["quiver"] and not vals["no_vort"]:
            cmd.append("--quiver")
        if not vals["export_mp4"]:
            cmd.append("--no_mp4")
        return cmd

    # ------------------------------------------------------------------
    # Run / stop / logging
    # ------------------------------------------------------------------
        # STEP 10: Launch the solver as a background subprocess.
        #
        # High-level flow:
        #   1. Refuse if a run is already active
        #   2. Validate settings and build the command
        #   3. Predict output/checkpoint locations
        #   4. Clear old log/history/diagnostics
        #   5. Start the subprocess
        #   6. Start background threads for log reading and process waiting
        #   7. Start live diagnostics updates
    def run_simulation(self):
        if self.process is not None:
            # Treat a dead process as finished even if the wait-thread has not
            # yet cleared self.process on the Tk thread.
            if self.process.poll() is None:
                messagebox.showinfo("Already running", "A simulation is already running. See the Run log / status line for its PID and output folder.")
                return
            self._on_process_finished(self.process.returncode if self.process.returncode is not None else -1)

        try:
            vals = self.collect_values(validate=True)
            cmd = self.build_command(validate=True)
        except Exception as e:
            messagebox.showerror("Cannot run", str(e))
            return

        solver_dir = Path(self.vars["solver_path"].get().strip()).expanduser().resolve().parent
        ckpt_path = self._expected_checkpoint_path(vals, solver_dir)
        out_dir = ckpt_path.parent
        self.last_run_context = {
            "vals": vals,
            "solver_dir": str(solver_dir),
            "ckpt_path": str(ckpt_path),
            "out_dir": str(out_dir),
        }

        self.log_text.delete("1.0", "end")
        self._append_log("Running command:\n" + shlex.join(cmd) + "\n\n")
        self._append_log(f"[GUI] Working directory: {solver_dir}\n")
        self._append_log(f"[GUI] Expected output folder: {out_dir}\n")
        self._append_log("[GUI] Note: solver stdout is captured into this Run log, not your launching terminal.\n\n")
        self.vars["status"].set("Running…")
        self._reset_live_history()
        self._set_diagnostics_empty("Waiting for a checkpoint from the current run…")
        self.vars["diag_live"].set("Waiting for checkpoint…")
        for btn in self.run_buttons:
            btn.configure(state="disabled")
        for btn in self.stop_buttons:
            btn.configure(state="normal")

        # Launch the solver in the solver directory and capture stdout into the GUI.
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=str(solver_dir),
            )
        except Exception as e:
            self._cancel_live_diagnostics()
            self.process = None
            self.vars["status"].set("Launch failed")
            for btn in self.run_buttons:
                btn.configure(state="normal")
            for btn in self.stop_buttons:
                btn.configure(state="disabled")
            messagebox.showerror("Launch failed", str(e))
            return

        self.vars["status"].set(f"Running (PID {self.process.pid})")
        self._append_log(f"[GUI] Started solver PID: {self.process.pid}\n\n")
        threading.Thread(target=self._reader_thread, daemon=True).start()
        threading.Thread(target=self._wait_thread, daemon=True).start()
        self._schedule_live_diagnostics()
        # STEP 10A: Background thread that reads solver stdout line by line
        # and pushes it into a thread-safe queue.
    def _reader_thread(self):
        assert self.process is not None
        try:
            for line in self.process.stdout:
                self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"\n[GUI] Log reader error: {e}\n")

        # STEP 10B: Background thread that waits for the solver to exit, then
        # notifies the Tk thread so the GUI can clean up safely.
    def _wait_thread(self):
        assert self.process is not None
        code = self.process.wait()
        self.root.after(0, lambda: self._on_process_finished(code))

        # STEP 10C: Clean up after the solver exits, restore button states,
        # and refresh diagnostics one last time if possible.
    def _on_process_finished(self, code: int):
        self._cancel_live_diagnostics()
        self.process = None
        for btn in self.run_buttons:
            btn.configure(state="normal")
        for btn in self.stop_buttons:
            btn.configure(state="disabled")

        if code == 0:
            self.vars["status"].set("Finished")
            self._append_log("\n[GUI] Simulation finished successfully.\n")
            self.refresh_diagnostics()
        else:
            self.vars["status"].set(f"Stopped / failed (exit {code})")
            self._append_log(f"\n[GUI] Simulation exited with code {code}.\n")
            if self.last_run_context is not None and Path(self.last_run_context["ckpt_path"]).exists():
                self.refresh_diagnostics()

        # STEP 10D: Ask the running solver process to terminate.
    def stop_simulation(self):
        if self.process is None:
            return
        try:
            self.process.terminate()
            self._cancel_live_diagnostics()
            self._append_log("\n[GUI] Stop requested…\n")
            self.vars["status"].set("Stopping…")
        except Exception as e:
            messagebox.showerror("Stop failed", str(e))

        # STEP 11: Infer the most likely output folder for the current or
        # most recent run.
    def _guess_output_folder(self) -> Path | None:
        if self.last_run_context is not None:
            out_dir = Path(self.last_run_context["out_dir"])
            if out_dir.exists():
                return out_dir
        try:
            vals = self.collect_values(validate=True)
            solver_path_raw = self.vars["solver_path"].get().strip()
            if not solver_path_raw:
                return None
            solver_dir = Path(solver_path_raw).expanduser().resolve().parent
            return self._expected_checkpoint_path(vals, solver_dir).parent
        except Exception:
            return None

        # STEP 11A: Open the output folder in the platform file manager.
    def open_latest_output_folder(self):
        out_dir = self._guess_output_folder()
        if out_dir is None:
            messagebox.showerror("Open output folder", "Could not determine the output folder yet.")
            return
        if not out_dir.exists():
            messagebox.showinfo("Open output folder", f"Output folder does not exist yet:\n{out_dir}")
            return
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(out_dir)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            elif os.name == "nt":
                os.startfile(str(out_dir))
            else:
                messagebox.showinfo("Open output folder", f"Please open this folder manually:\n{out_dir}")
                return
            self.vars["status"].set("Opened output folder")
        except Exception as e:
            messagebox.showerror("Open output folder", f"Could not open folder:\n{out_dir}\n\n{e}")

        # STEP 11B: Save the live (time, peak |ωz|) history to a CSV file.
    def export_live_history_csv(self):
        if not self.live_plot_times or not self.live_plot_peaks:
            messagebox.showinfo("Export CSV", "No live |ωz| history points are available yet.")
            return

        default_dir = None
        guessed = self._guess_output_folder()
        if guessed is not None:
            try:
                guessed.mkdir(parents=True, exist_ok=True)
                default_dir = guessed
            except Exception:
                default_dir = None

        path = filedialog.asksaveasfilename(
            title="Export live |ωz| history to CSV",
            defaultextension=".csv",
            initialdir=str(default_dir) if default_dir is not None else None,
            initialfile="vortex_peak_history.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not path:
            return

        # Write the live history as a simple 3-column CSV.
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["index", "time", "peak_abs_omega_z"])
                for idx, (tt, pp) in enumerate(zip(self.live_plot_times, self.live_plot_peaks), start=1):
                    writer.writerow([idx, f"{float(tt):.12g}", f"{float(pp):.12g}"])
            self.vars["status"].set(f"CSV exported: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Export CSV", f"Could not export CSV.\n\n{e}")

        # STEP 11C: Copy the command preview text to the clipboard.
    def copy_command(self):
        text = self.vars["command_preview"].get()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.vars["status"].set("Command copied")


# STEP 12: Create the Tk root, configure the style, build the GUI object,
# and enter the Tk event loop.
def main():
    root = tk.Tk()
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Vortex.TEntry")
    VortexFinderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
