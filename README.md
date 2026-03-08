Model G 2D Vortex Finder GUI

A Tkinter GUI wrapper for Brendan's solver:

    model_g_2d_xy_safe_plus_vort__1a.py

Run with:

python3 model_g_2d_vortex_finder_gui__1k_step_commented.py

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

Brendan: Thank you very much! This is very helpful. Can you produce the code again, with step by 
step commentary, describing how the code works?

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
