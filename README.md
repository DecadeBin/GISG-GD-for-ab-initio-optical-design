# GISG-GD-for-ab-initio-optical-design
We have open-sourced the GSIG-GD code framework and its comparative experiments.
By modifying lens_optimization\main: Line1403 -1409 corresponding to:

file_name: Lens type, options are cooke, Ultraviolet, 6p

method_name: Method type, options are: GISG_GD (Global information selective guided gradient descent), GD (base gradient descent), CURR (Curriculum learning), GIG_GD (Global information guided gradient descent, no mask version of GISG-GD), PSO_GD (two stage algorithm, PSO first and then GD), PSO_GISG_GD (PSO first and then GSIG-GD)

Then adjust the following parameters: N (number of systems), epoch (number of optimization generations), lr_max (base learning rate), stop_time (time to determine if it has fallen into a minimum value).

The results are saved in data\data_txt\file_name directory, including the optimal solution's txt document and all output _all.txt files.
