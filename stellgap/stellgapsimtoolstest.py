from stellgapsimtools import StellgapSimTools
import plotly

my_dir = ""
stellgap_output = StellgapSimTools(alfven_spec_directory=my_dir)
figure = stellgap_output.plot_continuum()

figure.show()