from stellgapsimtools import StellgapSimTools
import plotly

my_alfven_spec_data_dir = ""
stellgap_output = StellgapSimTools(alfven_spec_directory=my_alfven_spec_data_dir)
figure = stellgap_output.plot_continuum()

figure.show()