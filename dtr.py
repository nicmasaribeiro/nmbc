import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.embed import file_html
from bokeh.resources import CDN


yc = pd.read_csv("csv/dtr.csv")
print(yc.columns)

p = figure(title="Closing Prices", x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')
p.line(yc["1 Mo"],line_width=2)
html = file_html(p, CDN, f"Closing Prices")
f = open('graph.html','w')
f.write(html)
f.flush()

#p.line(np.linspace(0,len(price_paths[0])),price_paths[1],line_width=2,color='red')
#p.line(np.linspace(0,len(price_paths[0])),price_paths[2],line_width=2,color='green')
#p.line(np.linspace(0,len(price_paths[0])),price_paths[3], line_width=2)
#p.line(np.linspace(0,len(price_paths[0])),price_paths[4],line_width=2,color='blue')
#p.line(np.linspace(0,len(price_paths[0])),price_paths[5],line_width=2,color ='yellow')
#p.line(np.linspace(0,len(price_paths[0])),price_paths[6], line_width=2)
#p.line(np.linspace(0,len(price_paths[0])),price_paths[7],line_width=2)
#p.line(np.linspace(0,len(price_paths[0])),price_paths[8],line_width=2)
#p.line(np.linspace(0,len(price_paths[0])),price_paths[9], line_width=2)