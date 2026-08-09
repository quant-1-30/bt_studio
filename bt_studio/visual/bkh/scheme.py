tableau20 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
]


class PlotScheme(object):
    def __init__(self):
        self.figure_width = 1200
        self.main_height = 400
        self.ind_height = 250
        self.ana_height = 250

        self.vbar_width = 0.6 * 24 * 60 * 60 * 1000
        self.line_width = 1.5
        self.scaling_factor = 0.15
        self.location = "top_left"