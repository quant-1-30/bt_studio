# import pandas as pd
# import numpy as np
# from bokeh.plotting import figure, show, output_file
# from bokeh.layouts import gridplot, column
# from bokeh.models import CDSView, BooleanFilter, Range1d, HoverTool, CrosshairTool, LinearAxis, Tabs, Span, CustomJS, PanTool, WheelZoomTool, NumeralTickFormatter
# from bokeh.io import export_png, export_svg

# from .scheme import tableau20
# from .utils import create_datasource, merge_cds, on_shift

# from backtest.metabase import with_metaclass, AutoInfoClass, MetaParams


class PlotScheme(object):
    def __init__(self):
        self.figure_width = 1500
        self.figure_height = [400, 200, 200]

        self.vbar_width = 3 
        self.line_width=5 # vbar / rect 
        self.line_alpha=0

        self.scaling_factor = 0.15 # volume height scaling
        self.fill_alpha = 0.05 # transparent

        # layout
        self.ncols = 1 # gridplot
        self.click_policy = "hide"
        self.location = "top_left"
        self.font_size = "8pt"

        # save figure
        self.width = 16
        self.height = 9
        self.dpi = 300
        self.tight = True

        self.sharex = None


class Plot(with_metaclass(MetaParams, object)):
    params = (('scheme', PlotScheme()),)

    def __init__(self):
        
        self.figures = list()
        self.bt_tooltips = list() # tooltips for hovertool
        self.bt_renderers = list() # render line for hovertool

        self.shared_x = None 
        self.datasource = None

    def plotdata(self, names, candle):
        dmaster = self.datasource[names[0]] # Feed
        strat_src = self.datasource[names[-1]] # Strategy
        dmaster, _tooltip = merge_cds(dmaster, strat_src)  
        self.bt_tooltips.append(_tooltip)
        # self.sharex = Range1d(data.index[0], data.index[-1]) # 同步x轴
        
        p_main = figure(
            width=self.p.scheme.figure_width,
            height=self.p.scheme.figure_height[0],
            title="Backtest",
            x_axis_type="datetime", # "linear",
            tools="pan,wheel_zoom,box_zoom,reset,save", # 基础工具
            # active_drag='pan',
            # active_scroll='wheel_zoom',
            tooltips=None
        )

        # ------------------------------------------------------------- volume -----------------------------------------------------------            

        dmaster.data['color'] = np.where(
            dmaster.data['close'] >= dmaster.data['open'],
            'green',
            'red'
        )
        
        # normalize
        volume_max = max(dmaster.data['volume'])
        price_min = min(dmaster.data['low'])
        
        # scale volume to suitable height
        scaling_factor = self.p.scheme.scaling_factor
        price_range = max(dmaster.data['high']) - price_min
        volume_height = price_range * scaling_factor
        
        dmaster.data['volume_scaled'] = price_min + (dmaster.data['volume'] / volume_max) * volume_height
        
        # add volume hist
        p_main.vbar(
            x='datetime',
            top='volume_scaled',
            bottom=price_min,
            width=self.p.scheme.vbar_width,
            source=dmaster,
            # fill_color='volume_color',
            line_color='line_color',
            line_width=self.p.scheme.line_width,
            fill_alpha=0.5,
            legend_label="Volume",
        )
        
        # ------------------------------------------------------------- price -----------------------------------------------------------            
        
        close_line = p_main.line(
                            "datetime", "close", 
                            source=dmaster,
                            line_width=2, 
                            color=tableau20[0], 
                            legend_label="close"
                            )

        self.bt_renderers.append(close_line)

        # set color
        dmaster.data['color'] = np.where(
            dmaster.data['close'] >= dmaster.data['open'],
            '#FF5252',   # 'red'  # 亮红色
            '#00C853',  # 'green' # 亮绿色
        )
        
        dmaster.data['line_color'] = np.where(
            dmaster.data['close'] >= dmaster.data['open'],
            '#D32F2F',   # darkred 
            '#009624',  # darkgreen
        )
        
        # ------------------------------------------------------------- price candle -----------------------------------------------------------            
        if candle:
            dmaster.data['top_body'] = np.maximum(dmaster.data['open'], dmaster.data['close'])
            dmaster.data['bottom_body'] = np.minimum(dmaster.data['open'], dmaster.data['close'])

            dmaster.data['body_height'] = dmaster.data['top_body'] - dmaster.data['bottom_body']
            dmaster.data['upper_shadow'] = dmaster.data['high'] - dmaster.data['top_body']
            dmaster.data['lower_shadow'] = dmaster.data['bottom_body'] - dmaster.data['low']
    
            p_main.segment(x0='datetime', y0='top_body', x1='datetime', y1='high', 
                    color='line_color', line_width=1.5, source=dmaster, legend_label="上影线")

            p_main.segment(x0='datetime', y0='bottom_body', x1='datetime', y1='low', 
                    color='line_color', line_width=1.5, source=dmaster, legend_label="下影线")

            p_main.vbar(
                x='datetime', 
                width=self.p.scheme.vbar_width, # self.p.scheme.vbar_width * 100  # 增加宽度，比如2倍
                top='top_body',
                bottom='bottom_body',
                source=dmaster,
                fill_color='color',
                # fill_alpha=0.7,
                line_color="line_color",
                line_width=self.p.scheme.line_width,
                legend_label="实体",
            )
        
        # ------------------------------------------------------------- strategy -----------------------------------------------------------            
    
        price_range = (dmaster.data['high'] - dmaster.data['low']).mean()
        dmaster.data['buy_price'] = dmaster.data['low'] - price_range * 0.05
        dmaster.data['sell_price'] = dmaster.data['high'] + price_range * 0.05

        b_mask = dmaster.data["buy"] > 0.0
        bview = CDSView(name="buy_filter", filter=BooleanFilter(b_mask))

        p_main.scatter("datetime",  "buy_price", 
                marker="triangle", 
                source=dmaster, 
                view=bview, 
                size=15, 
                color="firebrick", 
                alpha=0.6, 
                legend_label="正三角形",
                )

        s_mask = dmaster.data["sell"] < -0.0
        sview = CDSView(name="sell_filter", filter=BooleanFilter(s_mask))

        p_main.scatter("datetime", "sell_price", 
                marker="inverted_triangle", 
                source=dmaster, 
                view=sview, 
                size=15, 
                color="navy", 
                alpha=0.6, 
                legend_label="倒三角形",
                )

        p_main.legend.location = "top_left"
        self.figures.append(p_main)

# ------------------------------------------------------------------------ indicator ----------------------------------------------------------------

    def plotind(self, n_inds):
        subind_src = [self.datasource[name] for name in n_inds]
        ind_src, ind_tooltip = merge_cds(*subind_src)

        self.bt_tooltips.append(ind_tooltip)


        p_indicator = figure(
            title="Indicator",
            width=self.p.scheme.figure_width, 
            height=int(self.p.scheme.figure_height[1] * 1.2), # avoid y overlap
            x_axis_type="datetime",
            x_range=self.figures[0].x_range,  
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
    
        p_indicator.yaxis.visible = False
        p_indicator.yaxis.axis_label = ""

        for color_idx, ind_col in enumerate(ind_src.column_names):
            if ind_col != "datetime":

                ind_range = ind_src.data[ind_col]

                p_indicator.extra_y_ranges[ind_col] = Range1d( # multi-y_axis
                    start=min(ind_range), 
                    end=max(ind_range)
                )

                ind_line = p_indicator.line(
                    'datetime', ind_col, source=ind_src, line_width=2, color=tableau20[color_idx], legend_label=ind_col)

                y_axis = LinearAxis(
                    y_range_name=ind_col, 
                    axis_label=ind_col, 
                    axis_label_text_font_size=self.p.scheme.font_size, 
                    major_label_text_font_size=self.p.scheme.font_size)   # 设置刻度标签字体大小
                # y_axis.bounds = (color_idx/nbins, (color_idx+1)/nbins)  # 从中间到顶部
                p_indicator.add_layout(y_axis, 'left' if color_idx % 2 == 0 else 'right')

        self.bt_renderers.append(ind_line) # self.bt_renderers.append(ind_renders) # keep main indicator

        self.figures.append(p_indicator)


# --------------------------------------------------------------------- observe gridplots ----------------------------------------------------------------------
    
    def plotobs(self, n_obs):# or reuse indicator in one figure

        p_observers = []

        for i, name in enumerate(n_obs):
            # obs_src = self.datasource[name]
            obs_src = on_shift(self.datasource[name], shift=-1) # 

            title = "" if i > 0 else "Observer" # f'Observer: {name.split("(")[0]}',

            p_obs = figure(
                title = title,
                width=self.p.scheme.figure_width, 
                height=self.p.scheme.figure_height[2],
                x_axis_type="datetime",
                x_range=self.figures[0].x_range,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )

            obs_tooltip = [("Date", "@datetime{%F}")]

            for idx, col in enumerate(obs_src.column_names):
                if col != "datetime":
                    color_idx = idx % len(tableau20)
                    obs_line = p_obs.line('datetime', col, source=obs_src, line_width=2, color=tableau20[color_idx], legend_label=col)

            chain_tooltip = [f" {key}: @{{{key}}}{{0.2f}}<br>" for key in obs_src.column_names if  key != "datetime"]
            obs_tooltip.insert(0, (name, ''.join(chain_tooltip)))

            self.bt_tooltips.append(obs_tooltip)
 
            if i < len(n_obs) - 1:
                p_obs.xaxis.visible = False
            p_observers.append(p_obs)

            self.bt_renderers.append(obs_line)

        self.figures.extend(p_observers)

# --------------------------------------------------------------------------- plt ----------------------------------------------------------------------

    def show(self):

        # add vline
        _vlines = []
        for _plt in self.figures:
            vline = Span(location=0, dimension='height', line_color='red', line_width=1, line_alpha=0)
            _plt.add_layout(vline)
            _vlines.append(vline)
    
        # set HoverTool and CrosshairTool
        hover_callback = CustomJS(
            args=dict(vlines=_vlines), 
            code="""
                // 悬停时显示所有参考线
                for (let i = 0; i < vlines.length; i++) {
                    vlines[i].line_alpha = 0.5;
                    vlines[i].location = cb_data.geometry.x;
                }
            """
        )
        
        for idx, _plt in enumerate(self.figures):
            _tooltip = self.bt_tooltips[idx]

            figure_renders = self.bt_renderers[idx]
            renderers = figure_renders if isinstance(figure_renders, list) else [figure_renders]

            hover = HoverTool(
                tooltips=_tooltip,
                formatters={
                    '@datetime': 'datetime'
                },
                mode='vline', 
                callback=hover_callback,
                renderers = renderers,
                point_policy="follow_mouse"
            )

            crosshair = CrosshairTool(
                dimensions='both',
                line_alpha=0.5,
                line_color='black',
                line_width=1)

            _plt.add_tools(hover, crosshair)
            _plt.legend.click_policy = self.p.scheme.click_policy
            _plt.legend.location = self.p.scheme.location
            _plt.background_fill_alpha = self.p.scheme.fill_alpha
            _plt.border_fill_alpha = self.p.scheme.fill_alpha    
            _plt.yaxis.formatter = NumeralTickFormatter(format="0,0.00")  # 保留两位小数
 
        # plot gridplot
        grid = gridplot(self.figures, ncols=self.p.scheme.ncols, sizing_mode='stretch_width') # layout = column(bt_plts)
        grid.toolbar.active_drag=PanTool()
        grid.toolbar.active_scroll=WheelZoomTool()

        show(grid)
        return grid

    def plot(self, num_data: int, num_ind: int, num_obs: int, source:str, freq: str="D", candle=False, save=False, filename=None):
        self.datasource, _names = create_datasource(source, freq)

        data_names = _names[:num_data+1] # feed + dataclone(s) + strategy
        self.plotdata(data_names, candle=candle)

        ind_names = _names[num_data+1: num_ind+num_data+1]
        self.plotind(ind_names) # Indicator

        obs_names = _names[-num_obs:]
        self.plotobs(obs_names) # Observer

        grid_plot = self.show()
        
        if save:
            self.savefig(grid_plot, filename)

    def savefig(self, fig, filename):
        output_file(f"{filename}.html")
        save(grid)

        # fig.set_size_inches(self.p.scheme.width, self.p.scheme.height)
        # bbox_inches = 'tight' * self.p.scheme.tight or None
        # fig.savefig(filename, dpi=self.p.scheme.dpi, bbox_inches=bbox_inches) 
        # # 导出为 PNG，设置分辨率
        # for e_p in show_p:
        #     export_png(e_p, filename=f"{filename}.png", scale_factor=1, 
        #             width=self.p.scheme.width, height=self.p.scheme.heigh) # export_svg / need selenium driver

