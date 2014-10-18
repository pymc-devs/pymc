try:
    import IPython
    from IPython.html import widgets
    from IPython.core import display
    from IPython.utils.traitlets import Unicode, Integer, Float
    import json
    from numpy.random import seed
    import time
    from .sampling import iter_sample
except ImportError:
    IPython = False

_no_notebook_error_message = "nbsample can only be run inside IPython Notebook."
if IPython:
    __all__ = ['nbsample']

    _javascript = """<script type="text/javascript">
        require(["widgets/js/widget"], function(WidgetManager){
        var ISampleWidget =  IPython.WidgetView.extend({
            render: function(){
                var html = $("<table style='width:100%;'><tr><td style='width:60px'><button>Stop</button></td>"+
                             "<td class='pymc-clock' style='width:60px'></td>"+
                             "<td class='pymc-progress'>"+
                             "<div class='bar' style='width:0px; height: 20px; "+
                             "background-image: linear-gradient(to bottom, #dddddd 0%,#111111 100%)"+
                             "'>&nbsp;</div></td>"+
                             "<td class='pymc-current-samples' style='width:60px;'>0</td>"+
                             "<td style='width:10px;'>/</td>"+
                             "<td style='width:60px;' class='pymc-max-samples'></td>"+
                             "</tr>"+
                             "</table>");            
                this.setElement(html);
                this.$el.find("button").click($.proxy(function(){
                    this.send("stop","stop");
                    this.$el.find("button").attr("disabled", "disabled");
                }, this));
                this.model.on('change:max_samples', function(model, value){
                    this.$el.find(".pymc-max-samples").text(value);
                }, this);
                this.model.on('change:clock', function(model, value){
                    this.$el.find(".pymc-clock").text(value);
                }, this);
                this.model.on('change:current_samples', function(model, value){
                    this.$el.find(".pymc-current-samples").text(value);
                    var total_width = this.$el.find(".pymc-progress").width()-5;
                    var total_samples = this.model.get("max_samples");
                    var width = value * total_width / total_samples;
                    this.$el.find(".pymc-progress .bar").width(width)
                }, this);
            }
        });
        WidgetManager.register_widget_view('ISampleWidget', ISampleWidget)
    });
    </script>
    """

    class ISampleWidget(widgets.DOMWidget):
        _view_name = Unicode('ISampleWidget', sync=True)
        current_samples = Integer(sync=True)
        max_samples = Integer(sync=True)
        clock = Unicode(sync=True)

        def __init__(self, *args, **kwargs):
            widgets.DOMWidget.__init__(self,*args, **kwargs)
            self.iteration = 0
            self.on_msg(self._handle_custom_msg)
            self.send_state()
            self.stopped = False
        def _handle_custom_msg(self, message):
            if message == "stop":
                self.stopped = True
    
    

    def nbsample(draws, step, start=None, trace=None, chain=0, tune=None, model=None, random_seed=None):
        try:
            assert(hasattr(IPython.get_ipython(), 'comm_manager'))
        except (AssertionError, NameError, KeyError) as e:
            raise NotImplementedError(_no_notebook_error_message)
    
        display.display_html(_javascript, raw=True)
        w = ISampleWidget()
        display.display(w)
        t_start = time.time()
        t_last = time.time()

        w.max_samples = draws
        w.current_samples = 0
        for i,backend in enumerate(iter_sample(draws, step, start=start, trace=trace,
            chain=chain, tune=tune, model=model, random_seed=None), 1):
            elapsed = time.time() - t_start
            elapsed_last = time.time() - t_last

            if elapsed_last > 0.1:
                t_last = time.time()
                w.current_samples = i
                w.clock = "%02i:%02i:%02i" % (elapsed / 60 / 60, elapsed / 60 % 60, elapsed % 60)
                get_ipython().kernel.do_one_iteration()
                if w.stopped:
                    break
        w.current_samples = i
        return backend
else:
    def nbsample(*args, **kwargs):
        raise NotImplemented(_no_notebook_error_message)
