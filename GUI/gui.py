#!/usr/bin/env python
from __future__ import division
import sys,os, os.path, PyMC
import PyMC.database
try:
    import pygtk
    pygtk.require("2.0")
    # Note for OS X: You can get pygtk from the MacPort py-gtk (Python 2.4) or py25-gtk (Python 2.5)...
    # but the MacPorts py-gobject is buggy. I reported it on their trac page.
except:
    pass
try:
    import gtk, gobject
    import gtk.glade
except:
    sys.exit(1)
import re
handler_re = re.compile(r'(on|after)_(.*)_(.*)')
from threading import Thread

def progress_timeout(self):
    # Calculate the value of the progress bar
    if self.sampler.status == 'running':
        self.pbar.set_fraction(self.sampler._current_iter/self.sampler._iter)
    elif self.sampler.status == 'ready':
        self.pbar.set_fraction(0.)
        self.button2.set_label('Start')
        self.button2.set_image(gtk.image_new_from_stock('gtk-yes', gtk.ICON_SIZE_BUTTON))
        return False

    else:    # Sampling is interrupted.
        return False
    # As this is a timeout function, return TRUE so that it
    # continues to get called
    return True



class GladeWidget:
    def __init__(self, glade_file, widget_name):
        """Connects signal handling methods to Glade widgets.

        Methods named like on_widget__signal or after_widget__signal
        are connected to the appropriate widgets and signals.
        """
        get_widget = gtk.glade.XML(glade_file, widget_name).get_widget
        W = {}
        for attr in dir(self):
            match = handler_re.match(attr)
            if match:
                when, widget, signal = match.groups()
                method = getattr(self, attr)
                assert callable(method)
                if when == 'on':
                    get_widget(widget).connect(signal, method)
                elif when == 'after':
                    get_widget(widget).connect_after(signal, method)
                W[widget]=get_widget(widget)
            elif attr.startswith('on_') or attr.startswith('after_'):
                # Warn about some possible typos like separating
                # widget and signal name with _ instead of __.
                print ('Warning: attribute %r not connected'
                       ' as a signal handler' % (attr,))
        self.__dict__.update(W)
        self.get_widget = get_widget

        for db in PyMC.database.available_modules:
            self.combobox1.append_text(db)
            self.combobox1.set_active(0)

    def on_window1_destroy(self, widget):
        gtk.main_quit()

    def on_button1_clicked(self, widget):
        dialog = gtk.FileChooserDialog("Open python module",
            None,
            gtk.FILE_CHOOSER_ACTION_OPEN,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
            gtk.STOCK_OPEN, gtk.RESPONSE_OK))
        dialog.set_default_response(gtk.RESPONSE_OK)

        filter = gtk.FileFilter()
        filter.set_name("Python files")
        filter.add_pattern("*.py")
        dialog.add_filter(filter)
        dialog.set_filename('ExtremeRainfall.py')
        response = dialog.run()
        if response == gtk.RESPONSE_OK:
            self.filename= dialog.get_filename()
            self.modulename = os.path.splitext(os.path.basename(self.filename))[0]
            sys.path.append(os.path.dirname(self.filename))
            mod = __import__(self.modulename)
            db = self.combobox1.get_active_text()
            self.sampler = PyMC.Sampler(mod, db=db)

        elif response == gtk.RESPONSE_CANCEL:
            print 'Closed, no files selected'
        dialog.destroy()
        self.button1.set_label(self.modulename)

    def on_spinbuttonit_changed(self, widget):
        pass

    def on_spinbuttonburn_changed(self, widget):
        pass

    def on_spinbuttonthin_changed(self, widget):
        pass

    def on_combobox1_changed(self, widget):
        """Close last database and assign new one."""
        db = widget.get_active_text()
        try:
            self.sampler.db._finalize()
            self.sampler._assign_database_backend(db)
        except AttributeError:
            pass

    def on_button2_clicked(self, widget):
        self.pbar = self.get_widget('progressbar1')
        # Not started, not sampling
        if self.sampler.status =='ready':
            self.iter = int(self.spinbuttonit.get_value())
            self.burn = int(self.spinbuttonburn.get_value())
            self.thin = int(self.spinbuttonthin.get_value())
            #self.pbar.set_fraction(0.0)

            self.T = Thread(target=self.sampler.sample, args=(self.iter, self.burn, self.thin))
            self.T.start()

            self.timer = Thread(target=gobject.timeout_add, args= (100, progress_timeout, self))
            self.timer.start()

            # Change label to stop
            widget.set_label('Stop')
            widget.set_image(gtk.image_new_from_stock('gtk-stop', gtk.ICON_SIZE_BUTTON))

        elif self.sampler.status == 'running':
            self.sampler.status = 'paused'
            widget.set_label('Continue')
            widget.set_image(gtk.image_new_from_stock('gtk-yes', gtk.ICON_SIZE_BUTTON))

        elif self.sampler.status == 'paused':
            self.sampler.interactive_continue()

            widget.set_label('Stop')
            widget.set_image(gtk.image_new_from_stock('gtk-stop', gtk.ICON_SIZE_BUTTON))

            self.timer = Thread(target=gobject.timeout_add, args= (100, progress_timeout, self))
            self.timer.start()

if __name__ == "__main__":
    hwg = GladeWidget('gui.glade', 'window1')
    gtk.gdk.threads_init()
    gtk.main()
