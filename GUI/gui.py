#!/usr/bin/env python

import sys
try:
	import pygtk
	pygtk.require("2.0")
except:
	pass
try:
	import gtk
	import gtk.glade
except:
	sys.exit(1)
import re
handler_re = re.compile(r'(on|after)_(.*)_(.*)')


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
		response = dialog.run()
		if response == gtk.RESPONSE_OK:
			self.filename= dialog.get_filename()
			print self.filename, ' selected.'
		elif response == gtk.RESPONSE_CANCEL:
			print 'Closed, no files selected'
		dialog.destroy()
		self.button1.modify_text(self.filename.split('.')[-1])

if __name__ == "__main__":
	hwg = GladeWidget('gui.glade', 'window1')
	gtk.main()