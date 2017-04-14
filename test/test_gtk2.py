
import gtk
import math
import cairo

class PyApp(gtk.Window):

		def __init__(self):
				super(PyApp, self).__init__()

				self.set_title("Simple drawing")
				self.resize(800, 600)
				self.set_position(gtk.WIN_POS_CENTER)

				self.connect("destroy", gtk.main_quit)

				darea = gtk.DrawingArea()
				darea.connect("expose-event", self.expose)
				self.add(darea)

				darea.connect("button_press_event", self.button_press_event)
				darea.set_events(gtk.gdk.EXPOSURE_MASK
					| gtk.gdk.LEAVE_NOTIFY_MASK| gtk.gdk.BUTTON_PRESS_MASK
					| gtk.gdk.POINTER_MOTION_MASK| gtk.gdk.POINTER_MOTION_HINT_MASK)

				self.show_all()
				self.fc = 0

		def button_press_event(self, widget, event):
			if event.button == 1:
				self.queue_draw()
			return True

		def expose(self, widget, event):
				print self.fc
				self.fc = self.fc+1

				cr = widget.window.cairo_create()

				s1 = cairo.ImageSurface.create_from_png('bug.png')
				cr.save()
				cr.translate(400, 0)
				cr.scale(0.3, 0.5)
				cr.set_source_surface(s1, 0, 0)
				cr.paint()
				cr.restore()

				s1 = cairo.ImageSurface.create_from_png('math_1.png')
				cr.save()
				cr.scale(0.1, 0.1)
				cr.set_source_surface(s1, 0, 0)
				cr.paint()
				cr.restore()

				s1 = cairo.ImageSurface.create_from_png('paste_1.png')
				cr.save()
				cr.translate(200, 200)
				cr.scale(0.1, 0.1)
				cr.set_source_surface(s1, 0, 0)
				cr.paint()
				cr.restore()

				cr.set_line_width(9)
				cr.set_source_rgb(0.7, 0.2, 0.0)

				w = self.allocation.width
				h = self.allocation.height

				cr.translate(w/2, h/2)
				cr.arc(0, 0, 50, 0, 2*math.pi)
				cr.stroke_preserve()

				cr.set_source_rgb(0.3, 0.4, 0.6)
				cr.fill()


PyApp()
gtk.main()